// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Server.Server

open Argu
open System
open FSharp.Control.Tasks.V2.ContextInsensitive
open Giraffe
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging

open BCKG_REST_Server.Shared.Shared

open Config
open Storage

let requestHeaderUserId = "x-ms-client-principal-name"

let lookupAPI (ctx:HttpContext) : IBCKGApi =
    let apiConfig = ctx.GetService<IBCKGApiConfig>()
    let userId = ctx.TryGetRequestHeader requestHeaderUserId |> Option.defaultValue "local user"

    apiConfig.GetApi userId

type BCKG_REST_ServerArguments =
    | Local
    | Port of int
    | ConnectionString of string
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Local _ -> "Use a local Azure emulator"
            | Port _ -> "Port to run the server on"
            | ConnectionString _ -> "Backend connection string"

// Interim whitelist, move to active directory based roles someday
//
// https://juselius.github.io/posts/2019-12-03-fsharp-authentication-oid.html
//
let permittedUsers =
    set []

//Authentication is primarily based on AppService authentication, we additional
//do a lightweight, non-secure restriction to a whitelist. This is to reduce chance of
//accidental use by unauthorized users.
let checkPermitted (requestHeaderUserId:string) (permittedUsers:Set<string>) : HttpHandler =
    fun (next : HttpFunc) (ctx : HttpContext) ->
    task {
        let hostString = ctx.Request.Host
        if hostString.Host = "localhost" then
            return None
        else
            match ctx.TryGetRequestHeader requestHeaderUserId with
            | Some username ->
                if permittedUsers.Contains username then
                    return None //Continue
                else
                    return! text (sprintf "%s not permitted" username) next ctx
            | None ->
                return! text "Unable to identify user" next ctx
    }

// ---------------------------------
// Error handler
// ---------------------------------

let errorHandler (ex : Exception) (logger : ILogger) =
    match ex with
    | :? Microsoft.WindowsAzure.Storage.StorageException as dbEx ->
        let msg = sprintf "An unhandled Windows Azure Storage exception has occured: %s" dbEx.Message
        logger.LogError(EventId(), dbEx, "An error has occured when hitting the database.")
        ServerErrors.SERVICE_UNAVAILABLE msg
    | _ ->
        logger.LogError(EventId(), ex, "An unhandled exception has occurred while executing the request.")
        clearResponse >=> ServerErrors.INTERNAL_ERROR ex.Message

// ---------------------------------
// Config and Main
// ---------------------------------

let configureApp (webApp:HttpHandler) = fun (app : IApplicationBuilder) ->
    app
        .UseGiraffeErrorHandler(errorHandler)
        .UseGiraffe(webApp)

let configureServices (bckgApiConfig:IBCKGApiConfig) = fun (services : IServiceCollection) ->
    services
        .AddGiraffe()
        .AddSingleton bckgApiConfig
        |> ignore

let configureLogging (builder : ILoggingBuilder) =
    builder
        // .AddFilter(fun l -> l.Equals LogLevel.Error)
        .AddConsole()
        .AddDebug()
        |> ignore

[<EntryPoint>]
let main args =
    let parser = ArgumentParser.Create<BCKG_REST_ServerArguments>()
    let parserResults = parser.Parse(args)
    let port = parserResults.TryGetResult Port |> Option.defaultValue 8081

    let connectionString =
        if parserResults.Contains Local then
            @"UseDevelopmentStorage=true"
        else
            parserResults.GetResult ConnectionString

    let webApp = RouteTable.routeTable (checkPermitted requestHeaderUserId permittedUsers) lookupAPI
    let apiConfig = BCKGApiConfig(connectionString)

    try
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(
                fun webHostBuilder ->
                    webHostBuilder
                        .Configure(configureApp webApp)
                        .ConfigureServices(configureServices apiConfig)
                        .ConfigureLogging(configureLogging)
                        .UseUrls("http://0.0.0.0:" + port.ToString())
                        |> ignore)
            .Build()
            .Run()
        0
    with exn ->
        let color = Console.ForegroundColor
        Console.ForegroundColor <- System.ConsoleColor.Red
        Console.WriteLine(exn.Message)
        Console.ForegroundColor <- color
        1
