// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Server.HandlerUtils

open System.IO
open FSharp.Control.Tasks.V2.ContextInsensitive
open Giraffe
open Microsoft.AspNetCore.Http

open BCKG_REST_Server.Shared.Shared

let writeJson<'T> (encoder:'T->string) (o:'T) : HttpHandler =
    fun (_ : HttpFunc) (ctx : HttpContext) ->
        ctx.SetContentType "application/json; charset=utf-8"
        ctx.WriteStringAsync (encoder o)

let get0Handler<'U> (lookupAPI:HttpContext->IBCKGApi) (api:IBCKGApi->unit->Async<'U>) (encoder:'U->string) =
    fun (next : HttpFunc) (ctx : HttpContext) ->
        task {
            let f = lookupAPI ctx |> api
            let! u = f()
            return! writeJson encoder u next ctx
        }

let get1Handler<'T, 'U> (lookupAPI:HttpContext->IBCKGApi) (param_decoder:string->Option<'T>) (api:IBCKGApi->'T->Async<'U>) (encoder:'U->string) =
    fun (s:string) ->
        fun (next : HttpFunc) (ctx : HttpContext) ->
            task {
                let f = lookupAPI ctx |> api
                let t = param_decoder s
                match t with
                | Some tt ->
                    let! u = f tt
                    return! writeJson encoder u next ctx
                | None -> return failwith "Invalid parameter"
            }

let get2Handler<'T, 'U, 'V> (lookupAPI:HttpContext->IBCKGApi) (param1_decoder:string->Option<'T>) (param2_decoder:string->Option<'U>) (api:IBCKGApi->('T*'U)->Async<'V>) (encoder:'V->string) =
    fun ((s1,s2):string*string) ->
        fun (next : HttpFunc) (ctx : HttpContext) ->
            task {
                let f = lookupAPI ctx |> api
                let t1 = param1_decoder s1
                match t1 with
                | Some tt1 ->
                    let t2 = param2_decoder s2
                    match t2 with
                    | Some tt2 ->
                        let! u = f (tt1, tt2)
                        return! writeJson encoder u next ctx
                    | None -> return failwith "Invalid second parameter"
                | None -> return failwith "Invalid first parameter"
            }

let post0Handler<'U> (lookupAPI:HttpContext->IBCKGApi) (body_decoder:string->'U) (api:IBCKGApi->'U->Async<unit>) =
    fun (next : HttpFunc) (ctx : HttpContext) ->
        task {
            let f = lookupAPI ctx |> api
            use stream = new StreamReader(ctx.Request.Body)
            let! body = stream.ReadToEndAsync()
            let u = body_decoder body
            do! f u
            return! Successful.OK "" next ctx
        }

let post1Handler<'T, 'U> (lookupAPI:HttpContext->IBCKGApi) (param_decoder:string->Option<'T>) (body_decoder:string->'U) (api:IBCKGApi->'T->'U->Async<unit>) =
    fun (s:string) ->
        fun (next : HttpFunc) (ctx : HttpContext) ->
            task {
                let f = lookupAPI ctx |> api
                let t = param_decoder s
                use stream = new StreamReader(ctx.Request.Body)
                let! body = stream.ReadToEndAsync()
                let u = body_decoder body
                match t with
                | Some tt ->
                    do! f tt u
                    return! Successful.OK "" next ctx
                | None -> return failwith "Invalid parameter"
            }

let post0GetHandler<'U, 'V> (lookupAPI:HttpContext->IBCKGApi) (body_decoder:string->'U) (api:IBCKGApi->'U->Async<'V>) (encoder:'V->string) =
    fun (next : HttpFunc) (ctx : HttpContext) ->
        task {
            let f = lookupAPI ctx |> api
            use stream = new StreamReader(ctx.Request.Body)
            let! body = stream.ReadToEndAsync()
            let u = body_decoder body
            let! v = f u
            return! writeJson encoder v next ctx
        }

let post1GetHandler<'T, 'U, 'V> (lookupAPI:HttpContext->IBCKGApi) (param_decoder:string->Option<'T>) (body_decoder:string->'U) (api:IBCKGApi->'T->'U->Async<'V>) (encoder:'V->string) =
    fun (s:string) ->
        fun (next : HttpFunc) (ctx : HttpContext) ->
            task {
                let f = lookupAPI ctx |> api
                let t = param_decoder s
                use stream = new StreamReader(ctx.Request.Body)
                let! body = stream.ReadToEndAsync()
                let u = body_decoder body
                match t with
                | Some tt ->
                    let! v = f tt u
                    return! writeJson encoder v next ctx
                | None -> return failwith "Invalid parameter"
            }
