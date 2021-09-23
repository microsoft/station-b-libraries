// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Client.HandlerUtils

open Fable.SimpleHttp

// GET JSON content from a URL and decode it.
let getHandler<'T> (url:string) (jsonDecoder:string->'T) : Async<'T> =
    async {
        let! response =
            // https://docs.microsoft.com/en-us/dotnet/api/system.net.webutility.urlencode?view=net-5.0
            Http.request url // TODO Does this need to be URL Encoded?
            |> Http.method GET
            |> Http.header (Headers.contentType "application/json")
            |> Http.send

        match response.statusCode with
        | 200 ->
            return jsonDecoder response.responseText
        | _ ->
            return failwithf "Status %d => %s" response.statusCode response.responseText
    }

// GET JSON content from a URL with no parameters and decode it.
let get0Handler<'U> (url:string) (jsonDecoder:string->'U) : unit->Async<'U> =
    fun () ->
        getHandler url jsonDecoder

// GET JSON content from a URL with a single parameter and decode it.
let get1Handler<'T,'U> (parameterEncoder:'T->string) (url:PrintfFormat<string->string,unit,string,string,string>) (jsonDecoder:string->'U) : 'T->Async<'U> =
    fun (t:'T) ->
        let tt = parameterEncoder t
        getHandler (sprintf url tt) jsonDecoder

// GET JSON content from a URL with a two parameters and decode it.
let get2Handler<'T,'U,'V> (parameterEncoder1:'T->string) (parameterEncoder2:'U->string) (url:PrintfFormat<string*string->string,unit,string,string,string*string>) (jsonDecoder:string->'V) : ('T*'U)->Async<'V> =
    fun ((t,u):'T * 'U) ->
        let tt = parameterEncoder1 t
        let uu = parameterEncoder2 u
        getHandler (sprintf url (tt, uu)) jsonDecoder

// POST JSON content to a url.
let postHandler<'T> (url:string) (jsonEncoder:'T->string) : 'T -> Async<unit> =
    fun (o: 'T) ->
        async {
            let! response =
                Http.request url
                |> Http.method POST
                |> Http.header (Headers.contentType "application/json")
                |> Http.content (BodyContent.Text (jsonEncoder o))
                |> Http.send

            match response.statusCode with
            | 200 ->
                return ()
            | _ ->
                return failwithf "Status %d => %s" response.statusCode response.responseText
        }

// POST JSON content to a url with a single parameter.
let post1Handler<'T,'U> (parameterEncoder:'T->string) (url:PrintfFormat<string->string,unit,string,string,string>) (jsonEncoder:'U->string) : 'T->'U->Async<unit> =
    fun (t:'T) (u:'U) ->
        let tt = parameterEncoder t
        postHandler (sprintf url tt) jsonEncoder u

// POST JSON content to a url and get JSON response
let postGetHandler<'T, 'U> (url:string) (jsonEncoder:'T->string) (jsonDecoder:string->'U) : 'T -> Async<'U> =
    fun (o: 'T) ->
        async {
            let! response =
                Http.request url
                |> Http.method POST
                |> Http.header (Headers.contentType "application/json")
                |> Http.content (BodyContent.Text (jsonEncoder o))
                |> Http.send

            match response.statusCode with
            | 200 ->
                return jsonDecoder response.responseText
            | _ ->
                return failwithf "Status %d => %s" response.statusCode response.responseText
        }

// POST JSON content to a url with a single parameter and get JSON response
let post1GetHandler<'T,'U,'V> (parameterEncoder:'T->string) (url:PrintfFormat<string->string,unit,string,string,string>) (jsonEncoder:'U->string) (jsonDecoder:string->'V) : 'T->'U->Async<'V> =
    fun (t:'T) (u:'U) ->
        let tt = parameterEncoder t
        postGetHandler (sprintf url tt) jsonEncoder jsonDecoder u

