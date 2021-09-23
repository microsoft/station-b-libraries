// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Client.Index

open Elmish
open Fable.React
open Fable.React.Props
open Fulma

open BCKG_REST_Server.Shared.Shared

type Model =
    { api : IBCKGApi
      Errors: string list }

type Msg =
    | GotError of exn

let init (api : IBCKGApi) : unit -> Model * Cmd<Msg> = fun _ ->
    let model =
        { api = api
          Errors = [] }
    let cmd = Cmd.none
    model, cmd

let update (msg: Msg) (model: Model): Model * Cmd<Msg> =
    match msg with
    | GotError ex ->
        { model with Errors = ex.Message :: model.Errors }, Cmd.none

let navBrand =
    Navbar.Brand.div [ ] [
        Navbar.Item.a [
            Navbar.Item.Props [ Href "https://safe-stack.github.io/" ]
            Navbar.Item.IsActive true
        ] [
            img [
                Src "/favicon.png"
                Alt "Logo"
            ]
        ]
    ]

let containerBox (model : Model) (dispatch : Msg -> unit) =
    Box.box' [ ] [
        Content.content [ ] [
            for msg in model.Errors do
                Notification.notification [
                    Notification.Color Color.IsDanger ] [
                        p [] [ str msg ]
                ]
        ]
    ]

let view (model : Model) (dispatch : Msg -> unit) =
    Hero.hero [
        Hero.Color IsPrimary
        Hero.IsFullHeight
        Hero.Props [
            Style [
                Background """linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://unsplash.it/1200/900?random") no-repeat center center fixed"""
                BackgroundSize "cover"
            ]
        ]
    ] [
        Hero.head [ ] [
            Navbar.navbar [ ] [
                Container.container [ ] [ navBrand ]
            ]
        ]

        Hero.body [ ] [
            Container.container [ ] [
                Column.column [
                    Column.Width (Screen.All, Column.Is6)
                    Column.Offset (Screen.All, Column.Is3)
                ] [
                    Heading.p [ Heading.Modifiers [ Modifier.TextAlignment (Screen.All, TextAlignment.Centered) ] ] [ str "BCKG_REST_Server" ]
                    containerBox model dispatch
                ]
            ]
        ]
    ]
