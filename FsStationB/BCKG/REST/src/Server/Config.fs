// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Server.Config

open BCKG_REST_Server.Shared.Shared

type IBCKGApiConfig =
    abstract GetApi : userId:string -> IBCKGApi
