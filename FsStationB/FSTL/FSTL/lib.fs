// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module FSTL.Lib


let max (d1:double,d2:double) = 
    match (d1 > d2) with 
    | true -> d1
    | false -> d2

let min (d1:double,d2:double) = 
    match (d1 < d2) with 
    | true -> d1
    | false -> d2

