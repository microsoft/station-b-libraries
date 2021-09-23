// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module PlateReads

open System.IO

type Measurement = 
    { dateTime : System.DateTime
      name : string
      content : unit -> string }