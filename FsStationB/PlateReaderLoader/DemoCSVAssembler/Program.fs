// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
open System

open Argu

open Tagging

type DemoCSVAssemblerArguments =
    | ConnectionString of string
    | Local
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | ConnectionString _ -> "Backend connection string"
            | Local _ -> "Use a local Azure emulator"

let R33S175ExrepTet33AAVLac300ND_IPTGATC_titration connectionString =
    //This is an example of how you can specify a range. It's ideally replaced with something more explicit on
    //a given PC attached to the plate reader when someone is starting the experiment
    let makeDateTime asString =
        DateTime.ParseExact(asString, "yyyy-MM-dd HH:mm", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None)

    let from = makeDateTime "2018-11-22 14:50"
    let until = makeDateTime "2018-11-23 11:40"

    let uniformFiles = BCKG.Storage.ListPlateReaderUniformFileBetween connectionString from until

    let reads =
        uniformFiles
        |> Seq.map (fun uniformFile ->
            { PlateReads.Measurement.dateTime = uniformFile.dateTime
              PlateReads.Measurement.name = uniformFile.fullFileName
              PlateReads.Measurement.content = fun () -> BCKG.Storage.LoadPlateReaderUniformFile connectionString uniformFile.fullFileName }
            )
        |> Seq.toArray

    //This tagging process is intended to be replaced.
    ResetPlate()

    SerialDilution "B2" "B6" "IPTG" 1.0 3.0
    SerialDilution "C2" "C6" "IPTG" 1.0 3.0
    SerialDilution "D2" "D6" "IPTG" 1.0 3.0
    SerialDilution "E2" "E6" "IPTG" 1.0 3.0
    SerialDilution "F2" "F6" "IPTG" 1.0 3.0

    SerialDilution "B2" "F2" "C6" 10000.0 5.0
    SerialDilution "B3" "F3" "C6" 10000.0 5.0
    SerialDilution "B4" "F4" "C6" 10000.0 5.0
    SerialDilution "B5" "F5" "C6" 10000.0 5.0
    SerialDilution "B6" "F6" "C6" 10000.0 5.0

    SerialDilution "B6" "B11" "ATC" 200.0 2.0
    SerialDilution "C6" "C11" "ATC" 200.0 2.0
    SerialDilution "D6" "D11" "ATC" 200.0 2.0
    SerialDilution "E6" "E11" "ATC" 200.0 2.0
    SerialDilution "F6" "F11" "ATC" 200.0 2.0

    SerialDilution "B7" "F7" "C12" 10000.0 5.0
    SerialDilution "B8" "F8" "C12" 10000.0 5.0
    SerialDilution "B9" "F9" "C12" 10000.0 5.0
    SerialDilution "B10" "F10" "C12" 10000.0 5.0
    SerialDilution "B11" "F11" "C12" 10000.0 5.0

    Genotype "B2" "F11" "R33S175ExRepTet33AAVLac300ND"
    Conditions "B2" "F6" "ATC=200.0"
    Conditions "B2" "F6" "C12=10000.0"
    Conditions "B7" "F11" "IPTG=1.0"
    Conditions "B7" "F11" "C6=10000.0"

    let spec = PlateToCSVArray()
    CSVUtil.toCSV reads spec

[<EntryPoint>]
let main argv =
    
    //Maybe during development should just default to emulator, or auto fetch credentials by keyvault?
    let parser = ArgumentParser.Create<DemoCSVAssemblerArguments>()
    let parserResults = parser.Parse(argv)
    
    let connectionString =
        if parserResults.Contains Local then @"UseDevelopmentStorage=true"
        else
            parserResults.GetResult ConnectionString
    if connectionString = "%BCKG_CONNECTION_STRING%" || connectionString = "" then
        failwith "Connection string not set correctly. Try setting %BCKG_CONNECTION_STRING%"

    printf "WORKING..."
    let asCSV = R33S175ExrepTet33AAVLac300ND_IPTGATC_titration connectionString

    System.IO.File.WriteAllText(@"test.csv", asCSV)
    printfn "done"
    0
