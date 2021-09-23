// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module CSVUtil

open System.IO
open BCKG

let toCSV (experiment:PlateReads.Measurement[]) spec =

    let parsedBatch =
        experiment
        |> Seq.map(fun measurement -> measurement.content())
        |> Seq.map(PlateReaderFileParser.parseFrom)
        |> Seq.toArray

    let rows = parsedBatch.[0].Data.[0].Length
    let cols = parsedBatch.[0].Data.[0].[0].Length

    let fluorescenceReads = parsedBatch |> Seq.filter (fun read -> read.Configuration = PlateReaderFileParser.Configuration.Fluorescence) |> Array.ofSeq
    let absorbanceReads = parsedBatch |> Seq.filter (fun read -> read.Configuration = PlateReaderFileParser.Configuration.Absorbance) |> Array.ofSeq

    let fluorescenceDataCount = fluorescenceReads.[0].Data |> Array.length
    let absorbanceDataCount = absorbanceReads.[0].Data |> Array.length

    let fluorescenceFilterCount = fluorescenceReads.[0].Filters.Value |> Array.length //Strictly these two could change between measurements
    let absorbanceWavelengthCount = absorbanceReads.[0].Wavelengths.Value |> Array.length

    (* Produce data for characterisation *)

    let wellData_TimeRows =
        Array.init (cols * rows) (fun colrow ->
            let row = colrow / cols
            let col = colrow % cols

            let fluorescenceData =
                seq {0..(fluorescenceDataCount-1)}
                |> Seq.collect (fun fluorescenceIndex ->
                    fluorescenceReads
                    |> Seq.map (fun measurements -> float measurements.Data.[fluorescenceIndex].[row].[col].Value)
                    )

            let aborbanceData =
                seq {0..(absorbanceDataCount-1)}
                |> Seq.collect (fun aborbanceIndex ->
                    absorbanceReads
                    |> Seq.map (fun measurements -> (float measurements.Data.[aborbanceIndex].[row].[col].Value)/1000.0)
                    )

            Seq.append fluorescenceData aborbanceData |> Array.ofSeq
        )

    let timesCount = Array.length parsedBatch / 2

    //So that characterisation works, maybe declare this somewhere else.
    let filterToFP =
        [("430-10/480-10","ECFP");
         ("500-10/530", "EYFP");
         ("550-10/610-20", "mRFP1");
         ("485-12/520", "GFP");
         ("485-12/530", "GFP530")
        ] |> Map.ofList

    let wavelengthToOD =
        [("600nm","OD");
         ("700nm", "OD700")]
        |> Map.ofList

    let header =
        [fluorescenceReads.[0].Filters.Value |> Array.map (fun filter -> match Map.tryFind filter filterToFP with Some x -> x | None -> failwithf "Unknown filter: %s" filter);
         absorbanceReads.[0].Wavelengths.Value |> Array.map (fun wavelength -> match Map.tryFind wavelength wavelengthToOD with Some x -> x | None -> failwithf "Unknown wavelength: %s" wavelength)]
        |> Seq.concat
        |> Seq.collect (Seq.replicate timesCount)
        |> String.concat ","

    let startDateTime = parsedBatch.[0].ReadingDateTime

    let timesFluorescence =
        fluorescenceReads
        |> Seq.map (fun measurement -> (measurement.ReadingDateTime - startDateTime).TotalHours)
        |> Seq.replicate fluorescenceFilterCount
        |> Seq.concat
        |> Seq.cache

    (*let timesAborbance =
        absorbanceReads
        |> Seq.map (fun measurement -> (measurement.ReadingDateTime - startDateTime).TotalHours)
        |> Seq.replicate absorbanceWavelengthCount
        |> Seq.concat*)

    //Pretend that readings are aligned with flurorensce as characterisation needs them so currently.
    let timesAborbance =
        fluorescenceReads//<---- WE ARE USING FLUORESCENCE TIMES INSTEAD OF ABSORBANCE TO ALIGN TIMES***
        |> Seq.map (fun measurement -> (measurement.ReadingDateTime - startDateTime).TotalHours)
        |> Seq.replicate absorbanceWavelengthCount
        |> Seq.concat

    let timesCSV =
        //Seq.concat [timesFluorescence; timesFluorescence; timesFluorescence; timesAborbance]
        [timesFluorescence; timesAborbance]
        |> Seq.concat
        |> Seq.map string
        |> String.concat ", "

    let dataRows =
        wellData_TimeRows
        |> Array.map (fun row ->
            row
            |> Array.map string
            |> String.concat ", ")

    

    let combinedHeader = "Content, Colony, Well Col, Well Row, Content," + header + System.Environment.NewLine +
                         "       ,       ,         ,         ,        ," + timesCSV

    let specAndData =
        (spec, dataRows) ||> Array.map2 (fun specLine dataLine -> specLine + "," + dataLine)
        |> String.concat System.Environment.NewLine

    let all = combinedHeader + System.Environment.NewLine + specAndData

    all