// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Extraction

open System
open BCKG.Domain
open BCKG.Events


let SaveReagentWithDelay (r:Reagent) = 
    System.Threading.Thread.Sleep(200)
    r |> BCKG.Events.addReagent ""
    

let SavePartWithDelay (p:Part) =
    System.Threading.Thread.Sleep(200)
    p |> BCKG.Events.addPart ""

let SaveExperimentWithDelay (p:Experiment) =
    System.Threading.Thread.Sleep(200)
    p |> BCKG.Events.addExperiment ""
    




(**************************************************************************
    
 **************************************************************************)
let LoadPartsFromFile useGenbankTypes (partsFilePath:string) = 
    let typeFromString = if useGenbankTypes then Part.FromGenBankStringType else Part.FromStringType
    let parts = 
        partsFilePath
        |> System.IO.File.ReadAllLines
        |> Array.skip 1
        |> Array.map(fun l -> 
            let fields = l.Split(",")
            let props = { 
                PartProperties.name = fields.[0].Trim()
                PartProperties.sequence = fields.[2].ToUpper().Trim()
                PartProperties.deprecated = false
                }
            typeFromString (System.Guid.NewGuid()) props fields.[1] 
            )
    let partsMap = parts |> Array.map(fun p -> p.id, p) |> Map.ofArray
    let partEvents = parts |> Array.map SavePartWithDelay
    parts, partsMap, partEvents



(**************************************************************************
    Load genbank files from a folder
    
    If a set of parts is provided, then the part annotation is used to generate a name.
    Otherwise, the file name is used for the device name
 **************************************************************************)
let LoadAllDevicesFrom (path : string) (barcodeMap:Map<string,Barcode>) (partsMap:Map<PartId,Part> option) t =
    failwith "Needs to be reimplemented"
    (*
    let MkName filename parts sequence = 
        match parts with 
        | Some pMap -> 
            let deviceParts = Reagent.AnnotateSequence pMap sequence
            Reagent.MkDnaDeviceName pMap deviceParts
        | None -> filename
            
    System.IO.Directory.EnumerateFiles path
    |> Seq.toArray
    |> Array.map (fun file ->
            let fileName = file.Replace(path, "")
            let index = fileName.IndexOf(".")
            let fileName = fileName.[1..index - 1]
            let sequence = 
                file
                |> System.IO.File.ReadAllText
                |> Reagent.SequenceFromGenBank        
            { Reagent.barcode = barcodeMap.TryFind sequence
              //Reagent.context = None
              Reagent.files = List.empty
              Reagent.id = System.Guid.NewGuid() |> ReagentId
              Reagent.name = MkName fileName partsMap sequence
              Reagent.notes = sprintf "Extracted from file %s" fileName
              Reagent.sequence = Some sequence
              Reagent.Type = t
              Reagent.concentration = None
              Reagent.tags = None
            })       
    |> Array.map SaveReagentWithDelay
    |> Array.toList
    |> List.fold (fun acc x -> acc@x) []
    *)

//Function to load data from V1 barcoder DB (blob files by barcode + index table)
let LoadReagentsFromBarcodeBlobs (blobsPath:string) (indexPath: string) =     
    failwith "Needs to be reimplemented"
    (*
    let reagents, barcodeMap = 
        indexPath
        |> System.IO.File.ReadAllLines
        |> Array.skip 1
        |> Array.map(fun l -> 
            let fields = l.Split(",")
            let barcode = fields.[1]
            let filename = fields.[3]
            let notes = fields.[5]
            let name = fields.[7]
            
            let genBankFile = blobsPath + "\\" + barcode        
            let sequence = 
                if System.IO.File.Exists(genBankFile) then 
                    let file = 
                        genBankFile
                        |> System.IO.File.ReadAllText

                    if file.Trim() <> "" then 
                        file
                        |> Reagent.SequenceFromGenBank                  
                        |> Some
                    else 
                        None
                else
                    None
        
            let barcode = if barcode.Trim() <> "" then Some (Barcode (barcode.Trim())) else None
        
            let reagent = 
                { Reagent.barcode = barcode
                  //Reagent.context = None
                  Reagent.files = List.empty
                  Reagent.id = System.Guid.NewGuid() |> ReagentId
                  Reagent.name = name
                  Reagent.notes = notes
                  Reagent.sequence = sequence
                  Reagent.concentration = None
                  Reagent.tags = None
                  Reagent.Type = 
                      match sequence with 
                      | Some _ -> SourcePlasmidDNA
                      | None -> Chemical
                }
        
            let barcodeEntry = if barcode.IsSome && sequence.IsSome then Some (sequence,barcode) else None
            reagent, barcodeEntry
            )
        |> Array.unzip


    (reagents |> Array.map SaveReagentWithDelay)//, (barcodeMap |> Seq.choose id |> Map.ofSeq)
    |> Array.toList
    |> List.fold (fun acc x -> acc@x) []
    *)


let LoadReagentsFromFile (path:string) =        
    failwith "Needs to be reimplemented."
    (*path
    |> System.IO.File.ReadAllLines
    |> Array.skip 1
    |> Array.map(fun l -> 
        let fields = l.Split(",")
        { Reagent.barcode = None
          //Reagent.context = None
          Reagent.files = List.empty
          Reagent.id = System.Guid.NewGuid() |> ReagentId
          Reagent.name = fields.[0]
          Reagent.notes = ""
          Reagent.sequence = None
          Reagent.Type = fields.[1] |> ReagentType.fromString
          Reagent.concentration = None
          Reagent.tags = None
        })

    |> Array.map SaveReagentWithDelay
    |> Array.toList
    |> List.fold (fun acc x -> acc@x) []*)