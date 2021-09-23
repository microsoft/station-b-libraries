// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Compare

open System
open System.IO.Compression

open BCKG
open BCKG.Events
open BCKG.Domain
open Thoth.Json.Net


let compareParts (log1name,map1) (log2name,map2) = 
    
    let ids1 = map1 |> Map.toList |> List.unzip |> fst
    let ids2 = map2 |> Map.toList |> List.unzip |> fst

    let in1not2 = ids1 |> List.except ids2
    let in2not1 = ids2 |> List.except ids1 

    match in1not2 with 
    | [] -> () 
    | _ -> printfn "The following %d parts are in %s but not in %s:\n%A" (in1not2.Length) log1name log2name in1not2
    match in2not1 with 
    | [] -> ()
    | _ -> printfn "The following %d parts are in %s but not in %s:\n%A" (in2not1.Length) log2name log1name in2not1

    let in1and2 = ids1 |> List.filter (fun id -> (ids2 |> List.contains id))

    in1and2 
    |> List.iter (fun id -> 
        let entity1 = map1.Item(id)
        let entity2 = map2.Item(id)
        let change = BCKG.Events.modifyPart "" entity1 entity2
        match change with 
        | Some(ev) ->
            printfn "Difference between Part Id %s in log %s and log %s is: %s" (id.ToString()) (log1name) (log2name) (ev.change)
        | None -> ())

let compareReagents (log1name,map1) (log2name,map2) = 
    
    let ids1 = map1 |> Map.toList |> List.unzip |> fst
    let ids2 = map2 |> Map.toList |> List.unzip |> fst

    let in1not2 = ids1 |> List.except ids2
    let in2not1 = ids2 |> List.except ids1 

    match in1not2 with 
    | [] -> () 
    | _ -> printfn "The following %d reagents are in %s but not in %s:\n%A" (in1not2.Length) log1name log2name in1not2
    match in2not1 with 
    | [] -> ()
    | _ -> printfn "The following %d reagents are in %s but not in %s:\n%A" (in2not1.Length) log2name log1name in2not1

    let in1and2 = ids1 |> List.filter (fun id -> (ids2 |> List.contains id))

    in1and2 
    |> List.iter (fun id -> 
        let entity1 = map1.Item(id)
        let entity2 = map2.Item(id)
        let change = BCKG.Events.modifyReagent "" entity1 entity2
        match change with 
        | Some(ev) ->
            match ev.target with 
            | ReagentEvent _ -> printfn "Difference between Reagent Id %s in log %s and log %s is: %s" (id.ToString()) (log1name) (log2name) (ev.change)
            | _ -> failwith "Unexpected type of event encountered."        
        | None -> ()
        
                
                
        )

let compareSamples (log1name,map1) (log2name,map2) = 
    
    let ids1 = map1 |> Map.toList |> List.unzip |> fst
    let ids2 = map2 |> Map.toList |> List.unzip |> fst

    let in1not2 = ids1 |> List.except ids2
    let in2not1 = ids2 |> List.except ids1 

    match in1not2 with 
    | [] -> () 
    | _ -> printfn "The following %d samples are in %s but not in %s:\n%A" (in1not2.Length) log1name log2name in1not2
    match in2not1 with 
    | [] -> ()
    | _ -> printfn "The following %d samples are in %s but not in %s:\n%A" (in2not1.Length) log2name log1name in2not1

    let in1and2 = ids1 |> List.filter (fun id -> (ids2 |> List.contains id))

    in1and2 
    |> List.iter (fun id -> 
        let entity1 = map1.Item(id)
        let entity2 = map2.Item(id)
        let changes = BCKG.Events.modifySample "" entity1 entity2
         
        match changes with 
        | Some(ev) -> 
            match ev.target with 
            | SampleEvent _ -> 
                printfn "Difference between Sample Id %s in log %s and log %s is: %s" (id.ToString()) (log1name) (log2name) (ev.change)
            | _ -> failwith "Unexpected type of event encountered."
        | None -> ()   )
           
        

let compareExperiments (log1name,map1) (log2name,map2) = 
    
    let ids1 = map1 |> Map.toList |> List.unzip |> fst
    let ids2 = map2 |> Map.toList |> List.unzip |> fst

    let in1not2 = ids1 |> List.except ids2
    let in2not1 = ids2 |> List.except ids1 

    match in1not2 with 
    | [] -> () 
    | _ -> printfn "The following %d experiment are in %s but not in %s:\n%A" (in1not2.Length) log1name log2name in1not2
    match in2not1 with 
    | [] -> ()
    | _ -> printfn "The following %d experiments are in %s but not in %s:\n%A" (in2not1.Length) log2name log1name in2not1

    let in1and2 = ids1 |> List.filter (fun id -> (ids2 |> List.contains id))

    in1and2 
    |> List.iter (fun id -> 
        let entity1 = map1.Item(id)
        let entity2 = map2.Item(id)
        let changes = BCKG.Events.modifyExperiment "" entity1 entity2
        match changes with 
        | Some (ev) -> 
            match ev.target with 
            | ExperimentEvent _ -> 
                printfn "Difference between Experiment Id %s in log %s and log %s is: %s" (id.ToString()) (log1name) (log2name) (ev.change)
            | _ -> failwith "Unexpected type of event encountered." 
        | None -> ()
         
                
                
        )

let compareExperimentOperations (log1name,map1) (log2name,map2) = 
    
    let ids1 = map1 |> Map.toList |> List.unzip |> fst
    let ids2 = map2 |> Map.toList |> List.unzip |> fst

    let in1not2 = ids1 |> List.except ids2
    let in2not1 = ids2 |> List.except ids1 

    match in1not2 with 
    | [] -> () 
    | _ -> printfn "The following %d experiment events are in %s but not in %s:\n%A" (in1not2.Length) log1name log2name in1not2
    match in2not1 with 
    | [] -> ()
    | _ -> printfn "The following %d experiment events are in %s but not in %s:\n%A" (in2not1.Length) log2name log1name in2not1

    let in1and2 = ids1 |> List.filter (fun id -> (ids2 |> List.contains id))

    in1and2 
    |> List.iter (fun id -> 
        let entity1 = map1.Item(id)
        let entity2 = map2.Item(id)
        let change = BCKG.Events.addExperimentOperation "" (ExperimentId.Create()) entity1 entity2
        match change with 
        | Some(ev) ->
            printfn "Difference between ExperimentEvent Id %s in log %s and log %s is: %s" (id.ToString()) (log1name) (log2name) (ev.change)
        | None -> ())


let compareDatabases (source:BCKG.API.Instance) (dest:BCKG.API.Instance) = 
    let sourceId = "source"
    let destinationId = "destination"


    printfn "Retrieving Source Parts"
    let sourceParts = 
        source.GetParts()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun p -> (p.id,p))
        |> Map.ofList

    printfn "Retrieving Destination Parts"
    let destParts = 
        dest.GetParts()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun p -> (p.id,p))
        |> Map.ofList

    printfn "Begin Part comparision."
    compareParts (sourceId,sourceParts) (destinationId,destParts)
    printfn "End Part comparision."
    printfn "------------------------------"

    printfn "Retrieving Source Reagents"
    let sourceReagents = 
        source.GetReagents()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun r -> (r.id,r))
        |> Map.ofList
    
    printfn "Retrieving Destination Reagents"
    let destReagents = 
        dest.GetReagents()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun r -> (r.id,r))
        |> Map.ofList
    
    printfn "Begin Reagent comparision."
    compareReagents (sourceId,sourceReagents) (destinationId,destReagents)
    printfn "End Reagent comparision."
    printfn "------------------------------"
    
    printfn "Retrieving Source Experiments"
    let sourceExperiments = 
        source.GetExperiments()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun e -> (e.id,e))
        |> Map.ofList
    
    printfn "Retrieving Destination Experiments"
    let destExperiments = 
        dest.GetExperiments()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun e -> (e.id,e))
        |> Map.ofList
   
    printfn "Begin Experiment comparision."
    compareExperiments (sourceId,sourceExperiments) (destinationId,destExperiments)
    printfn "End Experiment comparision."
    printfn "------------------------------"

    printfn "Retrieving Source Samples"
    let sourceSamples =
        source.GetSamples()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun e -> (e.id,e))
        |> Map.ofList
    
    printfn "Retrieving Destination Samples"
    let destinationSamples =
        dest.GetSamples()
        |> Async.RunSynchronously
        |> Array.toList
        |> List.map (fun e -> (e.id,e))
        |> Map.ofList
    

    printfn "Begin Sample comparision."
    compareSamples (sourceId,sourceSamples) (destinationId,destinationSamples)
    printfn "End Experiment comparision."
    printfn "------------------------------"

    

    

    