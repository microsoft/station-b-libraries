// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Utilities



open Microsoft.Azure.Storage
open Microsoft.Azure.Storage.Blob


type EntityType = 
    | Sample
    | Part
    | Reagent
    | Experiment

let LoadResource name = 
    let assembly  = System.Reflection.Assembly.GetExecutingAssembly()    
    let resource = 
        name
        |> sprintf "BckgAdmin.%s"
        |> assembly.GetManifestResourceStream
    let stream = new System.IO.StreamReader(resource)
    stream.ReadToEnd()



let get_log_events_sorted (logfilefp:string)  =    
    
    if not (System.IO.File.Exists(logfilefp)) then failwithf "%s is not a valid filepath." logfilefp
    let log = System.IO.File.ReadAllText(logfilefp)
    
    (BCKG.Events.Event.decode_list log)
    |> List.sortBy (fun e -> e.timestamp)


(*Load All Reagents*)
let LoadBckgReagents (db:BCKG.API.Instance) =
    let reagents = 
        (db.GetReagents() |> Async.RunSynchronously)
        |> Array.map(fun r -> r.name, r.id)
        
    let nameErrors = 
        reagents 
        |> Array.map fst 
        |> Array.groupBy id
        |> Array.map (fun (name, L) -> name, L.Length)
        |> Array.filter (fun (_,cnt) -> cnt<>1)
        |> Array.map fst
        |> Set.ofSeq
    if not (Set.isEmpty nameErrors) then
        printf "WARNING: Duplicate reagent names found in BCKG. These reagents will be filtered to avoid errors and would not be available for matching by name."

    reagents
    |> Array.filter (fun (name,_) -> not (nameErrors.Contains name))
    |> Map.ofSeq



(*AZURE Table and Blob Operations*)
let ListTableContainers (account:Microsoft.WindowsAzure.Storage.CloudStorageAccount) = 
    (false, null)
    |> Seq.unfold(fun (isNotInitial, token) -> 
        if isNotInitial && token = null then None 
        else    
            let segment = account.CreateCloudTableClient().ListTablesSegmentedAsync(token) |> Async.AwaitTask |> Async.RunSynchronously
            Some(segment.Results, (true, segment.ContinuationToken))
        )
    |> Seq.concat
    |> Seq.map(fun container -> container.Name)

let getTableContainers sourceConnectionString = 
    let accountSourceTable = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse sourceConnectionString
    ListTableContainers accountSourceTable

let getAccountTable (connectionString) = 
    Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse connectionString

let CleanTableContainer (account:Microsoft.WindowsAzure.Storage.CloudStorageAccount) container = 
    let tableRef = account.CreateCloudTableClient().GetTableReference container
    let exists = tableRef.ExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously
    if exists then 
        let mutable counter = 100
        let mutable error = true
        
        tableRef.DeleteIfExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore        
        while error do
            if counter < 0  then
                failwith "Max number of iterations reached" 
            try                                
                tableRef.CreateIfNotExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore
                error <- false
            with 
            | e ->        
                printfn "Waiting for an Azure table delete...(%i left)\n%s" counter (e.Message)
                System.Threading.Thread.Sleep 1000          
                counter <- counter - 1
    else
        ()
 
let CleanBlobContainer (account:CloudStorageAccount) container = 
    let blobRef = account.CreateCloudBlobClient().GetContainerReference container 
    let exists = blobRef.ExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously
    if exists then 
        let mutable counter = 100
        let mutable error = true
        
        blobRef.DeleteIfExists() |> ignore
        while error do
            if counter < 0  then
                failwith "Max number of iterations reached" 
            try                                
                blobRef.CreateIfNotExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore
                error <- false
            with 
            | e ->        
                printfn "Waiting for an Azure blob delete...(%i left)\n%s" counter (e.Message)
                System.Threading.Thread.Sleep 1000          
                counter <- counter - 1
    else
        ()

let clearAllTables (connectionString) = 
    let tableContainers = getTableContainers connectionString 
    let accountTable = getAccountTable connectionString
    tableContainers
    |> Seq.iter (fun tableContainer ->             
        printf "\t - Cleaning %s..." tableContainer
        CleanTableContainer accountTable tableContainer
        printfn "done")

let initializeTables (connectionString) = 
    let _ = BCKG.Storage.initialiseDatabase connectionString
    ()

let clearAllBlobs (connectionString) = 
    let accountSource = CloudStorageAccount.Parse(connectionString)
    let blobContainers = accountSource.CreateCloudBlobClient().ListContainers() |> Seq.map(fun container -> container.Name)  
    blobContainers
    |> Seq.iter (fun blobContainer ->       
        printf "\t - Cleaning %s..." blobContainer
        CleanBlobContainer accountSource blobContainer
        printfn "done")
    