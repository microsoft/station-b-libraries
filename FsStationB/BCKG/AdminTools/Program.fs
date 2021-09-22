// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Main

open Argu
open BCKG.Admin.Utilities

type Command = 
    | ValidateLog
    | ReplayLog
    | CopyAll
    | CopyBlobs
    | CopyTables
    | ClearTables
    | InitializeTables
    | BackupBlobs
    | BackupState
    | BackupEventsLog
    | FullBackup
    | PrintStats 
    | LogToHtml
    | GenBankToEvents
    | PartsFileToEvents
    | ReagentsFileToEvents
    | BarcodeDbToEvents
    | DiskDbToEvents
    | CompareDatabases
    | CompareLogToDatabase
    | Debug
    
type AdminArguments = 
    | [<Mandatory>] Command of Command 
    | [<Mandatory>] Source of string
    | Dest of string
    | LogName of string
    | Clean of bool
    | GenBankTypes of bool
    | ConnectionString of string
    | Tag of string
    | Date of string
    | EntityType of EntityType
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Command c -> "Command/Operation to perform." 
            | Source _ -> "Filepath or Connection string of Source. e.g. %BCKG_STAGING_CONNECTION_STRING%"
            | Dest _ -> "Filepath or Connection string of Destination. e.g. %BCKG_STAGING_CONNECTION_STRING%"
            | LogName _ -> "Name of the logfile (with the file extension). E.g. EventsLog.json "
            | Clean _ -> "Should the destination containers be erased first?"
            | GenBankTypes _ -> "are part types specified using GenBank or BCKG strings"
            | ConnectionString _ -> "Only used for the command --DataToEvents"
            | Tag _ -> "specify a set of tags (separated by ; or , or /) to label devices/experiments/etc"
            | Date _ -> "specify the date on which an experiment was performed (default is 01/01/2000)"
            | EntityType _ -> "specify the entity type for you would like to debug."

let private getTags tagsOption command = 
    match tagsOption with 
    | Some(tags) -> tags
    | None -> failwithf "%s requires tags. Please use the --tag flag." (command.ToString())

let private getEntityType entityTypeOption command = 
    match entityTypeOption with 
    | Some(entityT) -> entityT
    | None -> failwithf "%s requires an entity type. Please use the --entitytype flag." (command.ToString())

let private getGenBankTypes genBankTypesOption command = 
    match genBankTypesOption with 
    | Some(gbt) -> gbt 
    | None -> failwithf "%s requires a GenBankTypes Boolean value. Please use the --genbanktypes flag." (command.ToString())

let private getProject projectOption command = 
    match projectOption with 
    | Some(proj) -> proj
    | None -> failwithf "%s requires a Project to be specified. Please use the --project flag." (command.ToString())

let private getDestinationString destinationOption command = 
    match destinationOption with 
    | Some(destVal) -> destVal
    | None -> failwithf "%s requires a destination value. Please use the --dest flag." (command.ToString())

let private checkSourceConnectionString (sourceConnectionString) =
    if sourceConnectionString = "" then
        failwith "Source connection string not set correctly."    

let private checkDesintationConnectionString (connectionString) =    
    if connectionString = "" then 
        failwith "Destination connection string not set correctly."
    if connectionString.ToLowerInvariant().Contains("production") then
        failwith "Your destination connection string contains \"production\", aborting."

let private validateConnectionStrings (sourceConnectionString) (destinationConnectionString) = 
    checkSourceConnectionString sourceConnectionString
    checkDesintationConnectionString destinationConnectionString
    if sourceConnectionString.ToLowerInvariant() = destinationConnectionString.ToLowerInvariant() then
        failwith "Your source and destination connection strings are the same, aborting."

let private getConnectionString connstringOption command = 
    match connstringOption with 
    | Some(connstr) -> connstr
    | None -> failwithf "%s requires a Connection String. Please use the --connectionstring flag." (command.ToString())

let private getLogname lognameOption command = 
    match lognameOption with 
    | Some(lognameVal) -> lognameVal
    | None -> failwithf "%s requires a log name value. Please use the --logname flag." (command.ToString())


[<EntryPoint>]
let main argv =  
    
    printfn "Welcome to the BCKG Admin Console Tool. Please read the warnings before you proceed."
    let parser = ArgumentParser.Create<AdminArguments>(programName = "BckgAdmin.exe")
    let parserResults = parser.Parse(argv)
    
    let now = System.DateTime.UtcNow.ToString("yyyyMMdd")   
    
    let sourceString = parserResults.GetResult Source
    let command = parserResults.GetResult Command
    
    let destinationOption = parserResults.TryGetResult Dest
    let lognameOption = parserResults.TryGetResult LogName
    let genBankTypesOption = parserResults.TryGetResult GenBankTypes
    let entityTypeOption = parserResults.TryGetResult EntityType
    
    let date = parserResults.GetResult(Date, "01/01/2000")
    
    let tagsOption = 
        parserResults.TryGetResult(Tag)
        |> Option.map (fun t -> t.Split([|";"; ","; "/"|], System.StringSplitOptions.RemoveEmptyEntries) |> List.ofSeq )
    
    let connstringOption = parserResults.TryGetResult ConnectionString
    
    let enableCleaning = parserResults.GetResult(Clean, false)
    
    if enableCleaning then printfn "\tWARNING: DESTINATION CONTAINERS WILL BE DELETED!"
    
    
    printf "Proceed (y/n)? "
    let confirm = System.Console.ReadLine()

    if confirm.ToLowerInvariant() = "y" then 
        match command with
        | ValidateLog -> failwith "Validate Log has not been implemented yet."
        
        | CompareLogToDatabase -> failwith "Compare Log to Database not implemented yet."

        | ClearTables -> 
            checkDesintationConnectionString sourceString
            BCKG.Admin.Utilities.clearAllTables sourceString
        
        | InitializeTables -> 
            checkDesintationConnectionString sourceString
            BCKG.Admin.Utilities.initializeTables sourceString

        | ReplayLog -> 
            let destinationString = getDestinationString destinationOption command          
            let lognameValue = getLogname lognameOption command
            BCKG.Admin.Log.replayLog sourceString destinationString lognameValue enableCleaning

        | CopyAll -> 
            let destinationString = getDestinationString destinationOption command
            validateConnectionStrings sourceString destinationString
            BCKG.Admin.Copy.CopyAccountTables sourceString destinationString enableCleaning 
            BCKG.Admin.Copy.CopyAccountBlobs sourceString destinationString enableCleaning
            
        | CopyBlobs -> 
            let destinationString = getDestinationString destinationOption command
            validateConnectionStrings sourceString destinationString
            BCKG.Admin.Copy.CopyAccountBlobs sourceString destinationString enableCleaning

        | CopyTables -> 
            let destinationString = getDestinationString destinationOption command
            validateConnectionStrings sourceString destinationString
            BCKG.Admin.Copy.CopyAccountTables sourceString destinationString enableCleaning
        
        | BackupBlobs -> 
            let destinationString = getDestinationString destinationOption command
            checkSourceConnectionString sourceString
            let blobsZipFile = sprintf "%s/%s_Blobs.zip" destinationString now
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            BCKG.Admin.Backup.DownloadBlobs source "files" blobsZipFile

        | BackupState -> 
            let destinationString = getDestinationString destinationOption command
            checkSourceConnectionString sourceString
            let stateBackupFile = sprintf "%s/%s_State.json" destinationString now
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            BCKG.Admin.Backup.BackupStateTo source stateBackupFile

        | BackupEventsLog -> 
            let destinationString = getDestinationString destinationOption command
            checkSourceConnectionString sourceString
            let eventsBackupFile = sprintf "%s/%s_Events.json" destinationString now
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            BCKG.Admin.Backup.BackupEventsTo source eventsBackupFile
            
        | FullBackup -> 
            let destinationString = getDestinationString destinationOption command
            checkSourceConnectionString sourceString
            let blobsZipFile = sprintf "%s/%s_Blobs.zip" destinationString now
            let stateBackupFile = sprintf "%s/%s_State.json" destinationString now
            let eventsBackupFile = sprintf "%s/%s_Events.json" destinationString now
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            BCKG.Admin.Backup.DownloadBlobs source "files" blobsZipFile
            BCKG.Admin.Backup.BackupStateTo source stateBackupFile
            BCKG.Admin.Backup.BackupEventsTo source eventsBackupFile            
        
        | PrintStats ->
            checkSourceConnectionString sourceString
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            source.PrintStats() |> Async.RunSynchronously
        
        | LogToHtml -> 
            let destinationString = getDestinationString destinationOption command         
            let logname = getLogname lognameOption command
            BCKG.Admin.Visualization.Visualize sourceString logname destinationString 
        
        | GenBankToEvents ->
            failwith "Needs to be reimplemented."    

        | PartsFileToEvents -> 
            let destinationString = getDestinationString destinationOption command
            let genBankTypes = getGenBankTypes genBankTypesOption command
            let _, _, events = Extraction.LoadPartsFromFile genBankTypes sourceString
            let logFile = 
                (events |> Array.toList) 
                |> List.map (fun e -> BCKG.Events.Event.encode e)
                |> List.fold (fun acc x -> (acc + "," + x) ) ""
                |> sprintf "[%s]"
            System.IO.File.WriteAllText(destinationString, logFile)

        | ReagentsFileToEvents -> 
            let destinationString = getDestinationString destinationOption command
            let events = Extraction.LoadReagentsFromFile sourceString
            let logFile = 
                events
                |> List.map (fun e -> BCKG.Events.Event.encode e)
                |> List.fold (fun acc x -> (acc + "," + x) ) ""
                |> sprintf "[%s]"
            System.IO.File.WriteAllText(destinationString, logFile)  

        | BarcodeDbToEvents -> 
            failwith "Needs to be reimplemented."

        | DiskDbToEvents -> 
            failwith "Needs to be reimplemented."

        | CompareDatabases -> 
            let destinationString = getDestinationString destinationOption command
            let source = BCKG.API.Instance(BCKG.API.CloudInstance(sourceString), "Admin")
            let dest = BCKG.API.Instance(BCKG.API.CloudInstance(destinationString), "Admin")
            BCKG.Admin.Compare.compareDatabases source dest
        
        | Debug -> 
            let entityType = getEntityType entityTypeOption command 
            let destinationString = getDestinationString destinationOption command
            let id = (getTags tagsOption command).Head 

            BCKG.Admin.Log.debug sourceString destinationString entityType id
            ()

    else 
        printfn "Action Cancelled"
    0
    
