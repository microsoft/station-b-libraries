// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Log

open BCKG.Events
open BCKG.Admin.Utilities


let replayLog (source:string) (destination:string) (logname:string) enableCleaning = 
    let logfilefp = System.IO.Path.Join(source,logname)
    let logevents = BCKG.Admin.Utilities.get_log_events_sorted logfilefp
    let createstorage = BCKG.Storage.initialiseDatabase destination |> Async.AwaitTask
    
    //if enableCleaning then 
    //    BCKG.Admin.Utilities.clearAllBlobs destination
    //    BCKG.Admin.Utilities.clearAllTables destination
    let db = BCKG.API.Instance(BCKG.API.CloudInstance(destination), "Admin")
    printfn "Starting Replay Log. %s has %d events." logname logevents.Length 
    logevents 
    |> List.iteri (fun i e -> 
        printfn "Log Event [%d/%d] - %s" i (logevents.Length-1) (e.id.ToString())
        
        match e.target with 
        | EventTarget.PartEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.PartTagEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore

        | EventTarget.ReagentEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ReagentFileEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ReagentTagEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        | EventTarget.ExperimentEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ExperimentFileEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ExperimentOperationEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ExperimentSignalEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ExperimentTagEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        | EventTarget.SampleEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.SampleDataEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.SampleDeviceEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.SampleConditionEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.SampleTagEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.SampleReplicateEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore

        | EventTarget.CellEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.CellEntityEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.CellFileEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.CellTagEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        | EventTarget.InteractionEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        | EventTarget.DerivedFromEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore

        | EventTarget.FileEvent(fid) ->
            let filecontent = 
                System.IO.Path.Join(source,(fid.ToString()))
                |> System.IO.File.ReadAllText
                |> System.Text.Encoding.UTF8.GetBytes
            db.ProcessEvent ((e,filecontent) |> BCKG.API.BlobEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
            
        | EventTarget.TimeSeriesFileEvent(fid) ->
            let filecontent = 
                System.IO.Path.Join(source,(fid.ToString()))
                |> System.IO.File.ReadAllText
                |> System.Text.Encoding.UTF8.GetBytes
            db.ProcessEvent ((e,filecontent) |> BCKG.API.BlobEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
                        
        | EventTarget.BundleFileEvent(fid) -> 
            let bundlefp = System.IO.Path.Join(source,(fid.ToString()))
            let bundlebytearray = System.IO.File.ReadAllBytes(bundlefp)
            db.ProcessEvent ((e,bundlebytearray) |> BCKG.API.BlobEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
             
        | EventTarget.StartLogEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.FinishLogEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore

        | EventTarget.ProcessDataEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        | EventTarget.ParseLayoutEvent _ -> db.ProcessEvent (e |>  BCKG.API.TableEvent) |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        | EventTarget.ObservationEvent _ -> db.ProcessEvent (e |> BCKG.API.TableEvent)  |> BCKG.Events.EventResult.ExecuteAsyncAndIgnore
        
        )


let debug (source:string) (destination:string) (entityType:EntityType) (id:string) =
    
    let logevents = BCKG.Admin.Utilities.get_log_events_sorted source
    
    let events =
        match entityType with 
        | BCKG.Admin.Utilities.EntityType.Part -> 
            []
        | BCKG.Admin.Utilities.EntityType.Reagent -> []
        | BCKG.Admin.Utilities.EntityType.Experiment -> []
        | BCKG.Admin.Utilities.EntityType.Sample -> 
            let sampleId = id |> System.Guid |> BCKG.Domain.SampleId
            logevents |> List.filter (fun e -> 
                match e.target with
                | SampleEvent(sid) -> sid = sampleId
                | SampleConditionEvent (sid) -> sid = sampleId
                | SampleDataEvent (sid) -> sid = sampleId
                | SampleDeviceEvent(sid) -> sid = sampleId
                | _ -> false
                
                )
    
    let v2json = 
        events 
        |> List.map (fun e -> Event.encode e)
        |> String.concat ","
        |> sprintf "[%s]"
    System.IO.File.WriteAllText(destination,v2json)
    