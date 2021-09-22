// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Admin.Visualization


open BCKG.Domain
open BCKG.Events
open Thoth.Json.Net


let Visualize (source:string) (logname:string) outputFile = 
    let template = BCKG.Admin.Utilities.LoadResource "VisualizationTemplate.html"                
    //let filename = input.[input.LastIndexOf("\\")+1..].Replace(".json","")
    let logfilefp = System.IO.Path.Join(source,logname)
    let events = BCKG.Admin.Utilities.get_log_events_sorted logfilefp
    
    
    let groupedEvents = 
        events
        |> List.groupBy (fun e -> e.timestamp.ToString("dd-MM-yyyy"))

   
    let controls = 
        groupedEvents
        |> List.map (fun (t,_) -> sprintf "<button class=\"tablinks\" onclick=\"openCity(event, '%s')\">%s</button>" t t)
        |> String.concat "\n"
        |> sprintf "<div class=\"tab\">%s</div>"
                                                            

    let changeToHtml (changestring:string) = 
        let change = Decode.Auto.unsafeFromString<(string*string)list>(changestring)
        let changehtml = 
            change |> List.map (fun (op,c) -> 
                let (cdisplay,title) = 
                    if c.Length > 32 then 
                        ((c.[0..30] + "...."),c)
                    else 
                        (c,"")
                sprintf  "<p title=\"%s\">%s: %s<p/>" title op cdisplay
                )

        changehtml |> List.fold (fun acc x -> acc + x) ""
    

    let content = 
        groupedEvents
        |> List.map(fun (t,E) -> 
            E
            |> List.mapi(fun i (e) -> 
                let msg = BCKG.Events.EventTarget.toTargetTypeString e.target         
                
                let info = 
                    match e.target with
                    //Part Events
                    | PartEvent (pid)                     -> sprintf "<p>%s Part: %s<br>Change:</p>%s" (e.operation.ToString()) (pid.ToString()) (changeToHtml e.change)
                    //Reagent Events                      
                    | ReagentEvent     (rid)              -> sprintf "<p>%s Reagent: %s<br>Change:</p>%s" (e.operation.ToString()) (rid.ToString()) (changeToHtml e.change)
                    | ReagentFileEvent (rid)              -> sprintf "<p>%s Reagent File: %s<br>Change:</p>%s" (e.operation.ToString()) (rid.ToString()) (changeToHtml e.change)
                    //Experiment Events                   
                    | ExperimentEvent (eid)               -> sprintf "<p>%s Experiment: %s<br>Change:</p>%s" (e.operation.ToString()) (eid.ToString()) (changeToHtml e.change)
                    | ExperimentFileEvent (eid)           -> sprintf "<p>%s Experiment File: %s<br>Change:</p>%s" (e.operation.ToString()) (eid.ToString()) (changeToHtml e.change)
                    | ExperimentOperationEvent (eid)      -> sprintf "<p>%s Experiment Operation: %s<br>Change:</p>%s" (e.operation.ToString()) (eid.ToString()) (changeToHtml e.change)
                    | ExperimentSignalEvent (eid)         -> sprintf "<p>%s Experiment Signal: %s<br>Change:</p>%s" (e.operation.ToString()) (eid.ToString()) (changeToHtml e.change)
                    //File Events                         
                    | FileEvent (fid)                     -> sprintf "<p>%s File: %s</p>" (e.operation.ToString()) (fid.ToString()) 
                    | TimeSeriesFileEvent (fid)           -> sprintf "<p>%s Time Series File: %s</p>" (e.operation.ToString()) (fid.ToString()) 
                    | BundleFileEvent (fid)               -> sprintf "<p>%s Bundle: %s</p>" (e.operation.ToString()) (fid.ToString()) 
                    //Sample Events                       
                    | SampleEvent (sid)                   -> sprintf "<p>%s Sample : %s<br>Change:</p>%s" (e.operation.ToString()) (sid.ToString()) (changeToHtml e.change) 
                    | SampleDeviceEvent (sid)             -> sprintf "<p>%s Sample Device: %s<br>Change:</p>%s" (e.operation.ToString()) (sid.ToString()) (changeToHtml e.change)
                    | SampleDataEvent (sid)               -> sprintf "<p>%s Sample Data: %s<br>Change:</p>%s" (e.operation.ToString()) (sid.ToString()) (changeToHtml e.change)
                    | SampleConditionEvent (sid)          -> sprintf "<p>%s Sample Condition: %s<br>Change:</p>%s" (e.operation.ToString()) (sid.ToString()) (changeToHtml e.change)
                    //Replay Log Events                   
                    | StartLogEvent (lname)               -> sprintf "<p>Start Processing Log: %s</p>" (lname)  
                    | FinishLogEvent (lname)              -> sprintf "<p>Finish Processing Log: %s</p>" (lname)
                                                     
                    | EventTarget.ProcessDataEvent (eid)  -> sprintf "<p>Process Data for Experiment: %s</p>" (eid.ToString())
                    | EventTarget.ParseLayoutEvent (eid)  -> sprintf "<p>Parse Layout for Experiment: %s</p>" (eid.ToString())
                                
                sprintf "<h4>%s %s</h4><div style=\"padding-left:50px\">%s</div>" (e.timestamp.ToString("HH:mm:ss:fff")) msg info
                |> sprintf "<div class=\"container %s\"><div class=\"content\">%s</div></div>" (if i%2=0 then "left" else "right")         
                )
            |> String.concat "\n"
            |> sprintf "<div id=\"%s\" class=\"tabcontent\"><div class=\"timeline\">%s</div></div>" t            
            )        
        |> String.concat "\n"
        |> sprintf "<div style=\"max-height:90%%; overflow:auto\">%s</div>"
        |> sprintf "<div style=\"position:absolute; top:0; bottom:0; left:0; right:0\">%s\n%s</div>" controls

    System.IO.File.WriteAllText(outputFile, template.Replace("INSERT_CODE_HERE",content))