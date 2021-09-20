module BCKG.Events

open BCKG.Domain
#if FABLE_COMPILER
open Thoth.Json
#else
open Thoth.Json.Net
#endif
open System


//Should these be in Domain?
type EventId = EventId of System.Guid
    with 
    override this.ToString() = match this with EventId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> EventId
    static member toString (EventId x) = x.ToString()
    static member fromString (x:string) = 
        let result = ref (System.Guid.NewGuid())
        let flag = System.Guid.TryParse(x,result)
        if flag then Some (!result |> EventId) else None


type EventOperation = 
    | Add
    | Modify
    with 
    override this.ToString() = 
        match this with 
            | Add -> "Add"
            | Modify -> "Modify"
    static member toString (et:EventOperation) = et.ToString()
    static member fromString (str:string) = 
        match str with 
        | "Add" -> EventOperation.Add
        | "Modify" -> EventOperation.Modify
        | _ -> failwithf "%s event type not recognized" str


type EventTarget = 
    //Part Events
    | PartEvent of PartId
    | PartTagEvent of PartId
    //Reagent Events
    | ReagentEvent of ReagentId
    | ReagentFileEvent of ReagentId
    | ReagentTagEvent of ReagentId
    //Experiment Events
    | ExperimentEvent of ExperimentId
    | ExperimentFileEvent of ExperimentId
    | ExperimentOperationEvent of ExperimentId
    | ExperimentSignalEvent of ExperimentId
    | ExperimentTagEvent of ExperimentId
    //File Events
    | FileEvent of FileId
    | TimeSeriesFileEvent of FileId
    | BundleFileEvent of FileId
    //Sample Events
    | SampleEvent of SampleId
    | SampleDeviceEvent of SampleId
    | SampleDataEvent of SampleId
    | SampleConditionEvent of SampleId
    | SampleTagEvent of SampleId
    | SampleReplicateEvent of SampleId
    //Cell Events
    | CellEvent of CellId
    | CellEntityEvent of CellId
    | CellFileEvent of CellId
    | CellTagEvent of CellId
    //Interaction
    | InteractionEvent of InteractionId

    //Observation
    | ObservationEvent of ObservationId

    //DerivedFrom 
    | DerivedFromEvent

    //Replay Log Events
    | StartLogEvent of string
    | FinishLogEvent of string

    | ProcessDataEvent of ExperimentId
    | ParseLayoutEvent of ExperimentId
    with
    static member toTargetTypeString (e:EventTarget) = 
        match e with 
        | PartEvent pid -> (PartId.GetType pid) + "PartEvent"
        | PartTagEvent pid -> (PartId.GetType pid) + "PartTagEvent"
        
        | ReagentEvent rid -> (ReagentId.GetType rid) + "ReagentEvent"
        | ReagentFileEvent rid -> (ReagentId.GetType rid) + "ReagentFileEvent"
        | ReagentTagEvent rid -> (ReagentId.GetType rid) + "ReagentTagEvent"
        
        | ExperimentEvent _ -> "ExperimentEvent"
        | ExperimentFileEvent _ -> "ExperimentFileEvent"
        | ExperimentOperationEvent _ -> "ExperimentOperationEvent"
        | ExperimentSignalEvent _ -> "ExperimentSignalEvent"
        | ExperimentTagEvent _ -> "ExperimentTagEvent"
        
        | SampleEvent _ -> "SampleEvent"
        | SampleDeviceEvent _ -> "SampleDeviceEvent"
        | SampleDataEvent _ -> "SampleDataEvent"
        | SampleConditionEvent _ -> "SampleConditionEvent"
        | SampleTagEvent _ -> "SampleTagEvent"
        | SampleReplicateEvent _ -> "SampleReplicateEvent"

        | CellEvent _ -> "CellEvent"
        | CellEntityEvent _ -> "CellEntityEvent"
        | CellTagEvent _ -> "CellTagEvent"
        | CellFileEvent _ -> "CellFileEvent"

        | InteractionEvent _ -> "InteractionEvent"

        | DerivedFromEvent -> "DerivedFromEvent"

        | FileEvent _ -> "FileEvent"
        | TimeSeriesFileEvent _ -> "TimeSeriesFileEvent"
        | BundleFileEvent _ -> "BundleFileEvent"
        
        | StartLogEvent _ -> "StartLogEvent"
        | FinishLogEvent _ -> "FinishLogEvent"
        | ProcessDataEvent _ -> "ProcessDataEvent"
        | ParseLayoutEvent _ -> "ParseLayoutEvent"
    
        | ObservationEvent _ -> "ObservationEvent"

    static member toTargetId (e:EventTarget) = 
        match e with 
        | PartEvent x -> x.ToString()
        | PartTagEvent x -> x.ToString()
        
        | ReagentEvent x -> x.ToString()
        | ReagentFileEvent x -> x.ToString()
        | ReagentTagEvent x -> x.ToString()
        
        | ExperimentEvent x -> x.ToString()
        | ExperimentFileEvent x -> x.ToString()
        | ExperimentOperationEvent x -> x.ToString()
        | ExperimentSignalEvent x -> x.ToString()
        | ExperimentTagEvent x -> x.ToString()
        
        | SampleEvent x -> x.ToString()
        | SampleDeviceEvent x -> x.ToString()
        | SampleDataEvent x -> x.ToString()
        | SampleConditionEvent x -> x.ToString()
        | SampleTagEvent x -> x.ToString()
        | SampleReplicateEvent x -> x.ToString()

        | CellEvent x -> x.ToString()
        | CellEntityEvent x -> x.ToString()
        | CellFileEvent x -> x.ToString()
        | CellTagEvent x -> x.ToString()

        | InteractionEvent x -> x.ToString()

        | FileEvent x -> x.ToString()
        | TimeSeriesFileEvent x -> x.ToString()
        | BundleFileEvent x -> x.ToString()
        
        | StartLogEvent x -> x
        | FinishLogEvent x -> x
        | ProcessDataEvent x -> x.ToString()
        | ParseLayoutEvent x -> x.ToString()  
        
        | DerivedFromEvent -> ""
    
        | ObservationEvent x -> x.ToString()

type Event = {
    id: EventId;
    operation:EventOperation; 
    target: EventTarget;
    timestamp:System.DateTime;
    change:string;
    message:string
    triggeredBy:EventId option
    user:string
    }
    with
    static member ToUiString (event:Event) = 
        let change = 
            if event.change = "" then [] else Decode.Auto.unsafeFromString<(string*string)list>(event.change) 
        let eventType = (event.operation.ToString() + " " + (EventTarget.toTargetTypeString event.target))
        eventType, event.timestamp, event.user, change

        //let change = 
        //    Decode.Auto.unsafeFromString<(string*string)list>(event.change)                
        //    |> List.map (fun (op, content) -> 
        //        let content' = if content.Length > 32 then content.[0..30]+"..." else content                
        //        //sprintf  "%s:%s" op content'
        //        op, content'
        //        )
        //    |> String.concat ""                
        //sprintf "[%A | %s]: %s%s" event.timestamp event.user change (if event.triggeredBy.IsSome then "(triggered)" else "")                
        

    static member Create (message:string) (operation:EventOperation) (target:EventTarget) (change:string) = 
        { Event.id = EventId.Create()
          Event.operation = operation
          Event.target = target
          Event.timestamp = System.DateTime.Now.ToUniversalTime()
          Event.message = message
          Event.change = change
          Event.triggeredBy = None
          Event.user = ""
        }
    
    static member empty = {
        id = EventId.Create();
        operation = EventOperation.Add;
        target = EventTarget.PartEvent(UserDefinedId.Create() |> UserDefinedPartId)
        timestamp = System.DateTime.Now
        change = ""
        message = ""
        triggeredBy = None
        user = ""
    }
    static member encode (e:Event) =
        
        
        let json = 
            Encode.object[
                "id", Encode.string (e.id.ToString())
                "operation",Encode.string (e.operation.ToString())
                "target", Encode.string (Encode.Auto.toString(0,e.target))
                "timestamp",Encode.string (e.timestamp.ToUniversalTime().ToString("o"))
                "message",Encode.string e.message
                "change",Encode.string e.change
                "triggeredBy", Encode.string (match e.triggeredBy with | Some(trig) -> trig.ToString() | None -> "")
                "user", Encode.string e.user
            ]
        json.ToString()
        //Encode.Auto.toString(0,e)
    static member Decoder = 
        Decode.object
            (fun get -> 
                {
                    id = get.Required.Field "id" (Decode.string) |> System.Guid |> EventId;
                    operation = get.Required.Field "operation" (Decode.string) |> EventOperation.fromString;
                    target = 
                        let targetstring = get.Required.Field "target" (Decode.string)
                        Decode.Auto.unsafeFromString<EventTarget>(targetstring)
                    timestamp = (get.Required.Field "timestamp" (Decode.string) |> System.DateTime.Parse).ToUniversalTime() 
                    message = get.Required.Field "message" (Decode.string)
                    change = get.Required.Field "change" (Decode.string)
                    triggeredBy = 
                        let trigopt = get.Required.Field "triggeredBy" (Decode.string)
                        match trigopt with 
                        | "" -> None
                        | _ -> Some(trigopt |> System.Guid |> EventId)
                    user = get.Required.Field "user" (Decode.string)
                }
            )
    static member serialize_events (events:Event list) = 
        events
        |> List.map Event.encode
        |> String.concat ","
        |> sprintf "[%s]"

    static member serialize_triggeredByProperty (evidopt:EventId option) =
        match evidopt with
        | Some(evid) -> evid.ToString()
        | None -> ""
    static member decode (str:string) =
        let result = Decode.fromString Event.Decoder str
        match result with 
        | Ok(res) -> res
        | Error e -> failwithf "Error decoding: %s" e 
    
    static member decode_list (str:string) = 
        let listdecoder = Decode.list Event.Decoder
        let result = Decode.fromString listdecoder str
        match result with 
        | Ok(res) -> res
        | Error e -> failwithf "Error decoding: %s" e 

let private serializeEventChange (event:(string * string)list) = 
    let event_enc = 
        Encode.list (event |> List.map (fun (x,y) -> Encode.list [Encode.string x; Encode.string y]))
    Encode.toString 0 event_enc

let private deserializeEventChange (eventlist:string) :(string * string)list =       
    let result = Decode.fromString (Decode.list (Decode.list Decode.string)) eventlist
    match result with 
    | Ok(res) -> 
        res
        |> List.map (fun x -> 
            match x with 
            | [a;b] -> (a,b)
            | _ -> failwithf "Malformed Event list: %s" eventlist
            )
    | Error e -> failwithf "Error decoding: %s" eventlist


//File Events
let private addFileRef (message:string) (target:EventTarget) (current:FileRef) = 
    let changemap = [
        (FileRef.addFileId,current.fileId.ToString());
        (FileRef.addType,FileType.toString current.Type);
        (FileRef.addFileName,current.fileName)
    ]
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Add) target change

let private modifyFileRef (message:string) (target:EventTarget) (current:FileRef) (modified:FileRef) = 
    let changemap = []
    let changemap = if current.fileName <> modified.fileName then [(FileRef.removeFileName,current.fileName);(FileRef.addFileName,modified.fileName)]@changemap else changemap
    let changemap = if current.Type <> modified.Type then [(FileRef.removeType,FileType.toString current.Type);(FileRef.addType,FileType.toString modified.Type)]@changemap else changemap
    match changemap with 
    | [] -> None 
    | _ ->
        let changemap = (FileRef.targetFileId,current.fileId.ToString())::changemap
        let change = serializeEventChange changemap
        let e = Event.Create message (EventOperation.Modify) target change
        Some(e)
 
let private removeFileRef (message:string) (target:EventTarget) (current:FileRef) = 
    let changemap = [
        (FileRef.removeFileId,current.fileId.ToString());
    ]
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Modify) target change
        
(*Part Events*)
let addPart (message:string) (current:Part) = 
    let changemap = [
        (Part.addType,Part.GetType(current));
        (Part.addSequence,current.getProperties.sequence);
        (Part.addName,current.name)
        (Part.addDeprecate,current.deprecated.ToString())
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.PartEvent(current.id)
    Event.Create message (EventOperation.Add) target change    

let modifyPart (message:string) (current:Part) (modified:Part) = 
    
    let changemap = []
    let changemap = if current.sequence <> modified.sequence then [(Part.removeSequence,current.sequence);(Part.addSequence,modified.sequence)]@changemap else changemap
    let changemap = if current.name <> modified.name then [(Part.removeName,current.name);(Part.addName,modified.name)]@changemap else changemap
    let changemap = if current.getType <> modified.getType then [(Part.removeType, current.getType);(Part.addType,modified.getType)]@changemap else changemap
    let changemap = if (current.deprecated <> modified.deprecated) then [(Part.removeDeprecate,current.deprecated.ToString());(Part.addDeprecate,modified.deprecated.ToString())]@changemap else changemap
        
    match changemap with 
    | [] -> None
    | _ -> 
        let change = serializeEventChange changemap
        let target = EventTarget.PartEvent(current.id)
        Some(Event.Create message (EventOperation.Modify) target change)


let addPartTag (message:string) (partId:PartId) (existingtags:Tag list) (tag:Tag) =     
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) -> None 
    | None -> 
        let changemap = [
            (Part.addTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.PartTagEvent(partId)
        (Event.Create message (EventOperation.Add) target change) |> Some
   
let removePartTag (message:string) (partId:PartId) (existingtags:Tag list) (tag:Tag) = 
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) ->  
        let changemap = [
            (Part.removeTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.PartTagEvent(partId)
        (Event.Create message (EventOperation.Modify) target change) |> Some    
    | None -> None
    
    

(*Reagent Events*)
let private addReagentProperties (current:ReagentProperties) = 
    let changemap = [
           (Reagent.addName,current.name);
           (Reagent.addNotes,current.notes);
           (Reagent.addDeprecate,current.deprecated.ToString())
       ]
    let changemap = match current.barcode with | Some(b) -> (Reagent.addBarcode,b.ToString())::changemap | None -> changemap
    changemap

let private modifyReagentProperties (current:ReagentProperties) (modified:ReagentProperties) = 
    let changemap = []
    let changemap = if current.name <> modified.name then [(Reagent.removeName,current.name);(Reagent.addName,modified.name)]@changemap else changemap
    let changemap = if current.notes <> modified.notes then [(Reagent.removeNotes,current.notes);(Reagent.addNotes,modified.notes)]@changemap else changemap
    let changemap = if (current.deprecated <> modified.deprecated) then [(Reagent.removeDeprecate,current.deprecated.ToString());(Reagent.addDeprecate,modified.deprecated.ToString())]@changemap else changemap
    
    let changemap  = 
        if current.barcode <> modified.barcode then 
            (match current.barcode with | Some(b) -> [(Reagent.removeBarcode,b.ToString())] | None -> [])
            @(match modified.barcode with | Some(b) -> [(Reagent.addBarcode,b.ToString())] | None -> [])
            @changemap
        else changemap
    
    changemap

let addReagent (message:string) (current:Reagent) = 
    
    let changemap = addReagentProperties current.getProperties
    let changemap = (Reagent.addType,current.getType)::changemap
    
    let changemap = 
        match current with 
        | Chemical cr -> (Reagent.addChemicalType,ChemicalType.ToString cr.Type)::changemap
        | DNA dna  -> 
            let changemap = (Reagent.addDNAType,DNAType.ToString dna.Type)::changemap
            let changemap = match dna.sequence with | "" ->  changemap  | _ -> (Reagent.addSequence,dna.sequence)::changemap
            let changemap = match dna.concentration with | Some(conc) -> (Reagent.addConcentration,Concentration.toString conc)::changemap | None -> changemap
            changemap
        | RNA rna -> 
            let changemap = (Reagent.addRNAType,RNAType.ToString rna.Type)::changemap
            let changemap = match rna.sequence with | "" ->  changemap  | _ -> (Reagent.addSequence,rna.sequence)::changemap
            changemap
        | Protein prot -> (Reagent.addIsReporter,prot.isReporter.ToString())::changemap
        | GenericEntity _ -> changemap


        
    
    let change = serializeEventChange changemap
    let target = EventTarget.ReagentEvent(current.id)
    let reagentevent = Event.Create message (EventOperation.Add) target change

    reagentevent

let modifyReagent (message:string) (current:Reagent) (modified:Reagent) = 
    (*Check if two reagents have the same id?*)
    
    let changemap = modifyReagentProperties current.getProperties modified.getProperties
    let changemap = 
        match current.getType <> modified.getType with 
        | true -> 
            let changemap = [(Reagent.removeType,current.getType);(Reagent.addType,modified.getType)]@changemap
            let changemap =
                match current with 
                | Chemical chem -> 
                    (Reagent.removeChemicalType,ChemicalType.ToString chem.Type)::changemap
                | DNA dna ->
                    let dnaconc = match dna.concentration with | Some(conc) -> [(Reagent.removeConcentration,Concentration.toString conc)] | None -> []
                    [(Reagent.removeSequence,dna.sequence);(Reagent.removeDNAType,DNAType.ToString dna.Type)]@dnaconc@changemap
                | RNA rna ->
                    [(Reagent.removeSequence,rna.sequence);(Reagent.removeRNAType,RNAType.ToString rna.Type)]@changemap
                | Protein prot -> 
                    (Reagent.removeIsReporter,prot.isReporter.ToString())::changemap
                | GenericEntity ge -> changemap

            let changemap =
                match modified with 
                | Chemical chem -> 
                    (Reagent.addChemicalType,ChemicalType.ToString chem.Type)::changemap
                | DNA dna ->
                    let dnaconc = match dna.concentration with | Some(conc) -> [(Reagent.addConcentration,Concentration.toString conc)] | None -> []
                    [(Reagent.addSequence,dna.sequence);(Reagent.addDNAType,DNAType.ToString dna.Type)]@dnaconc@changemap
                | RNA rna ->
                    [(Reagent.addSequence,rna.sequence);(Reagent.addRNAType,RNAType.ToString rna.Type)]@changemap
                | Protein prot -> 
                    (Reagent.addIsReporter,prot.isReporter.ToString())::changemap
                | GenericEntity ge -> changemap
            changemap
        | false -> 
            match (current,modified) with 
            | Chemical(c),Chemical(m) -> 
                if (ChemicalType.ToString c.Type) <> (ChemicalType.ToString m.Type) then 
                    [(Reagent.removeChemicalType,(ChemicalType.ToString c.Type));(Reagent.addChemicalType,(ChemicalType.ToString m.Type))]@changemap
                else 
                    changemap
            | DNA(c),DNA(m) -> 
                let changemap = 
                    if (DNAType.ToString c.Type) <> (DNAType.ToString m.Type) then 
                        [(Reagent.removeDNAType,(DNAType.ToString c.Type));(Reagent.addDNAType,(DNAType.ToString m.Type))]@changemap 
                    else changemap
                let changemap  = 
                    if c.sequence <> m.sequence then 
                        (match c.sequence with | "" -> [] | _ -> [(Reagent.removeSequence,c.sequence)] )
                        @(match m.sequence with | "" -> [] | _ -> [(Reagent.addSequence,m.sequence)] )
                        @changemap
                    else changemap
                
                let changemap  = 
                    if c.concentration <> m.concentration then 
                        (match c.concentration with | Some(conc) -> [(Reagent.removeConcentration,conc.ToString())] | None -> [])
                        @(match m.concentration with | Some(conc) -> [(Reagent.addConcentration,conc.ToString())] | None -> [])
                        @changemap
                    else changemap
                changemap
            | RNA(c),RNA(m) -> 
                let changemap = 
                    if (RNAType.ToString c.Type) <> (RNAType.ToString m.Type) then 
                        [(Reagent.removeRNAType,(RNAType.ToString c.Type));(Reagent.addRNAType,(RNAType.ToString m.Type))]@changemap 
                    else changemap
                let changemap  = 
                    if c.sequence <> m.sequence then 
                        (match c.sequence with | "" -> [] | _ -> [(Reagent.removeSequence,c.sequence)] )
                        @(match m.sequence with | "" -> [] | _ -> [(Reagent.addSequence,m.sequence)] )
                        @changemap
                    else changemap
                changemap
            | Protein(c),Protein(m) -> 
                let changemap = 
                    if (c.isReporter) <> (m.isReporter) then 
                        [(Reagent.removeIsReporter,c.isReporter.ToString());(Reagent.addIsReporter,m.isReporter.ToString())]@changemap 
                    else changemap               
                changemap
            | GenericEntity(c),GenericEntity(m) -> changemap
            | _ -> failwithf "Can't ever reach this state"
             
    match changemap with 
    | [] -> None 
    | _ -> 
        let change = serializeEventChange changemap
        let target = EventTarget.ReagentEvent(current.id)
        Some(Event.Create message (EventOperation.Modify) target change)    


//Reagent File Events
let addReagentFile (message:string) (reagentId:ReagentId) (existing:FileRef list) (file:FileRef) = 
    let fileExists = existing |> List.tryFind (fun f -> f.fileId = file.fileId)
    match fileExists with 
    | Some (existingfile) -> modifyFileRef message (ReagentFileEvent(reagentId)) existingfile file
    | None -> (addFileRef message (ReagentFileEvent(reagentId)) file) |> Some

let unlinkReagentFile (message:string) (reagentId:ReagentId) (file:FileRef) = 
    (removeFileRef message (ReagentFileEvent(reagentId)) file)        

//Reagent Tag Events
let addReagentTag (message:string) (reagentId:ReagentId) (existingtags:Tag list) (tag:Tag) =     
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) -> None 
    | None -> 
        let changemap = [
            (Reagent.addTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.ReagentTagEvent(reagentId)
        (Event.Create message (EventOperation.Add) target change) |> Some
   
let removeReagentTag (message:string) (reagentId:ReagentId) (existingtags:Tag list) (tag:Tag) = 
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) ->  
        let changemap = [
            (Reagent.removeTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.ReagentTagEvent(reagentId)
        (Event.Create message (EventOperation.Modify) target change) |> Some    
    | None -> None

(*Experiment Events*)
let addExperiment (message:string) (current:Experiment) = 
    
    let changemap = [
        (Experiment.addName,current.name);
        (Experiment.addNotes,current.notes);
        (Experiment.addType,ExperimentType.toString current.Type)
        (Experiment.addDeprecate,current.deprecated.ToString())
    ]
    
    let change = serializeEventChange changemap
    let target = EventTarget.ExperimentEvent(current.id)
    
    let exptevent = Event.Create message (EventOperation.Add) target change
    

    exptevent

let modifyExperiment (message:string) (current:Experiment) (modified:Experiment) = 
    let changemap = []
    let changemap = if current.name <> modified.name then [(Experiment.removeName,current.name);(Experiment.addName,modified.name)]@changemap else changemap
    let changemap = if current.notes <> modified.notes then [(Experiment.removeNotes,current.notes);(Experiment.addNotes,modified.notes)]@changemap else changemap
    let changemap = if current.Type <> modified.Type then [(Experiment.removeType,ExperimentType.toString current.Type);(Experiment.addType,ExperimentType.toString  modified.Type)]@changemap else changemap
    let changemap = if (current.deprecated <> modified.deprecated) then [(Experiment.removeDeprecate,current.deprecated.ToString());(Experiment.addDeprecate,modified.deprecated.ToString())]@changemap else changemap
    
    
    let change = serializeEventChange changemap
    let target = EventTarget.ExperimentEvent(current.id)
    
    
    match changemap with 
    | [] -> None
    | _ -> (Event.Create message (EventOperation.Modify) target change) |> Some 


//Experiment Tag Events    
let addExperimentTag (message:string) (experimentId:ExperimentId) (existingtags:Tag list) (tag:Tag) =     
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) -> None 
    | None -> 
        let changemap = [
            (Experiment.addTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.ExperimentTagEvent(experimentId)
        (Event.Create message (EventOperation.Add) target change) |> Some
   
let removeExperimentTag (message:string) (experimentId:ExperimentId) (existingtags:Tag list) (tag:Tag) = 
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) ->  
        let changemap = [
            (Experiment.removeTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.ExperimentTagEvent(experimentId)
        (Event.Create message (EventOperation.Modify) target change) |> Some    
    | None -> None

//Experiment Operation Events
let private addExperimentOperationEvent (message:string) (experimentId:ExperimentId) (current:ExperimentOperation) = 
    let changemap = [
        (ExperimentOperation.addTimestamp,current.timestamp.ToUniversalTime().ToString("o"));
        (ExperimentOperation.addExperimentOperationId,current.id.ToString());
        (ExperimentOperation.addType,ExperimentOperationType.toString current.Type)
    ]
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Add) (ExperimentOperationEvent(experimentId)) change

let private modifyExperimentOperationEvent (message:string) (experimentId:ExperimentId) (current:ExperimentOperation) (modified:ExperimentOperation) =
    let changemap = []
    let changemap = if current.timestamp <> modified.timestamp then [("--timestamp",current.timestamp.ToUniversalTime().ToString("o"));("++timestamp",modified.timestamp.ToUniversalTime().ToString("o"))]@changemap else changemap
    let changemap = if current.Type <> modified.Type then [("--type",ExperimentOperationType.toString current.Type);("++type",ExperimentOperationType.toString modified.Type)]@changemap else changemap
    match changemap with 
    | [] -> None
    | _ -> 
        let changemap = (ExperimentOperation.targetExperimentOperationId,current.id.ToString())::changemap
        let change = serializeEventChange changemap
        let e = Event.Create message (EventOperation.Modify) (ExperimentOperationEvent(experimentId)) change
        Some(e)    

let private removeExperimentOperationEvent (message:string) (experimentId:ExperimentId) (current:ExperimentOperation) = 
    let changemap = [
        (ExperimentOperation.removeExperimentOperationId,current.id.ToString());
    ]
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Modify) (ExperimentOperationEvent(experimentId)) change

let addExperimentOperation (message:string) (experimentId:ExperimentId) (existing:ExperimentOperation[]) (op:ExperimentOperation) = 
    let opexist = existing |> Array.tryFind (fun eo -> eo.id = op.id)
    match opexist with 
    | Some(eo) -> modifyExperimentOperationEvent message experimentId eo op
    | None -> (addExperimentOperationEvent message experimentId op) |> Some

let removeExperimentOperation (message:string) (experimentId:ExperimentId) (existing:ExperimentOperation[]) (op:ExperimentOperation) = 
    let opexist = existing |> Array.tryFind (fun eo -> eo.id = op.id)
    match opexist with 
    | Some(eo) -> (removeExperimentOperationEvent message experimentId eo) |> Some
    | None -> None

//Experiment Signal Events
let private addSignalEvent (message:string) (experimentId:ExperimentId) (current:Signal) = 
    let changemap = 
        [ ("++signalId",current.id.ToString());
          ("type",(current.settings |> SignalSettings.toTypeString))
        ]
        |> List.append (match current.units with Some x -> [("++units"), x] | None -> [])
    let changemap = 
        match current.settings with 
        | PlateReaderFluorescence prf -> 
            [("++gain",prf.gain.ToString());("++emission",prf.emissionFilter |> PlateReaderFilter.toString);("++excitation", prf.excitationFilter |> PlateReaderFilter.toString)]
            @changemap
        | PlateReaderAbsorbance pra -> 
            [("++gain",pra.gain.ToString());("++wavelength",pra.wavelength.ToString());("++correction",pra.correction.ToString())]
            @changemap
        | PlateReaderLuminescence -> changemap
        | PlateReaderTemperature -> changemap
        | Titre -> changemap
        | GenericSignal _ -> changemap
        
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Add) (ExperimentSignalEvent(experimentId)) change

let private  removeSignalEvent (message:string) (experimentId:ExperimentId) (current:Signal) =
    let changemap = [
        ("--signalId",current.id.ToString())
    ]
    let change = serializeEventChange changemap
    Event.Create message (EventOperation.Modify) (ExperimentSignalEvent(experimentId)) change

let addExperimentSignal (message:string) (experimentId:ExperimentId) (existing:Signal list) (signal:Signal) = 
    let sigexists = existing |> List.tryFind (fun s -> s.id = signal.id)
    match sigexists with 
    | Some _ -> 
        printfn "[WARNING] Signal %s already exists. Skipping add experiment signal" (signal.id.ToString())
        None
    | None -> (addSignalEvent message experimentId signal) |> Some

let removeExperimentSignal (message:string) (experimentId:ExperimentId) (existing:Signal list) (signal:Signal) = 
    let sigexists = existing |> List.tryFind (fun s -> s.id = signal.id)
    match sigexists with 
    | Some _ -> (removeSignalEvent message experimentId signal) |> Some
    | None -> None
    

//Experiment File Events
let addExperimentFile (message:string) (experimentId:ExperimentId) (existing:FileRef list) (file:FileRef) = 
    let fileExists = existing |> List.tryFind (fun f -> f.fileId = file.fileId)
    match fileExists with 
    | Some (existingfile) -> modifyFileRef message (ExperimentFileEvent(experimentId)) existingfile file
    | None -> (addFileRef message (ExperimentFileEvent(experimentId)) file) |> Some

let unlinkExperimentFile (message:string) (experimentId:ExperimentId) (file:FileRef) = 
    (removeFileRef message (ExperimentFileEvent(experimentId)) file)


(*Sample Events*)
let addSample (message:string) (current:Sample) = 
    let changemap = [
        (Sample.addExperimentId,current.experimentId.ToString());
        (Sample.addMetaType,(current.meta |> SampleMeta.toStringType));
        (Sample.addDeprecate,current.deprecated.ToString())
    ]
    let changemap = 
        match current.meta with
        | PlateReaderMeta (prmd) -> 
            
            let virtualWell = [(Sample.addVirtualWellRow,prmd.virtualWell.row.ToString());(Sample.addVirtualWellCol,prmd.virtualWell.col.ToString())]
                
            let physicalPlateName = 
                match prmd.physicalPlateName with 
                | Some(n) -> [(Sample.addPhysicalPlateName,n)] 
                | _ -> []
            
            let physicalWell = 
                match prmd.physicalWell with 
                | Some(p) -> 
                    [(Sample.addPhysicalWellRow,p.row.ToString());(Sample.addPhysicalWellCol,p.col.ToString())]
                | None -> []

            changemap@virtualWell@physicalPlateName@physicalWell
                
        | _ -> changemap
    
    let change = serializeEventChange changemap
    let target = EventTarget.SampleEvent(current.id)
    let sampleevent = Event.Create message (EventOperation.Add) target change

    sampleevent

let modifySample (message:string) (current:Sample) (modified:Sample) = 
    let changemap = []
    let changemap = if current.experimentId <> modified.experimentId then [(Sample.removeExperimentId,current.experimentId.ToString());(Sample.addExperimentId,modified.experimentId.ToString())]@changemap else changemap 
    let changemap = if (current.meta |> SampleMeta.toStringType) <> (modified.meta |> SampleMeta.toStringType) then [("--metaType",(current.meta |> SampleMeta.toStringType));(Sample.addMetaType,(modified.meta |> SampleMeta.toStringType))]@changemap  else changemap    
    let changemap = if (current.deprecated <> modified.deprecated) then [(Sample.removeDeprecate,current.deprecated.ToString());(Sample.addDeprecate,modified.deprecated.ToString())]@changemap else changemap

    let changemap = 
        match current.meta with 
        | PlateReaderMeta(currentprmd) -> 
            match modified.meta with 
            | PlateReaderMeta(modifiedprmd) -> 
                let physicalPlateNameChange =  
                    if currentprmd.physicalPlateName <> modifiedprmd.physicalPlateName then
                        let currentplatename = match currentprmd.physicalPlateName with | Some(n) -> [("--physicalPlateName",n)] | None -> []
                        let modifiedplatename = match modifiedprmd.physicalPlateName with | Some(n) -> [("++physicalPlateName",n)] | None -> []
                        currentplatename@modifiedplatename
                    else []
                
                let physicalWellChange = 
                    match currentprmd.physicalWell with 
                    | Some(currpw)-> 
                        match modifiedprmd.physicalWell with 
                        | Some(modpw) -> 
                            let rowchange = if currpw.row <> modpw.row then [(Sample.removePhysicalWellRow,currpw.row.ToString());(Sample.addPhysicalWellRow,modpw.row.ToString())] else []
                            let colchange = if currpw.col <> modpw.col then [(Sample.removePhysicalWellCol,currpw.col.ToString());(Sample.addPhysicalWellCol,modpw.col.ToString())] else []
                            rowchange@colchange
                        | None -> 
                            [(Sample.removePhysicalWellRow,currpw.row.ToString());(Sample.removePhysicalWellCol,currpw.col.ToString())]                                                
                    | None -> 
                        match modifiedprmd.physicalWell with 
                        | Some(modpw) -> 
                            [(Sample.addPhysicalWellRow,modpw.row.ToString());(Sample.addPhysicalWellCol,modpw.col.ToString())]
                        | None -> []
                
                let virtualwellrowchange = if (currentprmd.virtualWell.row <> modifiedprmd.virtualWell.row) then [(Sample.removeVirtualWellRow,currentprmd.virtualWell.row.ToString());(Sample.addVirtualWellRow,modifiedprmd.virtualWell.row.ToString())] else []
                let virtualwellcolchange = if (currentprmd.virtualWell.col <> modifiedprmd.virtualWell.col) then [(Sample.removeVirtualWellCol,currentprmd.virtualWell.col.ToString());(Sample.addVirtualWellCol,modifiedprmd.virtualWell.col.ToString())] else []
                virtualwellrowchange@virtualwellcolchange@physicalWellChange@physicalPlateNameChange@changemap
            
            | MissingMeta -> 
                let virtualwellchange = [(Sample.removeVirtualWellRow,currentprmd.virtualWell.row.ToString());(Sample.removeVirtualWellCol,currentprmd.virtualWell.col.ToString())]
                
                let physicalPlateNameChange = 
                    match currentprmd.physicalPlateName with | Some(n) -> [(Sample.removePhysicalPlateName,n)] | None -> []
                
                let physicalwellchange = 
                    match currentprmd.physicalWell with | Some(pos) -> [(Sample.removePhysicalWellRow,pos.row.ToString());(Sample.removePhysicalWellCol,pos.col.ToString())] | None -> []
                virtualwellchange@physicalPlateNameChange@physicalwellchange@changemap

        | MissingMeta -> 
            match modified.meta with
            | PlateReaderMeta(modifiedprmd) -> 
                let virtualwellchange = [(Sample.addVirtualWellRow,modifiedprmd.virtualWell.row.ToString());(Sample.addVirtualWellCol,modifiedprmd.virtualWell.col.ToString())]
                
                let physicalPlateNameChange = 
                    match modifiedprmd.physicalPlateName with | Some(n) -> [(Sample.addPhysicalPlateName,n)] | None -> []
                
                let physicalwellchange = 
                    match modifiedprmd.physicalWell with | Some(pos) -> [(Sample.addPhysicalWellRow,pos.row.ToString());(Sample.addPhysicalWellCol,pos.col.ToString())] | None -> []
                virtualwellchange@physicalPlateNameChange@physicalwellchange@changemap
            | MissingMeta -> changemap

    
    match changemap with 
    | [] -> None
    | _ -> 
        let change = serializeEventChange changemap
        let target = EventTarget.SampleEvent(current.id)
        let e = Event.Create message (EventOperation.Modify) target change
        e |> Some   

//Sample Condition Events
let private addSampleCondition (message:string) (condition:Condition) = 
    let changemap = [
        (Condition.addReagentId,condition.reagentId.ToString());
        (Condition.addReagentEntity,ReagentId.GetType condition.reagentId)
        (Condition.addValue, (Concentration.toString condition.concentration))
    ]
    let changemap = 
        match condition.time with 
        | Some(time) -> 
            [(Condition.addTime,(Time.getValue time).ToString());(Condition.addTimeUnits,(Time.getUnit time))]@changemap
        | None -> changemap
    let change = serializeEventChange changemap
    let target = EventTarget.SampleConditionEvent(condition.sampleId)

    Event.Create message (EventOperation.Add) target change

let private modifySampleCondition (message:string) (current:Condition) (modified:Condition) = 
    let changemap = []
    let changemap = 
        if (current.reagentId <> modified.reagentId) then 
            [(Condition.removeReagentId,current.reagentId.ToString());(Condition.removeReagentEntity,ReagentId.GetType current.reagentId);(Condition.addReagentId,modified.reagentId.ToString());(Condition.addReagentEntity,ReagentId.GetType modified.reagentId)]@changemap 
        else changemap
    let changemap = if ( (Concentration.toString current.concentration) <> (Concentration.toString modified.concentration)) then [(Condition.removeValue,Concentration.toString current.concentration);(Condition.addValue,Concentration.toString modified.concentration)]@changemap else changemap
    
    let changemap = 
        match (current.time,modified.time) with 
        | Some(currenttime),Some(modifiedtime) ->
            if (Time.toString currenttime) <> (Time.toString modifiedtime) then 
                let additions = [(Condition.addTime,(Time.getValue modifiedtime).ToString());(Condition.addTimeUnits,(Time.getUnit modifiedtime))]
                let removals = [(Condition.removeTime,(Time.getValue currenttime).ToString());(Condition.removeTimeUnits,(Time.getUnit currenttime))]
                additions@removals@changemap
            else 
                changemap
        | None,Some(modifiedtime) ->
            [(Condition.addTime,(Time.getValue modifiedtime).ToString());(Condition.addTimeUnits,(Time.getUnit modifiedtime))]@changemap
        | Some(currenttime),None ->
            [(Condition.removeTime,(Time.getValue currenttime).ToString());(Condition.removeTimeUnits,(Time.getUnit currenttime))]@changemap
        | None,None -> changemap

    match changemap with 
    | [] -> None 
    | _ -> 
        let change = serializeEventChange changemap
        let target = EventTarget.SampleConditionEvent(current.sampleId)
        let e = Event.Create message (EventOperation.Modify) target change
        Some(e)
    
let private removeSampleCondition (message:string) (condition:Condition) =
    let changemap = [
        (Condition.removeReagentId,condition.reagentId.ToString())
        (Condition.removeReagentEntity,ReagentId.GetType condition.reagentId)
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.SampleConditionEvent(condition.sampleId)
    Event.Create message (EventOperation.Modify) target change

let addCondition (message:string) (existing:Condition list) (condition:Condition) = 
    let conditionExists = existing |> List.tryFind (fun c -> c.reagentId = condition.reagentId)
    match conditionExists with 
    | Some(existingcondition) -> modifySampleCondition message existingcondition condition
    | None -> (addSampleCondition message condition) |> Some

let removeCondition (message:string) (condition:Condition) = 
    (removeSampleCondition message condition)

//Sample Device Events
let private addCellStore (message:string) (sampleId:SampleId) (device:CellId, (cellDensity:float option, cellPreSeeding: float option)) = 
    
    let changemap = 
        let cellId = [(Sample.addCellId,device.ToString())]
        let cellDens = match cellDensity with Some x -> [("++cellDensity", x.ToString())] | None -> []
        let cellPre = match cellPreSeeding with Some x -> [("++cellPreSeeding", x.ToString())] | None -> []
        cellId@cellDens@cellPre
    let change = serializeEventChange changemap 
    let target = EventTarget.SampleDeviceEvent(sampleId)

    Event.Create message (EventOperation.Add) target change


let private addDevice (message: string) (sampleId:SampleId) (device:SampleDevice) = 
    let changemap = 
        let cellId =  [(Sample.addCellId, device.cellId.ToString())]
        let cellDensity = match device.cellDensity with Some x -> [("++cellDensity", x.ToString())] | None -> []
        let cellPreSeeding = match device.cellPreSeeding with Some x -> [("++cellPreSeeding", x.ToString())] | None -> []
        cellId@cellDensity@cellPreSeeding
    let change = serializeEventChange changemap
    let target = EventTarget.SampleDeviceEvent(sampleId)

    Event.Create message (EventOperation.Add) target change

let private removeDevice (message:string) (sampleId:SampleId) (current:CellId) = 
    let changemap = [
        (Sample.removeCellId,current.ToString())
    ]    
    
    let change = serializeEventChange changemap
    let target = EventTarget.SampleDeviceEvent(sampleId)

    Event.Create message (EventOperation.Modify) target change

let addSampleCellStore (message:string) (sampleId:SampleId) (existing:(CellId * (float option * float option)) list) (current:(CellId * (float option * float option))) = 
    let cellExists = existing |> List.tryFind (fun e -> e = current)
    match cellExists with 
    | Some _ -> None
    | None -> (addCellStore message sampleId current) |> Some

let addSampleDevice (message:string) (sampleId:SampleId) (existing: SampleDevice list) (current:SampleDevice) = 
    let deviceExists = existing |> List.tryFind (fun e -> (e.cellId = current.cellId) && (e.sampleId = current.sampleId) && (e.cellDensity = current.cellDensity) && (e.cellPreSeeding = current.cellPreSeeding))
    match deviceExists with
    | Some _ -> None
    | None -> (addDevice message sampleId current) |> Some

let removeSampleDevice (message:string) (sampleId:SampleId) (current:CellId) = 
    (removeDevice message sampleId current)

//Sample File Events
let addSampleFile (message:string) (sampleId:SampleId) (existing:FileRef list) (file:FileRef) = 
    let fileExists = existing |> List.tryFind (fun f -> f.fileId = file.fileId)
    match fileExists with 
    | Some (existingfile) -> modifyFileRef message (SampleDataEvent(sampleId)) existingfile file
    | None -> (addFileRef message (SampleDataEvent(sampleId)) file) |> Some

let unlinkSampleFile (message:string) (sampleId:SampleId) (file:FileRef) = 
    (removeFileRef message (SampleDataEvent(sampleId)) file)

//Sample Replicate Event
let addSampleReplicate (message:string) (sampleId:SampleId) (replicateId:ReplicateId) = 
    let changemap = [
        (Sample.addReplicate, replicateId.ToString())
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.SampleReplicateEvent(sampleId)
    Event.Create message (EventOperation.Add) target change

let unlinkSampleReplicate (message:string) (sampleId:SampleId) (replicateId:ReplicateId) = 
    let changemap = [
        (Sample.removeReplicate, replicateId.ToString())
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.SampleReplicateEvent(sampleId)
    Event.Create message (EventOperation.Modify) target change

//Sample Tag Events
let addSampleTag (message:string) (sampleId:SampleId) (existingtags:Tag list) (tag:Tag) =     
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) -> None 
    | None -> 
        let changemap = [
            (Sample.addTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.SampleTagEvent(sampleId)
        (Event.Create message (EventOperation.Add) target change) |> Some
   
let removeSampleTag (message:string) (sampleId:SampleId) (existingtags:Tag list) (tag:Tag) = 
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) ->  
        let changemap = [
            (Sample.removeTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.SampleTagEvent(sampleId)
        (Event.Create message (EventOperation.Modify) target change) |> Some    
    | None -> None

(*Cell Events*)
let private addCellProperties (current:CellProperties) = 
    let changemap = [
           (Cell.addName,current.name);
           (Cell.addNotes,current.notes);
           (Cell.addGenotype,current.genotype)
           (Cell.addDeprecate,current.deprecated.ToString())
       ]
    let changemap = match current.barcode with | Some(b) -> (Cell.addBarcode,b.ToString())::changemap | None -> changemap
    changemap

let private modifyCellProperties (current:CellProperties) (modified:CellProperties) = 
    let changemap = []
    let changemap = if current.name <> modified.name then [(Cell.removeName,current.name);(Cell.addName,modified.name)]@changemap else changemap
    let changemap = if current.notes <> modified.notes then [(Cell.removeNotes,current.notes);(Cell.addNotes,modified.notes)]@changemap else changemap
    let changemap = if current.genotype <> modified.genotype then [(Cell.removeGenotype,current.genotype);(Cell.addGenotype,modified.genotype)]@changemap else changemap
    let changemap = if (current.deprecated <> modified.deprecated) then [(Cell.removeDeprecate,current.deprecated.ToString());(Cell.addDeprecate,modified.deprecated.ToString())]@changemap else changemap
    
    let changemap  = 
        if current.barcode <> modified.barcode then 
            (match current.barcode with | Some(b) -> [(Cell.removeBarcode,b.ToString())] | None -> [])
            @(match modified.barcode with | Some(b) -> [(Cell.addBarcode,b.ToString())] | None -> [])
            @changemap
        else changemap  

    changemap

let addCell (message:string) (cell:Cell) = 
    let changemap = (Cell.addType,cell.getType)::(addCellProperties (Cell.GetProperties cell))
    let changemap = 
        match cell with 
        | Prokaryote prok -> 
            (Cell.addProkaryoteType,ProkaryoteType.ToString prok.Type)::changemap
        | Eukaryote eu -> changemap //TODO: fixme
    let change = serializeEventChange changemap 
    let target = EventTarget.CellEvent(cell.id)

    Event.Create message (EventOperation.Add) target change

let modifyCell (message:string) (current:Cell) (modified:Cell) = 
    let changemap = modifyCellProperties (Cell.GetProperties current) (Cell.GetProperties modified)    
    //No need to check for changes in Type - we can do that when we extend BCKG.
    match changemap with 
    | [] -> None
    | _ -> 
        let change = serializeEventChange changemap 
        let target = EventTarget.CellEvent(current.id)
        Event.Create message (EventOperation.Add) target change
        |> Some

let private addEntity (message:string) (cellEntity:CellEntity) = 
    let changemap = [
        (CellEntity.addCompartment,CellCompartment.toString cellEntity.compartment)
        (CellEntity.addEntity,cellEntity.entity.ToString())
        (CellEntity.addEntityType,ReagentId.GetType cellEntity.entity)
    ]
    let change = serializeEventChange changemap 
    let target = EventTarget.CellEntityEvent(cellEntity.cellId)
    Event.Create message (EventOperation.Add) target change

let private modifyEntity (message:string) (current:CellEntity) (modified:CellEntity) = 
    let changemap = 
        if current.compartment <> modified.compartment then            
            [
                (CellEntity.targetEntity,current.entity.ToString())
                (CellEntity.targetEntityType,ReagentId.GetType current.entity)
                (CellEntity.removeCompartment,CellCompartment.toString current.compartment);
                (CellEntity.addCompartment,CellCompartment.toString modified.compartment)
            ]
        else    
            []
    match changemap with 
    | [] -> None 
    | _ -> 
        let change = serializeEventChange changemap
        let target = EventTarget.CellEntityEvent(current.cellId)
        (Event.Create message (EventOperation.Modify) target change) |> Some

let private removeEntity (message:string) (cellEntity:CellEntity) = 
    let changemap = [
        (CellEntity.removeCompartment,CellCompartment.toString cellEntity.compartment)
        (CellEntity.removeEntity,cellEntity.entity.ToString())
        (CellEntity.removeEntityType,ReagentId.GetType cellEntity.entity)
    ]
    let change = serializeEventChange changemap 
    let target = EventTarget.CellEntityEvent(cellEntity.cellId)
    Event.Create message (EventOperation.Modify) target change

let addCellEntity (message:string) (existing:CellEntity list) (current:CellEntity) = 
    let entityExists = existing |> List.tryFind (fun e -> (e.entity = current.entity) && (e.compartment = current.compartment))
    match entityExists with
    | Some(_) -> None
    | None -> (addEntity message current) |> Some

let removeCellEntity (message:string) (existing:CellEntity list) (current:CellEntity) =
    let entityExists = existing |> List.tryFind (fun e -> (e.entity = current.entity) && (e.compartment = current.compartment))
    match entityExists with
    | Some(existingEntity) -> (removeEntity message existingEntity) |> Some 
    | None -> None

//Cell File Events
let addCellFile (message:string) (cellId:CellId) (existing:FileRef list) (file:FileRef) = 
    let fileExists = existing |> List.tryFind (fun f -> f.fileId = file.fileId)
    match fileExists with 
    | Some (existingfile) -> modifyFileRef message (CellFileEvent(cellId)) existingfile file
    | None -> (addFileRef message (CellFileEvent(cellId)) file) |> Some

let unlinkCellFile (message:string) (cellId:CellId) (file:FileRef) = 
    (removeFileRef message (CellFileEvent(cellId)) file)

//Cell Tag Events
let addCellTag (message:string) (cellId:CellId) (existingtags:Tag list) (tag:Tag) =     
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) -> None 
    | None -> 
        let changemap = [
            (Cell.addTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.CellTagEvent(cellId)
        (Event.Create message (EventOperation.Add) target change) |> Some
   
let removeCellTag (message:string) (cellId:CellId) (existingtags:Tag list) (tag:Tag) = 
    let tagexists = existingtags |> List.tryFind (fun x -> x = tag)
    match tagexists with 
    | Some (_) ->  
        let changemap = [
            (Cell.removeTag,tag.ToString())
        ]
        let change = serializeEventChange changemap
        let target = EventTarget.CellTagEvent(cellId)
        (Event.Create message (EventOperation.Modify) target change) |> Some    
    | None -> None
 
let addInteraction (message:string) (interaction:Interaction) = 
    let changemap = [
        (Interaction.addNotes,interaction.notes)
        (Interaction.addType,interaction.getType)
        (Interaction.addDeprecate,interaction.deprecated.ToString())
    ]
    let changemap = 
        match interaction with 
        | CodesFor (i) -> 
            let prop = [
                    (CodesForInteraction.addCDS,i.cds.ToString())
                    (CodesForInteraction.addProtein,i.protein.ToString())
                ]
            prop@changemap
        | GeneticActivation(i) -> 
            let activator = 
                let rlist = i.activator //|> List.map (fun r -> r.ToString())
                Encode.Auto.toString(0,rlist)
            let prop = [
                (GeneticActivationInteraction.addActivated,i.activated.ToString())
                (GeneticActivationInteraction.addActivator,activator)
            ]
            prop@changemap
        | GeneticInhibition(i) -> 
            let inhibitor = 
                let rlist = i.inhibitor //|> List.map (fun r -> r.ToString())
                Encode.Auto.toString(0,rlist)
            let prop = [
                (GeneticInhibitionInteraction.addInhibited,i.inhibited.ToString())
                (GeneticInhibitionInteraction.addInhibitor,inhibitor)
            ]
            prop@changemap
        | Reaction(i) -> 
            let reactants = 
                let rlist = i.reactants //|> List.map (fun rs -> rs |> List.map (fun r -> r.ToString())) 
                Encode.Auto.toString(0,rlist)
            let products = 
                let rlist = i.products //|> List.map (fun rs -> rs |> List.map (fun r -> r.ToString()))
                Encode.Auto.toString(0,rlist)
            let enzyme = 
                let rlist = i.enzyme //|> List.map (fun r -> r.ToString())
                Encode.Auto.toString(0,rlist)
            let prop = [
                (ReactionInteraction.addReactants, reactants)
                (ReactionInteraction.addEnzyme, enzyme)
                (ReactionInteraction.addProducts, products)
                
            ]
            prop@changemap
        | GenericActivation(i) -> 
            let regulated = 
                Encode.Auto.toString(0,i.regulated)
            let regulator = 
                let rlist = i.regulator
                Encode.Auto.toString(0,rlist)
            let prop = [
                (GenericInteraction.addRegulated,regulated)
                (GenericInteraction.addRegulator,regulator)
            ]
            prop@changemap
        | GenericInhibition(i) -> 
            let regulated = 
                Encode.Auto.toString(0,i.regulated)
            let regulator = 
                let rlist = i.regulator
                Encode.Auto.toString(0,rlist)
            let prop = [
                (GenericInteraction.addRegulated,regulated)
                (GenericInteraction.addRegulator,regulator)
            ]
            prop@changemap
    let change = serializeEventChange changemap 

    let target = EventTarget.InteractionEvent(interaction.id)
    (Event.Create message (EventOperation.Add) target change) 

let deprecateInteraction (message:string) (current:Interaction) (modified:Interaction)= 
    let changemap = if (current.deprecated <> modified.deprecated) then [(Interaction.removeDeprecate,current.deprecated.ToString());(Interaction.addDeprecate,modified.deprecated.ToString())] else []
    let change = serializeEventChange changemap 
    let target = EventTarget.InteractionEvent(current.id)
    Event.Create message (EventOperation.Modify) target change

let addDerivedFrom (message:string) (derivedFrom:DerivedFrom) = 
    let changemap = [
        (DerivedFrom.addSource,(DerivedFrom.GetSourceGuid derivedFrom).ToString())
        (DerivedFrom.addTarget,(DerivedFrom.GetTargetGuid derivedFrom).ToString())
        (DerivedFrom.addType,(DerivedFrom.GetType derivedFrom))
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.DerivedFromEvent
    (Event.Create message (EventOperation.Add) target change) 
     
let removeDerivedFrom (message:string) (derivedFrom:DerivedFrom) = 
    let changemap = [
        (DerivedFrom.removeSource,(DerivedFrom.GetSourceGuid derivedFrom).ToString())
        (DerivedFrom.removeTarget,(DerivedFrom.GetTargetGuid derivedFrom).ToString())
        (DerivedFrom.removeType,(DerivedFrom.GetType derivedFrom))
    ]
    let change = serializeEventChange changemap
    let target = EventTarget.DerivedFromEvent
    (Event.Create message (EventOperation.Modify) target change) 


//File Events
let uploadFile (message:string) (fileId:FileId) (filepath:string) = 
    let changemap = [(FileId.addFile,filepath)]
    let change = serializeEventChange changemap
    let target = EventTarget.FileEvent(fileId)
    Event.Create message (EventOperation.Add) target change

let replaceFile (message:string) (fileId:FileId) (filepath:string)  = 
    let changemap = [(FileId.addFile,filepath)]
    let change = serializeEventChange changemap
    let target = EventTarget.FileEvent(fileId)
    Event.Create message (EventOperation.Modify) target change

let uploadTimeSeriesFile (message:string) (fileId:FileId) (filepath:string) = 
    let changemap = [(FileId.addFile,filepath)]
    let change = serializeEventChange changemap
    let target = EventTarget.TimeSeriesFileEvent(fileId)
    Event.Create message (EventOperation.Add) target change

let replaceTimeSeriesFile (message:string) (fileId:FileId) (filepath:string) = 
    let changemap = [(FileId.addFile,filepath)]
    let change = serializeEventChange changemap
    let target = EventTarget.TimeSeriesFileEvent(fileId)
    Event.Create message (EventOperation.Modify) target change

let uploadBundle (message:string) (fileId:FileId) (files:string list)  = 
    
    let filearray = Encode.Auto.toString(0,files)
    let changemap =  [(FileId.addFiles,filearray)]
    let change = serializeEventChange changemap
    let target = EventTarget.BundleFileEvent(fileId)
    Event.Create message (EventOperation.Add) target change


//Log Events
let startReplayLog (message:string) (logname:string) =
    Event.Create message (EventOperation.Add) (EventTarget.StartLogEvent(logname)) ""

let endReplayLog (message:string) (logname:string) =
    Event.Create message (EventOperation.Add) (EventTarget.FinishLogEvent(logname)) ""

//Process Data Event
let processDataEvent (message:string) (exptId:ExperimentId) = 
    Event.Create message (EventOperation.Add) (EventTarget.ProcessDataEvent(exptId)) ""

let parseLayoutEvent (message:string) (exptId:ExperimentId) = 
    Event.Create message (EventOperation.Add) (EventTarget.ParseLayoutEvent(exptId)) ""



let addObservation (message:string) (current:Observation) = 
    let changemap = 
        [ (Observation.addSampleId, current.sampleId.ToString())
          (Observation.addSignalId, current.signalId.ToString())
          (Observation.addValue, current.value.ToString())
        ]
        |> List.append (match current.timestamp with Some t -> [Observation.addTimestamp, t.ToString()] | None -> [])
        |> List.append (match current.replicate with Some r -> [Observation.addReplicate, r.ToString()] | None -> [])
        
    let change = serializeEventChange changemap
    let target = EventTarget.ObservationEvent current.id
    let sampleevent = Event.Create message (EventOperation.Add) target change

    sampleevent

type EventResult = 
    | PartEventResult of Async<bool>
    | PartTagEventResult of Async<bool>
    
    | ReagentEventResult of Async<bool>
    | ReagentFileEventResult of Async<bool>
    | ReagentTagEventResult of Async<bool> 
    
    | ExperimentEventResult of Async<bool>
    | ExperimentFileEventResult of Async<bool>
    | ExperimentEventEventResult of Async<bool>
    | ExperimentSignalEventResult of Async<bool>
    | ExperimentTagEventResult of Async<bool> 
    
    | SampleEventResult of Async<bool>
    | SampleDeviceEventResult of Async<bool>
    | SampleDataEventResult of Async<bool>
    | SampleConditionEventResult of Async<bool>
    | SampleTagEventResult of Async<bool>
    | SampleReplicateEventResult of Async<bool>

    | CellEventResult of Async<bool> 
    | CellFileEventResult of Async<bool> 
    | CellTagEventResult of Async<bool>
    | CellEntityEventResult of Async<bool>

    | InteractionEventResult of Async<bool> 
    
    | DerivedFromEventResult of Async<bool> 
    
    | FileEventResult of Async<unit>
    | TimeSeriesFileEventResult of Async<unit>
    | BundleFileEventResult of Async<unit>
    | EmptyEventResult
    | ObservationEventResult of Async<bool> 
    //| StartLogEventResult of Async<bool>
    //| FinishLogEventResult of Async<bool>
    static member ExecuteAsyncAndIgnore = 
        function
        | PartEventResult              b -> b |> Async.RunSynchronously |> ignore
        | PartTagEventResult           b -> b |> Async.RunSynchronously |> ignore
        
        | ReagentEventResult           b -> b |> Async.RunSynchronously |> ignore
        | ReagentFileEventResult       b -> b |> Async.RunSynchronously |> ignore
        | ReagentTagEventResult        b -> b |> Async.RunSynchronously |> ignore
        
        | ExperimentEventResult        b -> b |> Async.RunSynchronously |> ignore
        | ExperimentFileEventResult    b -> b |> Async.RunSynchronously |> ignore
        | ExperimentEventEventResult   b -> b |> Async.RunSynchronously |> ignore
        | ExperimentSignalEventResult  b -> b |> Async.RunSynchronously |> ignore
        | ExperimentTagEventResult     b -> b |> Async.RunSynchronously |> ignore
        
        | CellEventResult              b -> b |> Async.RunSynchronously |> ignore
        | CellFileEventResult          b -> b |> Async.RunSynchronously |> ignore
        | CellTagEventResult           b -> b |> Async.RunSynchronously |> ignore
        | CellEntityEventResult        b -> b |> Async.RunSynchronously |> ignore
        
        | InteractionEventResult       b -> b |> Async.RunSynchronously |> ignore

        | DerivedFromEventResult       b -> b |> Async.RunSynchronously |> ignore

        | FileEventResult              b -> b |> Async.RunSynchronously |> ignore
        | TimeSeriesFileEventResult    b -> b |> Async.RunSynchronously |> ignore
        | BundleFileEventResult        b -> b |> Async.RunSynchronously |> ignore
        
        | SampleEventResult            b -> b |> Async.RunSynchronously |> ignore
        | SampleDeviceEventResult      b -> b |> Async.RunSynchronously |> ignore
        | SampleDataEventResult        b -> b |> Async.RunSynchronously |> ignore
        | SampleConditionEventResult   b -> b |> Async.RunSynchronously |> ignore
        | SampleTagEventResult         b -> b |> Async.RunSynchronously |> ignore
        | SampleReplicateEventResult   b -> b |> Async.RunSynchronously |> ignore

        | ObservationEventResult       b -> b |> Async.RunSynchronously |> ignore
       
        | EmptyEventResult               -> ()
        
        //| StartLogEventResult          b -> b |> Async.RunSynchronously |> ignore
        //| FinishLogEventResult         b -> b |> Async.RunSynchronously |> ignore



module EventsProcessor =
    let ProcessPartEvent savePart (tryGetPart:PartId -> Async<Part option>) (partId:PartId) (operation:EventOperation) (changestring:string) =     
        
        match operation with
        | EventOperation.Add -> 
            let change = deserializeEventChange changestring
            let changemap = change |> Map.ofList
            
            let partProperties = {
                PartProperties.name = changemap.[Part.addName]
                PartProperties.sequence = changemap.[Part.addSequence]
                PartProperties.deprecated = System.Boolean.Parse(changemap.[Part.addDeprecate])
            }
            let part = Part.FromStringType (partId.ToString() |> System.Guid) partProperties (changemap.[Part.addType])             
            savePart part |> PartEventResult
        
        | EventOperation.Modify -> 
        
            let change = deserializeEventChange changestring
            let changemap = change |> Map.ofList
        
            let partoption = tryGetPart partId |> Async.RunSynchronously
            let part = 
                match partoption with 
                | Some p -> p 
                | None -> failwithf "%s not found in BCKG." (partId.ToString())
            
            let typeString =
                if changemap.ContainsKey(Part.addType) then
                    if not (changemap.ContainsKey(Part.removeType)) then printfn "[WARNING]: Part modify event has a ++type but not a --type."    
                    changemap.[Part.addType] 
                else 
                    part.getType

            let partProperties = part.getProperties
            let partProperties = 
                if changemap.ContainsKey(Part.addName) && changemap.ContainsKey(Part.removeName) then 
                    if part.name <> changemap.[Part.removeName] then failwith "Part name doesn't match." else ()
                    {partProperties with name = changemap.[Part.addName]}  
                else partProperties 
            
            let partProperties = 
                if changemap.ContainsKey(Part.addSequence) && changemap.ContainsKey(Part.removeSequence) then 
                    if partProperties.sequence <> changemap.[Part.removeSequence] then failwith "Part sequence doesn't match." else ()
                    {partProperties with sequence = changemap.[Part.addSequence]}  
                else partProperties 

            let partProperties = 
                if changemap.ContainsKey(Part.addDeprecate) && changemap.ContainsKey(Part.removeDeprecate) then 
                    {partProperties with deprecated = changemap.[Part.addDeprecate] |> System.Boolean.Parse}
                else partProperties
        
            let modifiedPart = Part.FromStringType (partId.ToString() |> System.Guid) partProperties typeString 
            savePart modifiedPart |> PartEventResult
    
    let ProcessReagentEvent saveReagent (tryGetReagent:ReagentId->Async<Reagent option>) (reagentId:ReagentId) (operation:EventOperation) (changestring:string) = 
        
        
        
        let updateReagent (reagentProps:ReagentProperties) (reagentType:string) (changemap:Map<string,string>)= 
            match reagentType with 
            | "Chemical" -> Chemical({id = reagentId.guid |> ChemicalId; Type=ChemicalType.fromString changemap.[Reagent.addChemicalType];properties = reagentProps})
            | "DNA" -> 
                let concentration = if changemap.ContainsKey(Reagent.addConcentration) then (changemap.[Reagent.addConcentration] |>  Concentration.Parse |> Some) else None
                let sequence = changemap.[Reagent.addSequence]
                let dnaType = DNAType.fromString (changemap.[Reagent.addDNAType])
                DNA({id = reagentId.guid |> DNAId; Type=dnaType;sequence=sequence;concentration = concentration;properties = reagentProps})
            | "RNA" -> 
                let rnaType = RNAType.fromString (changemap.[Reagent.addRNAType])
                let sequence = changemap.[Reagent.addSequence]
                RNA({id = reagentId.guid |> RNAId; Type=rnaType;sequence=sequence;properties = reagentProps})
            | "Protein" -> 
                let isReporter = System.Boolean.Parse changemap.[Reagent.addIsReporter]
                Protein({id = reagentId.guid |> ProteinId; isReporter = isReporter;properties = reagentProps})
            | "Generic Entity" -> GenericEntity({id = reagentId.guid |> GenericEntityId; properties = reagentProps})
            | _ -> failwithf "%s is an unknown type of reagent." (changemap.[Reagent.addType])
        
        match operation with 
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let reagentProps = {
                ReagentProperties.name = changemap.[Reagent.addName]
                ReagentProperties.notes = changemap.[Reagent.addNotes]
                ReagentProperties.barcode = if changemap.ContainsKey(Reagent.addBarcode) then (changemap.[Reagent.addBarcode] |> Barcode |> Some) else None
                ReagentProperties.deprecated = changemap.[Reagent.addDeprecate] |> System.Boolean.Parse
            }
            
            let reagent = updateReagent reagentProps changemap.[Reagent.addType] changemap 
                
            saveReagent reagent |> ReagentEventResult
        
        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let reagentoption = tryGetReagent reagentId |> Async.RunSynchronously
            let reagent = match reagentoption with | Some (p) -> p | None -> failwithf "%s not found in BCKG." (reagentId.ToString())
            
            let reagentProperties = reagent.getProperties
            let reagentProperties =  if changemap.ContainsKey(Reagent.addName) then {reagentProperties with name = changemap.[Reagent.addName]} else reagentProperties
            let reagentProperties =  if changemap.ContainsKey(Reagent.addNotes) then {reagentProperties with notes = changemap.[Reagent.addNotes]} else reagentProperties
            let reagentProperties =  if changemap.ContainsKey(Reagent.removeBarcode) then {reagentProperties with barcode = None} else reagentProperties
            let reagentProperties =  if changemap.ContainsKey(Reagent.addBarcode) then {reagentProperties with barcode = changemap.[Reagent.addBarcode] |> Barcode |> Some} else reagentProperties
            let reagentProperties =  if changemap.ContainsKey(Reagent.addDeprecate) then {reagentProperties with deprecated = changemap.[Reagent.addDeprecate] |> System.Boolean.Parse} else reagentProperties

            let reagent = 
                if changemap.ContainsKey(Reagent.addType) then
                    if not (changemap.ContainsKey(Reagent.removeType)) then printfn "[WARNING]: Reagent modify event has a ++type but not a --type."    
                    updateReagent reagentProperties changemap.[Reagent.addType] changemap 
                else 
                    match reagent with 
                    | Chemical(chem) -> 
                        let chemtype = if changemap.ContainsKey(Reagent.addChemicalType) then ChemicalType.fromString changemap.[Reagent.addChemicalType] else chem.Type
                        Chemical({id = reagentId.guid |> ChemicalId;Type=chemtype;properties = reagentProperties})
                    | DNA(dna) -> 
                        let concentration = if changemap.ContainsKey(Reagent.addConcentration) then (changemap.[Reagent.addConcentration] |>  Concentration.Parse |> Some) else None
                        let sequence = if changemap.ContainsKey(Reagent.addSequence) then changemap.[Reagent.addSequence] else dna.sequence
                        let dnaType = if changemap.ContainsKey(Reagent.addDNAType) then DNAType.fromString (changemap.[Reagent.addDNAType]) else dna.Type
                        DNA({id = reagentId.guid |> DNAId;concentration = concentration;sequence=sequence;Type=dnaType;properties=reagentProperties})
                    | RNA(rna) -> 
                        let sequence = if changemap.ContainsKey(Reagent.addSequence) then changemap.[Reagent.addSequence] else rna.sequence
                        let rnaType = if changemap.ContainsKey(Reagent.addRNAType) then RNAType.fromString (changemap.[Reagent.addRNAType]) else rna.Type
                        RNA({id = reagentId.guid |> RNAId;sequence = sequence;Type = rnaType; properties = reagentProperties})
                    | Protein(prot) -> 
                        let isReporter = if changemap.ContainsKey(Reagent.addIsReporter) then System.Boolean.Parse(changemap.[Reagent.addIsReporter]) else prot.isReporter
                        Protein({id = reagentId.guid |> ProteinId;isReporter = isReporter; properties = reagentProperties})
                    | GenericEntity(ge) -> GenericEntity({id = reagentId.guid |> GenericEntityId;properties = reagentProperties})
            
            saveReagent reagent |> ReagentEventResult
    
    

    let ProcessUploadFileEvent uploadFile (fileId:FileId) (operation:EventOperation) (filecontents:byte []) =
        match operation with 
        | EventOperation.Add -> uploadFile(fileId,filecontents) |> FileEventResult    
        | EventOperation.Modify -> uploadFile(fileId,filecontents)|> FileEventResult
    
    let ProcessTimeSeriesFileEvent uploadTimeSeries (fileId:FileId) (operation:EventOperation) (filecontents:byte []) =
        match operation with 
        | EventOperation.Add -> uploadTimeSeries (fileId, filecontents) |> TimeSeriesFileEventResult    
        | EventOperation.Modify -> uploadTimeSeries (fileId, filecontents) |> TimeSeriesFileEventResult
    
    let ProcessBundleEvent uploadFile (fileId:FileId) (operation:EventOperation) (bundle:byte []) = 
        match operation with 
        | EventOperation.Add       -> uploadFile (fileId, bundle) |> BundleFileEventResult
        | EventOperation.Modify    -> failwith "File Bundle modification not supported yet."
    
    let ProcessSampleEvent saveSample tryGetSample (sampleId:SampleId) (operation:EventOperation) (changestring:string) = 
           match operation with 
           | EventOperation.Add -> 
               let changemap = deserializeEventChange changestring  |> Map.ofList
               let sample = {
                   Sample.id = sampleId
                   Sample.experimentId = changemap.["++experimentId"] |> System.Guid |> ExperimentId
                   Sample.meta = 
                       match SampleMeta.fromStringType(changemap.["++metaType"]) with 
                       | MissingMeta -> MissingMeta
                       | PlateReaderMeta _ -> 
                           let virtualWell = {row = int changemap.["++virtualWellRow"] ; col = int changemap.["++virtualWellCol"]}
                           let physicalWell = 
                               if changemap.ContainsKey("++physicalWellRow") && changemap.ContainsKey("++physicalWellCol") then 
                                   Some({row = int changemap.["++physicalWellRow"]; col = int changemap.["++physicalWellCol"]})
                               else None
                           let physicalPlateName = 
                               if changemap.ContainsKey("++physicalPlateName") then 
                                   Some(changemap.["++physicalPlateName"])
                               else None 
                           PlateReaderMeta({virtualWell = virtualWell;physicalWell = physicalWell; physicalPlateName = physicalPlateName })  
                   
                   Sample.deprecated = changemap.[Sample.addDeprecate] |> System.Boolean.Parse
               }
               saveSample sample |> SampleEventResult
           
           | EventOperation.Modify -> 
               let sampleoption = tryGetSample sampleId |> Async.RunSynchronously
               let (sample:Sample) = match sampleoption with | Some (s) -> s | None -> failwithf "%s not found in BCKG." (sampleId.ToString())
               let changemap = deserializeEventChange changestring  |> Map.ofList
               
               let sample =  if changemap.ContainsKey(Sample.addDeprecate) then {sample with deprecated = changemap.[Sample.addDeprecate] |> System.Boolean.Parse} else sample
               
               let sample = 
                   if changemap.ContainsKey("++experimentId") then 
                       {sample with experimentId = changemap.["++experimentId"] |> System.Guid |> ExperimentId}
                   else sample
               
               let meta = 
                   match sample.meta with 
                   | PlateReaderMeta(prmd) -> 
                       let vwellrowadd = changemap.ContainsKey("++virtualWellRow")
                       let vwellcoladd = changemap.ContainsKey("++virtualWellCol")
                       let vwellrowrem = changemap.ContainsKey("--virtualWellRow")
                       let vwellcolrem = changemap.ContainsKey("--virtualWellCol")
                       
                       let pwellrowadd = changemap.ContainsKey("++physicalWellRow")
                       let pwellcoladd = changemap.ContainsKey("++physicalWellCol")
                       let pwellrowrem = changemap.ContainsKey("--physicalWellRow")
                       let pwellcolrem = changemap.ContainsKey("--physicalWellCol")
                       
                       let platenameadd = changemap.ContainsKey("++physicalPlateName")
                       let platenamerem = changemap.ContainsKey("--physicalPlateName")
                       
                       let physicalPlateName = 
                           match prmd.physicalPlateName with 
                           | Some(ppm) -> 
                               match (platenamerem,platenameadd) with
                               | (false,false) -> ppm |> Some
                               | (false,true)  -> 
                                   printfn "WARNING: The existing Sample %s has a Physical Plate Name for Plate Reader Meta Data. Yet, change does not have a %s." (sampleId.ToString()) ("--physicalPlateName")
                                   changemap.["++physicalPlateName"] |> Some
                               | (true,false) -> None
                               | (true,true) -> changemap.["++physicalPlateName"] |> Some
                           | None -> 
                               if platenamerem then printfn "WARNING: The existing Sample %s does not have a Physical Plate Name for Plate Reader Meta Data. Yet, change has a %s." (sampleId.ToString()) ("--physicalPlateName")
                               if platenameadd then changemap.["++physicalPlateName"] |> Some else None
                       
       
                       let physicalWell =  
                           match prmd.physicalWell with 
                           | Some(pw) -> 
                               if (pwellrowrem && pwellcolrem && (not pwellrowadd) && (not pwellcoladd)) then None
                               else 
                                   let row = 
                                       match (pwellrowrem,pwellrowadd) with 
                                       | (false,false) -> pw.row
                                       | (true,true) -> int changemap.["++physicalWellRow"] 
                                       | _ -> 
                                           printfn "WARNING: The existing Sample %s already has a Physical Well Row. To modify it, make sure you have both remove and add in the change log." (sampleId.ToString())
                                           pw.row
                                   let col  = 
                                       match (pwellcolrem,pwellcoladd) with 
                                       | (false,false) -> pw.col
                                       | (true,true) -> int changemap.["++physicalWellCol"]
                                       | _ -> 
                                           printfn "WARNING: The existing Sample %s already has a Physical Well Col. To modify it, make sure you have both remove and add in the change log." (sampleId.ToString())
                                           pw.col
                                   {row = row; col = col} |> Some
       
       
                           | None -> 
                               if (pwellrowrem || pwellcolrem) then printfn "WARNING: The existing Sample %s does not have a Physical well. Yet the change has a %s or %s or both."  (sampleId.ToString()) ("--physicalWellRow") ("--physicalWellCol")
                               match (pwellrowadd,pwellcoladd) with 
                               | (false,false) -> None
                               | (true,true) -> {row = int changemap.["++physicalWellRow"]; col = int changemap.["++physicalWellCol"]} |> Some
                               | _ -> 
                                   printfn "WARNING: The existing Sample %s has None for Physical Well. Please make sure you add both row and col properties." (sampleId.ToString())
                                   None
                       
                       let virtualWellOpt = 
                           if (vwellrowrem && vwellcolrem && (not vwellrowadd) && (not vwellcoladd)) then
                               match (physicalPlateName,physicalWell) with 
                               | (None,None) -> None 
                               | _ -> 
                                   printfn "WARNING: The existing Sample %s needs to have a Virtual Well property." (sampleId.ToString())
                                   Some(prmd.virtualWell)
                           else
                               let row = 
                                   match (vwellrowrem,vwellrowadd) with 
                                   | (false,false) -> prmd.virtualWell.row
                                   | (true,true) -> int changemap.["++virtualWellRow"] 
                                   | _ -> 
                                       printfn "WARNING: The existing Sample %s already has a Physical Well Row. To modify it, make sure you have both remove and add in the change log." (sampleId.ToString())
                                       prmd.virtualWell.row
                               let col  = 
                                   match (vwellcolrem,vwellcoladd) with 
                                   | (false,false) -> prmd.virtualWell.col
                                   | (true,true) -> int changemap.["++virtualWellCol"]
                                   | _ -> 
                                       printfn "WARNING: The existing Sample %s already has a Physical Well Col. To modify it, make sure you have both remove and add in the change log." (sampleId.ToString())
                                       prmd.virtualWell.col
                               {row = row;col = col} |> Some
                       match virtualWellOpt with 
                       | Some(virtualWell) -> 
                           {virtualWell = virtualWell;physicalPlateName = physicalPlateName;physicalWell = physicalWell} |> PlateReaderMeta
                       | None -> MissingMeta                
                       
                   | MissingMeta -> 
                       if changemap.ContainsKey("++virtualWellRow") && changemap.ContainsKey("++virtualWellCol") then 
                           let virtualWell = {row = int changemap.["++virtualWellRow"]; col = int changemap.["++virtualWellCol"]}                    
                           let physicalPlateName = 
                               if changemap.ContainsKey("++physicalPlateName") then 
                                   Some(changemap.["++physicalPlateName"]) 
                               else None
                           let physicalWell = 
                               if changemap.ContainsKey("++physicalWellRow") && changemap.ContainsKey("++physicalWellCol") then 
                                   Some({row = int changemap.["++physicalWellRow"]; col = int changemap.["++physicalWellCol"]})
                               else None 
                           PlateReaderMeta({virtualWell = virtualWell;physicalWell = physicalWell;physicalPlateName = physicalPlateName})
                       else    
                           MissingMeta
                           
               let sample = {sample with meta = meta}            
               
               saveSample sample |> SampleEventResult
               
    let ProcessSampleConditionEvent saveCondition deleteCondition tryGetSampleCondition (sampleId:SampleId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add ->     
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let reagentId = ReagentId.FromType (changemap.[Condition.addReagentId] |> System.Guid) (changemap.[Condition.addReagentEntity]) 
            let value = Concentration.Parse changemap.[Condition.addValue]
            let time = if changemap.ContainsKey Condition.addTime then (Time.Create (changemap.[Condition.addTime] |> System.Double.Parse) changemap.[Condition.addTimeUnits])  |> Some else None
            saveCondition (sampleId, reagentId, value, time) |> SampleConditionEventResult
        
        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            match changemap.ContainsKey(Condition.removeReagentId) with 
            | true -> 
                let reagentId = ReagentId.FromType (changemap.[Condition.removeReagentId] |> System.Guid) (changemap.[Condition.removeReagentEntity]) 
                let (conditionstoreoption:Condition option) = tryGetSampleCondition (sampleId, reagentId) |> Async.RunSynchronously
                match conditionstoreoption with 
                | Some _ -> deleteCondition (sampleId, reagentId) |> SampleConditionEventResult 
                | None -> failwithf "No device with ReagentId %s found with sample id %s." (reagentId.ToString()) (sampleId.ToString())
                         
            
            | false ->
                let reagentId = ReagentId.FromType (changemap.[Condition.targetReagentId] |> System.Guid) (changemap.[Condition.targetReagentEntity]) 
                
                let sampleconditionstoreoption = tryGetSampleCondition (sampleId, reagentId) |> Async.RunSynchronously
                let samplecondition : Condition = 
                    match sampleconditionstoreoption with 
                    | Some v -> v 
                    | None -> failwithf "No device with ReagentId %s found with sample id %s." (reagentId.ToString()) (sampleId.ToString())
                               
                let value = 
                    if changemap.ContainsKey(Condition.removeValue) && changemap.ContainsKey(Condition.addValue) then 
                        Concentration.Parse changemap.[Condition.addValue]
                    else samplecondition.concentration
                let time = 
                    match samplecondition.time with 
                    | Some(t) -> 
                        match (changemap.ContainsKey(Condition.addTime),changemap.ContainsKey(Condition.addTimeUnits),changemap.ContainsKey(Condition.removeTime),changemap.ContainsKey(Condition.removeTimeUnits)) with
                        | (false,false,false,false) -> t |> Some
                        | (false,false,true,true)   -> None
                        | (true,true,true,true) -> Time.Create (changemap.[Condition.addTime] |> System.Double.Parse) (changemap.[Condition.addTimeUnits]) |> Some
                        | _ -> failwithf "Malformed Event. Time modification event in condition of Sample %s with reagent %s is not formatted correctly" (sampleId.ToString()) (reagentId.ToString())
                        
                    | None -> 
                        //Any combination other than adding both time and time units is malformed 
                        if changemap.ContainsKey(Condition.addTime) || changemap.ContainsKey(Condition.addTimeUnits) then 
                            Time.Create (changemap.[Condition.addTime] |> System.Double.Parse) (changemap.[Condition.addTimeUnits]) 
                            |> Some
                        else
                            failwithf "Malformed Event. Existing time was none. Hence time value cannot be removed."

                saveCondition (sampleId, reagentId, value, time) |> SampleConditionEventResult            
            
    let ProcessSampleDeviceEvent saveSampleDevice deleteSampleDeviceStore tryGetSampleDeviceStore (sampleId:SampleId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add ->
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let cellId = changemap.[Sample.addCellId] |> System.Guid |> CellId
            let cellDensity = if changemap.ContainsKey "++cellDensity" then Some (float changemap.["++cellDensity"]) else None
            let cellPreSeeding = if changemap.ContainsKey "++cellPreSeeding" then Some (float changemap.["++cellPreSeeding"]) else None
            saveSampleDevice (sampleId, cellId, cellDensity, cellPreSeeding) |> SampleDeviceEventResult

        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            match changemap.ContainsKey(Sample.addCellId) with 
            | true ->
                printfn "[WARNING] Should never reach this state."
                let oldCellId = changemap.[Sample.removeCellId] |> System.Guid |> CellId
                let newcellId = changemap.[Sample.addCellId] |> System.Guid |> CellId
                
                let sampledevicestoreoption = tryGetSampleDeviceStore (sampleId, oldCellId) |> Async.RunSynchronously      
                let cellDensity = if changemap.ContainsKey "++cellDensity" then Some (float changemap.["++cellDensity"]) else None
                let cellPreSeeding = if changemap.ContainsKey "++cellPreSeeding" then Some (float changemap.["++cellPreSeeding"]) else None
                match sampledevicestoreoption with 
                | Some _ -> saveSampleDevice (sampleId, newcellId, cellDensity, cellPreSeeding) |> SampleDeviceEventResult            
                | None -> failwithf "No device with CellId %s found with sample id %s." (oldCellId.ToString()) (sampleId.ToString())
            | false -> 
                let cellId = changemap.[Sample.removeCellId] |> System.Guid |> CellId
                let sampledevicestoreoption = tryGetSampleDeviceStore (sampleId, cellId) |> Async.RunSynchronously
                let sampledevicestore = 
                    match sampledevicestoreoption with 
                    | Some v -> v 
                    | None -> failwithf "No device with CellId %s found with sample id %s." (cellId.ToString()) (sampleId.ToString())
                deleteSampleDeviceStore sampledevicestore |> SampleDeviceEventResult          

    let ProcessSampleReplicateEvent saveSampleReplicate unlinkSampleReplicate (sampleId:SampleId) (operation:EventOperation) (changestring:string) = 
        match operation with
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let replicate = changemap.[Sample.addReplicate] |> System.Guid |> ReplicateId
            saveSampleReplicate (sampleId,replicate) |> SampleReplicateEventResult
        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let replicate = changemap.[Sample.removeReplicate] |> System.Guid |> ReplicateId
            unlinkSampleReplicate (replicate) |> SampleReplicateEventResult

    let ProcessExperimentSignalEvent saveExperimentSignal deleteExperimentSignal tryGetExperimentSignal (experimentId:ExperimentId) (operation:EventOperation) (changestring:string) =         
        match operation with 
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let signalId = changemap.["++signalId"] |> System.Guid |> SignalId 
            let units = if changemap.ContainsKey "++units" then Some changemap.["++units"] else None
            let settings = 
                match changemap.["type"] with 
                | str when str = SignalSettings.toTypeString(PlateReaderTemperature) -> PlateReaderTemperature  
                | str when str = SignalSettings.toTypeString(PlateReaderLuminescence) -> PlateReaderLuminescence  
                | str when str = SignalSettings.toTypeString(PlateReaderFluorescence(PlateReaderFluorescenceSettings.Create(PlateFilter_520,PlateFilter_520,0.0))) -> 
                    PlateReaderFluorescence(PlateReaderFluorescenceSettings.Create(PlateReaderFilter.fromString (changemap.["++excitation"]),PlateReaderFilter.fromString (changemap.["++emission"]), float changemap.["++gain"]))
                | str when str = SignalSettings.toTypeString(PlateReaderAbsorbance(PlateReaderAbsorbanceSettings.Create(0.0,0.0,0.0))) -> 
                    PlateReaderAbsorbance(PlateReaderAbsorbanceSettings.Create(float changemap.["++wavelength"], float changemap.["++gain"], float changemap.["++correction"]))
                | str when str = SignalSettings.toTypeString(Titre) -> Titre
                | _ -> 
                    //TODO: check
                    SignalSettings.fromTypeString changemap.["type"]
                    //failwithf "%s is not a supported Experiment Signal Type" (changemap.["type"])
            let signal = {Signal.id = signalId;Signal.settings = settings; units = units}
            saveExperimentSignal (experimentId, signal) |> ExperimentSignalEventResult        
                
        | EventOperation.Modify -> 
            
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let signalId = changemap.["--signalId"] |> System.Guid |> SignalId 
            
            let signaloption = tryGetExperimentSignal (experimentId, signalId) |> Async.RunSynchronously
            let signal = match signaloption with | Some(s) -> s | None -> failwithf "%s not found in BCKG." (signalId.ToString())
            deleteExperimentSignal (experimentId, signal) |> ExperimentSignalEventResult
        
    let ProcessExperimentOperationEvent saveExperimentEvent deleteExperimentEvent tryGetExperimentEvent (experimentId:ExperimentId) (operation:EventOperation) (changestring:string) =         
        match operation with 
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let event = {
                ExperimentOperation.id = changemap.[ExperimentOperation.addExperimentOperationId] |> System.Guid |> ExperimentOperationId
                ExperimentOperation.timestamp  = System.DateTime.Parse(changemap.["++timestamp"])
                ExperimentOperation.Type = changemap.["++type"] |> ExperimentOperationType.fromString
            }
            saveExperimentEvent (experimentId, event) |> ExperimentEventEventResult
    
        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            
            match changemap.ContainsKey(ExperimentOperation.removeExperimentOperationId) with 
            | true ->
                let experimentEventGuid = changemap.[ExperimentOperation.removeExperimentOperationId] |> ExperimentOperationId.fromString
                let experimenteventoption = 
                    match experimentEventGuid with 
                    | Some guid -> tryGetExperimentEvent guid |> Async.RunSynchronously
                    | None  -> failwithf "Not a valid guid %A" changemap.["--experimentOperationId"]

                let experimentevent = match experimenteventoption with | Some(e) -> e | None -> failwithf "%s not found in BCKG." (experimentEventGuid.ToString())    
                deleteExperimentEvent (experimentId, experimentevent) |> ExperimentEventEventResult
                
            | false ->
                let experimentEventGuid = changemap.[ExperimentOperation.targetExperimentOperationId] |> ExperimentOperationId.fromString
                let experimenteventoption:ExperimentOperation option =
                    match experimentEventGuid with 
                    | Some guid -> tryGetExperimentEvent guid |> Async.RunSynchronously
                    | None  -> failwithf "Not a valid guid %A" changemap.[ExperimentOperation.targetExperimentOperationId]
                let experimentevent = match experimenteventoption with | Some(e) -> e | None -> failwithf "%s not found in BCKG." (experimentEventGuid.ToString())    
                
                let experimentevent = if changemap.ContainsKey("++type") then {experimentevent with Type = changemap.["++type"] |>  ExperimentOperationType.fromString } else experimentevent           
                let experimentevent = if changemap.ContainsKey("++timestamp") then {experimentevent with timestamp =  changemap.["++timestamp"] |> System.DateTime.Parse } else experimentevent
    
                saveExperimentEvent (experimentId, experimentevent) |> ExperimentEventEventResult
                        
    
    let ProcessExperimentEvent saveExperiment tryGetExperiment (experimentId:ExperimentId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let experiment = {
                Experiment.id = experimentId
                Experiment.name = changemap.["++name"]
                Experiment.notes = changemap.["++notes"]
                Experiment.Type = ExperimentType.fromString(changemap.["++type"])
                Experiment.deprecated = changemap.[Experiment.addDeprecate] |> System.Boolean.Parse
            } 
            saveExperiment experiment |> ExperimentEventResult
        
        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let experimentoption = tryGetExperiment experimentId |> Async.RunSynchronously
            let experiment = match experimentoption with | Some(e) -> e | None -> failwithf "%s not found in BCKG." (experimentId.ToString())
            let experiment = if changemap.ContainsKey("++name") then {experiment with Experiment.name = changemap.["++name"]} else experiment
            let experiment = if changemap.ContainsKey("++type") then {experiment with Experiment.Type = ExperimentType.fromString(changemap.["++type"])} else experiment
            let experiment = if changemap.ContainsKey("++notes") then {experiment with Experiment.notes = changemap.["++notes"]} else experiment
            let experiment =  if changemap.ContainsKey(Experiment.addDeprecate) then {experiment with deprecated = changemap.[Experiment.addDeprecate] |> System.Boolean.Parse} else experiment
            
            saveExperiment experiment |> ExperimentEventResult
    
    let ProcessCellEvent saveCell (tryGetCell:CellId->Async<Cell option>) (cellId:CellId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add       ->   
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let cellProperties = 
                {
                    CellProperties.id = cellId
                    CellProperties.name = changemap.[Cell.addName]
                    CellProperties.notes = changemap.[Cell.addNotes]
                    CellProperties.barcode = if changemap.ContainsKey(Cell.addBarcode) then (changemap.[Cell.addBarcode] |> Barcode |> Some) else None
                    CellProperties.deprecated = changemap.[Cell.addDeprecate] |> System.Boolean.Parse
                    CellProperties.genotype = changemap.[Cell.addGenotype]
                }
            let cell = Prokaryote({properties = cellProperties; Type = ProkaryoteType.Bacteria})
            saveCell cell |> CellEventResult
            
        | EventOperation.Modify    -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let celloption = tryGetCell cellId |> Async.RunSynchronously
            let cell = match celloption with | Some (p) -> p | None -> failwithf "%s not found in BCKG." (cellId.ToString())
            let cellProperties = Cell.GetProperties cell
            
            let cellProperties = if changemap.ContainsKey(Cell.addName) then {cellProperties with name = changemap.[Cell.addName]} else cellProperties
            let cellProperties = if changemap.ContainsKey(Cell.addNotes) then {cellProperties with notes = changemap.[Cell.addNotes]} else cellProperties
            let cellProperties = if changemap.ContainsKey(Cell.addGenotype) then {cellProperties with genotype = changemap.[Cell.addGenotype]} else cellProperties
            
            let cellProperties = if changemap.ContainsKey(Cell.removeBarcode) then {cellProperties with barcode = None} else cellProperties
            let cellProperties = if changemap.ContainsKey(Cell.addBarcode) then {cellProperties with barcode = changemap.[Cell.addBarcode] |> Barcode |> Some} else cellProperties
            
            let cellProperties =  if changemap.ContainsKey(Cell.addDeprecate) then {cellProperties with deprecated = changemap.[Cell.addDeprecate] |> System.Boolean.Parse} else cellProperties
            

            let cell = Prokaryote({properties = cellProperties; Type = ProkaryoteType.Bacteria})
            saveCell cell |> CellEventResult

    let ProcessCellEntityEvent (saveCellEntity) (removeCellEntity) (getCellEntities) (cellId:CellId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add ->   
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let cellEntity = 
                {
                   CellEntity.cellId = cellId
                   CellEntity.entity = ReagentId.FromType (changemap.[CellEntity.addEntity] |> System.Guid) (changemap.[CellEntity.addEntityType])
                   CellEntity.compartment = changemap.[CellEntity.addCompartment] |> CellCompartment.fromString
                }
            saveCellEntity cellEntity |> CellEntityEventResult
            
        | EventOperation.Modify    ->
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let entity = ReagentId.FromType (changemap.[CellEntity.removeEntity] |> System.Guid) (changemap.[CellEntity.removeEntityType])
            let compartment = changemap.[CellEntity.removeCompartment] |> CellCompartment.fromString
            let cellEntities = getCellEntities cellId |> Async.RunSynchronously
            match cellEntities |> Array.tryFind (fun (ce:CellEntity) -> (ce.entity = entity) && (ce.compartment = compartment)) with 
            | Some(existingEntity) -> 
                removeCellEntity existingEntity |> CellEntityEventResult
            | None -> failwithf "Entity %s of Cell %s not found in BCKG." (cellId.ToString()) (entity.ToString())
            
    let ProcessInteractionEvent (saveInteraction) (interactionId:InteractionId) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add ->
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let interactionProperties = {
                InteractionProperties.id = interactionId
                InteractionProperties.notes = changemap.[Interaction.addNotes]
                InteractionProperties.deprecated = changemap.[Interaction.addDeprecate] |> System.Boolean.Parse
            }

            let entities = 
                match changemap.[Interaction.addType] with
                | "CodesFor" -> 
                    let prot = changemap.[CodesForInteraction.addProtein] |> System.Guid
                    let cds = changemap.[CodesForInteraction.addCDS] |> System.Guid
                    [
                        (interactionId,InteractionNodeType.Template,cds,PartId.GetType (cds |> CDSId |> CDSPartId) ,0)
                        (interactionId,InteractionNodeType.Product,prot,ReagentId.GetType (prot |> ProteinId |> ProteinReagentId),  0)
                    ]
                | "Genetic Activation" -> 
                    let prom = changemap.[GeneticActivationInteraction.addActivated] |> System.Guid
                    let activator = 
                        let rlist = Decode.Auto.unsafeFromString<(ReagentId)list>(changemap.[GeneticActivationInteraction.addActivator])
                        rlist |> List.map (fun r ->  (interactionId,InteractionNodeType.Activator,r.guid, ReagentId.GetType r,0))
                    (interactionId,InteractionNodeType.Activated,prom, PartId.GetType (prom |> PromoterId |> PromoterPartId) ,0)::activator
                | "Genetic Inhibition" -> 
                    let prom = changemap.[GeneticInhibitionInteraction.addInhibited] |> System.Guid
                    let activator = 
                        let rlist = Decode.Auto.unsafeFromString<(ReagentId)list>(changemap.[GeneticInhibitionInteraction.addInhibitor])
                        rlist |> List.map (fun r ->  (interactionId,InteractionNodeType.Inhibitor,r.guid,ReagentId.GetType r,0))
                    (interactionId,InteractionNodeType.Inhibited,prom,PartId.GetType (prom |> PromoterId |> PromoterPartId) ,0)::activator
                | "Reaction" -> 
                    let reactants = 
                        let rlistlist = Decode.Auto.unsafeFromString<((ReagentId) list) list>(changemap.[ReactionInteraction.addReactants])
                        rlistlist |> List.mapi (fun i rlist -> 
                            rlist |> List.map (fun r -> (interactionId,InteractionNodeType.Reactant,r.guid,ReagentId.GetType r,i))
                            )
                        |> List.fold (fun acc x -> acc@x) []
                    let products = 
                        let rlistlist = Decode.Auto.unsafeFromString<((ReagentId) list) list>(changemap.[ReactionInteraction.addProducts])
                        rlistlist |> List.mapi (fun i rlist -> 
                            rlist |> List.map (fun r -> (interactionId,InteractionNodeType.Product,r.guid,ReagentId.GetType r,i))
                            )
                        |> List.fold (fun acc x -> acc@x) []
                    let enzyme = 
                        let rlist = Decode.Auto.unsafeFromString<(ReagentId)list>(changemap.[ReactionInteraction.addEnzyme])
                        rlist |> List.map (fun r ->  (interactionId,InteractionNodeType.Enzyme,r.guid,ReagentId.GetType r,0))
                    reactants@products@enzyme
                | "Generic Inhibition" -> 
                    let regulators = 
                        let rlist = Decode.Auto.unsafeFromString<(ReagentId) list>(changemap.[GenericInteraction.addRegulator])
                        rlist |> List.map (fun r -> (interactionId,InteractionNodeType.Regulator,r.guid,ReagentId.GetType r,0))
                    let regulated = 
                        let r = Decode.Auto.unsafeFromString<ReagentId>(changemap.[GenericInteraction.addRegulated])
                        (interactionId,InteractionNodeType.Regulated,r.guid,ReagentId.GetType r,0)
                    regulated::regulators
                | "Generic Activation" -> 
                    let regulators = 
                        let rlist = Decode.Auto.unsafeFromString<(ReagentId) list>(changemap.[GenericInteraction.addRegulator])
                        rlist |> List.map (fun r -> (interactionId,InteractionNodeType.Regulator,r.guid,ReagentId.GetType r,0))
                    let regulated = 
                        let r = Decode.Auto.unsafeFromString<ReagentId>(changemap.[GenericInteraction.addRegulated])
                        (interactionId,InteractionNodeType.Regulated,r.guid,ReagentId.GetType r,0)
                    regulated::regulators
                | _ -> failwith "Unknown type of interaction."
            saveInteraction (interactionProperties,(changemap.[Interaction.addType]),entities) |> InteractionEventResult
        | EventOperation.Modify -> failwith "Interactions cannot be modified, only deprecated?"
    
    let ProcessDerivedFromEvent (saveDerivedFrom) (tryGetDerivedFrom) (removeDerivedFrom) (operation:EventOperation) (changestring:string) = 
        match operation with 
        | EventOperation.Add -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let source = changemap.[DerivedFrom.addSource] |> System.Guid
            let target = changemap.[DerivedFrom.addTarget] |> System.Guid
            let dftype = changemap.[DerivedFrom.addType] 
            let derivedFrom = DerivedFrom.fromType source target dftype
            let dfExists = tryGetDerivedFrom derivedFrom |> Async.RunSynchronously
            match dfExists with 
            | Some(_) -> 
                printfn "Derived from already exists. This will return an empty event"
                EmptyEventResult
            | None -> 
                saveDerivedFrom derivedFrom |> DerivedFromEventResult

        | EventOperation.Modify -> 
            let changemap = deserializeEventChange changestring  |> Map.ofList
            let source = changemap.[DerivedFrom.removeSource] |> System.Guid
            let target = changemap.[DerivedFrom.removeTarget] |> System.Guid
            let dftype = changemap.[DerivedFrom.removeType] 
            let derivedFrom = DerivedFrom.fromType source target dftype
            let dfExists = tryGetDerivedFrom derivedFrom |> Async.RunSynchronously
            match dfExists with 
            | Some(_) -> 
                removeDerivedFrom derivedFrom |> DerivedFromEventResult
            | None ->
                printfn "Derived from does not exist. This will return an empty event"
                EmptyEventResult
    
    type TagSourceId =
        | PartTag of PartId
        | ReagentTag of ReagentId
        | CellTag of CellId
        | SampleTag of SampleId
        | ExperimentTag of ExperimentId
        with
        override this.ToString() = 
            match this with
            | PartTag pid -> pid.ToString()
            | ReagentTag rid -> rid.ToString()
            | CellTag cid -> cid.ToString()
            | SampleTag sid -> sid.ToString()
            | ExperimentTag eid -> eid.ToString()
    
    let ProcessTagEvent saveTag deleteTag (tagId:TagSourceId) (operation:EventOperation) (changestring:string) =
        let changemap = deserializeEventChange changestring  |> Map.ofList
        let guid,eventResult = 
            match tagId with 
            | PartTag (partId) -> (partId.ToString() |> System.Guid),PartTagEventResult
            | ReagentTag (reagentId:ReagentId) -> (reagentId.guid),ReagentTagEventResult
            | ExperimentTag (ExperimentId guid) -> guid,ExperimentTagEventResult
            | CellTag (CellId guid) -> guid,CellTagEventResult
            | SampleTag (SampleId guid) -> guid,SampleTagEventResult
        match operation with 
        | EventOperation.Add ->
            let tag = Tag changemap.["++tag"]
            saveTag(guid,tag) |> eventResult
        | EventOperation.Modify -> 
            let tag = Tag changemap.["--tag"]
            deleteTag(guid,tag) |> eventResult
    
    type FileSourceId =
        | ExperimentSource of ExperimentId
        | ReagentSource of ReagentId
        | SampleSource of SampleId
        | CellSource of CellId

    let ProcessAttachedFileEvent saveFileRef deleteFileRef tryGetFileRef (sourceId:FileSourceId) (operation:EventOperation) (changestring:string) =                        
        let changemap = deserializeEventChange changestring  |> Map.ofList
        let guid, eventResult = 
            match sourceId with
            | ReagentSource (reagentId:ReagentId) -> (reagentId.guid), ReagentFileEventResult
            | ExperimentSource (ExperimentId guid) -> guid, ExperimentFileEventResult
            | SampleSource (SampleId guid) -> guid, SampleDataEventResult
            | CellSource (CellId guid) -> guid, CellFileEventResult

        match operation with 
        | EventOperation.Add -> 
            let fileRef = {
                FileRef.fileId = changemap.[FileRef.addFileId] |> System.Guid |> FileId
                FileRef.fileName = changemap.[FileRef.addFileName]
                FileRef.Type = changemap.[FileRef.addType] |> FileType.fromString 
            }
            
            saveFileRef (guid, fileRef) |> eventResult
            
        | EventOperation.Modify -> 
            match changemap.ContainsKey(FileRef.removeFileId) with 
            | true ->
                let fileGuid = changemap.[FileRef.removeFileId] |> FileId.fromString
                let filerefoption = 
                    match fileGuid with 
                    | Some guid -> tryGetFileRef guid |> Async.RunSynchronously
                    | None  -> failwithf "Not a valid fileId %A" changemap.[FileRef.removeFileId]
                let fileRef = match filerefoption with | Some(fref) -> fref | None -> failwithf "%s FileRef not found in BCKG." (fileGuid.ToString())
                
                deleteFileRef (guid, fileRef) |> eventResult
                
            | false ->                 
                let fileGuid = changemap.[FileRef.targetFileId] |> FileId.fromString
                let filerefoption = 
                    match fileGuid with 
                    | Some guid -> tryGetFileRef guid |> Async.RunSynchronously
                    | None  -> failwithf "Not a valid fileId %A" changemap.[FileRef.targetFileId]
                    
                let fileRef = match filerefoption with | Some(fref) -> fref | None -> failwithf "%s FileRef not found in BCKG." (fileGuid.ToString())
                
                let fileRef = if (changemap.ContainsKey(FileRef.addFileName)) then {fileRef with fileName = changemap.[FileRef.addFileName]} else fileRef
                let fileRef = if (changemap.ContainsKey(FileRef.addType)) then {fileRef with Type = changemap.[FileRef.addType] |> FileType.fromString} else fileRef
                
                saveFileRef (guid, fileRef) |> eventResult            

    let ProcessObservationEvent saveObservation (operation:EventOperation) observationId changestring =
        match operation with 
        | EventOperation.Add ->
            let change = deserializeEventChange changestring
            let changemap = change |> Map.ofList
                              
            let observation = 
                { Observation.id = observationId
                  Observation.sampleId = changemap.[Observation.addSampleId] |> System.Guid |> SampleId
                  Observation.signalId = changemap.[Observation.addSignalId] |> System.Guid |> SignalId
                  Observation.value    = float changemap.[Observation.addValue]
                  Observation.timestamp = if changemap.ContainsKey Observation.addTimestamp then Some (System.DateTime.Parse changemap.[Observation.addTimestamp]) else None
                  Observation.replicate = if changemap.ContainsKey Observation.addReplicate then changemap.[Observation.addReplicate] |> System.Guid |> ReplicateId |> Some else None
                }
                              
            saveObservation observation |> ObservationEventResult 

        | EventOperation.Modify -> failwith "Operation not supported: observations are immutable"