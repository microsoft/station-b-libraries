module BCKG_REST_Server.Server.StorageUtils

open BCKG.Events

let serverErrorMessage = "Server error"
let parameterErrorMessage = "Parameter error"

type RequestType =
    | POST
    | PATCH

type AddRemoveType =
    | ADD
    | REMOVE

let unpack1ReturnBool<'A> (fn:'A->Async<bool>) : ('A->Async<unit>) =
    fun (a:'A) -> async {
        let! r = fn a
        match r with
        | true -> return ()
        | false -> return failwith serverErrorMessage }

let unpack2ReturnBool<'A,'B> (fn:'A->'B->Async<bool>) : ('A->'B->Async<unit>) =
    fun (a:'A) (b:'B) -> async {
        let! r = fn a b
        match r with
        | true -> return ()
        | false -> return failwith serverErrorMessage }

let unpack3ReturnBool<'A,'B, 'C> (fn:'A->'B->'C->Async<bool>) : ('A->'B->'C->Async<unit>) =
    fun (a:'A) (b:'B) (c:'C) -> async {
        let! r = fn a b c
        match r with
        | true -> return ()
        | false -> return failwith serverErrorMessage }

let unpack1ReturnOption<'A, 'R> (fn:'A->Async<Option<'R>>) : ('A->Async<'R>) =
    fun (a:'A) -> async {
        let! r = fn a
        match r with
        | Some t-> return t
        | None-> return failwith serverErrorMessage }

let unpack2ReturnOption<'A, 'B, 'R> (fn:'A->'B->Async<Option<'R>>) : ('A->'B->Async<'R>) =
    fun (a:'A) (b:'B) -> async {
        let! r = fn a b
        match r with
        | Some t-> return t
        | None-> return failwith serverErrorMessage }

let unpack3ReturnOption<'A, 'B, 'C, 'R> (fn:'A->'B->'C->Async<Option<'R>>) : ('A->'B->'C->Async<'R>) =
    fun (a:'A) (b:'B) (c:'C) -> async {
        let! r = fn a b c
        match r with
        | Some t-> return t
        | None-> return failwith serverErrorMessage }

let convertPair<'A, 'B> (convA:string->Option<'A>) (convB:string->Option<'B>) : (string*string->'A*'B) =
    fun ((a, b):string*string) ->
        let optA = convA a
        match optA with
        | Some aa ->
            let optB = convB b
            match optB with
            | Some bb -> (aa, bb)
            | None -> failwith parameterErrorMessage
        | None -> failwith parameterErrorMessage

let unpackAsyncList<'A,'B,'C> (fn:'A->'B->Async<'C>) =
    fun e ts -> async {
        ts
        |> Array.map (fun t -> fn e t)
        |> Async.Sequential
        |> Async.RunSynchronously
        |> ignore
}

let processPart (db:BCKG.API.Instance) (guid:System.Guid, part:BCKG.Domain.Part) (request:RequestType) =    
    async{
        match guid.ToString() = part.id.ToString() with
        | true ->
            let! partExists = db.TryGetPart part.id
            match (partExists,request) with
            | Some(x), RequestType.POST  -> return failwithf "Part with ID %s already exists. Either submit a PATCH request or create a new part ID." (part.id.ToString())
            | None, RequestType.POST ->
                let! savePart = db.SavePart part
                return ()
            | Some(x), RequestType.PATCH  ->
                let! res = db.SavePart part
                return ()        
            | None, RequestType.PATCH -> return failwithf "Part with ID %s does not exist in BCKG." (part.id.ToString())
        | false -> return failwithf "GUID provided in the URL %s does not match the GUID provided in the Entity %s" (guid.ToString()) (part.id.ToString())
    }

let processReagent (db:BCKG.API.Instance) (guid:System.Guid, reagent:BCKG.Domain.Reagent) (request:RequestType) =    
    async{
        match guid.ToString() = reagent.id.ToString() with
        | true ->
            let! reagentExists = db.TryGetReagent reagent.id
            match (reagentExists,request) with
            | Some(x), RequestType.POST  -> return failwithf "Reagent with ID %s already exists. Either submit a PATCH request or create a new reagent ID." (reagent.id.ToString())
            | None, RequestType.POST ->
                let! res = db.SaveReagent reagent
                return ()
            | Some(x), RequestType.PATCH  ->
                let! res = db.SaveReagent reagent
                return ()        
            | None, RequestType.PATCH -> return failwithf "Reagent with ID %s does not exist in BCKG." (reagent.id.ToString())
        | false -> return failwithf "GUID provided in the URL %s does not match the GUID provided in the Entity %s" (guid.ToString()) (reagent.id.ToString())
    }

let processCell (db:BCKG.API.Instance) (guid:System.Guid, cell:BCKG.Domain.Cell) (request:RequestType) =    
    async{
        match guid.ToString() = cell.id.ToString() with
        | true ->
            let! cellExists = db.TryGetCell cell.id
            match (cellExists,request) with
            | Some(x), RequestType.POST  -> return failwithf "Cell with ID %s already exists. Either submit a PATCH request or create a new cell ID." (cell.id.ToString())
            | None, RequestType.POST ->
                let! res = db.SaveCell cell
                return ()
            | Some(x), RequestType.PATCH  ->
                let! res = db.SaveCell cell
                return ()        
            | None, RequestType.PATCH -> return failwithf "Cell with ID %s does not exist in BCKG." (cell.id.ToString())
        | false -> return failwithf "GUID provided in the URL %s does not match the GUID provided in the Entity %s" (guid.ToString()) (cell.id.ToString())
    }

let processExperiment (db:BCKG.API.Instance) (guid:System.Guid, expt:BCKG.Domain.Experiment) (request:RequestType) =    
    async{
        match guid.ToString() = expt.id.ToString() with
        | true ->
            let! cellExists = db.TryGetExperiment expt.id
            match (cellExists,request) with
            | Some(x), RequestType.POST  -> return failwithf "Experiment with ID %s already exists. Either submit a PATCH request or create a new experiment ID." (expt.id.ToString())
            | None, RequestType.POST ->
                let! res = db.SaveExperiment expt
                return ()
            | Some(x), RequestType.PATCH  ->
                let! res = db.SaveExperiment expt
                return ()        
            | None, RequestType.PATCH -> return failwithf "Experiment with ID %s does not exist in BCKG." (expt.id.ToString())
        | false -> return failwithf "GUID provided in the URL %s does not match the GUID provided in the Entity %s" (guid.ToString()) (expt.id.ToString())
    }

let processExperimentOperation (db:BCKG.API.Instance) (experimentId:BCKG.Domain.ExperimentId, op:BCKG.Domain.ExperimentOperation) (request: AddRemoveType) =
    async{
        match request with
        | ADD ->
            let! res = db.SaveExperimentOperation(experimentId, op)
            return ()
        | REMOVE ->
            let! res = db.RemoveExperimentOperation experimentId op
            return ()
    }
    

let processTagEvent (db:BCKG.API.Instance) (entityId:EventsProcessor.TagSourceId, tags:string[]) (request:AddRemoveType)=
    let tags_array = tags |> Array.map (fun x -> BCKG.Domain.Tag x)
    //Have some way of check if that Entity ID exist?
    async{
        match entityId with
        | EventsProcessor.PartTag(pid) ->
            match request with
            | ADD ->
                let! _ = db.AddPartTags (pid,tags_array)
                return ()
            | REMOVE ->
                let! _ = db.RemovePartTags (pid,tags_array)
                return ()
        | EventsProcessor.ReagentTag(rid) ->
            match request with
            | ADD ->
                let! _ = db.AddReagentTags (rid,tags_array)
                return ()
            | REMOVE ->
                let! _ = db.RemoveReagentTags (rid,tags_array)
                return ()
        | EventsProcessor.CellTag(cid) ->
            match request with
            | ADD ->
                let! _ = db.AddCellTags (cid,tags_array)
                return ()
            | REMOVE ->
                let! _ = db.RemoveCellTags (cid,tags_array)
                return ()
        | EventsProcessor.SampleTag(sid) ->
            match request with
            | ADD ->
                let! _ = db.AddSampleTags (sid,tags_array)
                return ()
            | REMOVE ->
                let! _ = db.RemoveSampleTags (sid,tags_array)
                return ()
        | EventsProcessor.ExperimentTag(eid) ->
            match request with
            | ADD ->
                let! _ = db.AddExperimentTags (eid,tags_array)
                return ()
            | REMOVE ->
                let! _ = db.RemoveExperimentTags (eid,tags_array)
                return ()
    }

let getTags (db:BCKG.API.Instance) (entityId:EventsProcessor.TagSourceId) =
    async{
        let! tags =
            match entityId with
            | EventsProcessor.PartTag(pid) -> db.GetPartTags pid
            | EventsProcessor.ReagentTag(rid) -> db.GetReagentTags rid
            | EventsProcessor.CellTag(cid) -> db.GetCellTags cid
            | EventsProcessor.SampleTag(sid) -> db.GetSampleTags sid
            | EventsProcessor.ExperimentTag(eid) -> db.GetExperimentTags eid
        let tag_string_list = tags |> Array.map (fun t -> t.ToString())
        return tag_string_list
    }

let processSample (db: BCKG.API.Instance) (sampleId: BCKG.Domain.SampleId, sample:BCKG.Domain.Sample) (request: RequestType) =
    async {
        match sampleId.ToString() = sample.id.ToString() with
        | true ->
            let! sampleExists = db.TryGetSample sample.id
            match (sampleExists, request) with
            | Some(x), RequestType.POST -> return failwithf "Sample with ID %s already exists. Either submit a PATCH request or create a new reagent ID." (sample.id.ToString())
            | None, RequestType.POST ->
                let! res = db.SaveSamples [| sample |]
                return ()
            | Some(x), RequestType.PATCH ->
                let! res = db.SaveSamples [| sample |]
                return ()
            | None, RequestType.PATCH -> return failwithf "Sample with ID %s does not exist in BCKG" (sample.id.ToString())

        | false -> return failwithf "GUID provided in the URL %s does not match the GUID provided in the Entity %s" (sampleId.ToString()) (sample.id.ToString())
    }

let processSampleConditionEvent (db: BCKG.API.Instance) (sampleId: BCKG.Domain.SampleId, conditions:BCKG.Domain.Condition[]) (request: AddRemoveType)  =
    async {
        match request with
        | ADD ->
            let! res = db.SaveSampleConditions(sampleId, conditions)
            return ()
        | REMOVE ->
            let! res = db.RemoveSampleConditions(sampleId, conditions)
            return()
    }

let processDeviceEvent (db: BCKG.API.Instance) (sampleId: BCKG.Domain.SampleId, devices: BCKG.Domain.SampleDevice[]) (request: AddRemoveType)  =
    async {
        match request with
        | ADD ->
            let! res = db.SaveSampleDevices(sampleId, devices)
            return ()
        | REMOVE ->
            let! res = db.RemoveSampleDevices(sampleId, devices)
            return()
    }