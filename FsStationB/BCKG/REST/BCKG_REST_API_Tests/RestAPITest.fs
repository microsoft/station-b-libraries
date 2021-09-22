// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Test.APITests

open BCKG.Test.Entities
open BCKG.Domain
open BCKG.Events
open FSharp.Data
open Thoth.Json.Net
open FSharp.Data.HttpRequestHeaders
open FSharp.Data.JsonExtensions

type RequestType =
    | POST
    | PATCH

type AddRemoveType =
    | ADD
    | REMOVE

type RestAPI(endpoint:string) =

    member private this.get_request(url) =
        FSharp.Data.Http.RequestString
            (url,
            httpMethod = "GET",
            headers =
              [
                  Accept FSharp.Data.HttpContentTypes.Json
              ]) |> FSharp.Data.JsonValue.Parse

    member private this.post_request (url) (request:Thoth.Json.Net.JsonValue) =
        FSharp.Data.Http.RequestString
            ( url,
              httpMethod = "POST",
              headers =
                [
                    Accept FSharp.Data.HttpContentTypes.Json
                ],
              body = FSharp.Data.TextRequest (Encode.toString 2 request))
        |> FSharp.Data.JsonValue.Parse

    member private this.patch_request (url) (request:Thoth.Json.Net.JsonValue) =
        FSharp.Data.Http.RequestString
            ( url,
              httpMethod = "PATCH",
              headers =
                [
                    Accept FSharp.Data.HttpContentTypes.Json
                ],
              body = FSharp.Data.TextRequest (Encode.toString 2 request))
        |> FSharp.Data.JsonValue.Parse

    member private this.endpoint =
        if not (endpoint.EndsWith("/")) then
            endpoint + "/"
        else
            endpoint

    member this.getPart (part_id:PartId) =
        let rest_uri = this.endpoint + "api/parts/" + (PartId.GetType part_id).ToLower() + "/" + part_id.ToString()
        get_entity (Decode.fromString Part.decode ((this.get_request rest_uri).ToString()))

    member this.modifyPart (part:Part) (modifyType:RequestType) =
        let rest_uri = this.endpoint + "api/parts/" + (part.id.ToString())
        match modifyType with
        | POST -> this.post_request rest_uri (Part.encode part)
        | PATCH -> this.patch_request rest_uri (Part.encode part)

    member this.getReagent (reagent_id:ReagentId) =
        let rest_uri = this.endpoint + "api/reagents/" + (ReagentId.GetType reagent_id).ToLower() + "/" + reagent_id.ToString()
        get_entity (Decode.fromString Reagent.decode ((this.get_request rest_uri).ToString()))

    member this.modifyReagent (reagent:Reagent) (modifyType:RequestType) =
        let rest_uri = this.endpoint + "api/reagents/" + (ReagentId.GetType reagent.id).ToLower() + "/" + (reagent.id.ToString())
        match modifyType with
        | POST -> this.post_request rest_uri (Reagent.encode reagent)
        | PATCH -> this.patch_request rest_uri (Reagent.encode reagent)

    member this.getCell (cell_id:CellId) =
        let rest_uri = this.endpoint + "api/cells/" + cell_id.ToString()
        get_entity (Decode.fromString Cell.decode ((this.get_request rest_uri).ToString()))

    member this.modifyCell (cell:Cell) (modifyType:RequestType) =
        let rest_uri = this.endpoint + "api/cells/" + (cell.id.ToString())
        match modifyType with
        | POST -> this.post_request rest_uri (Cell.encode cell)
        | PATCH -> this.patch_request rest_uri (Cell.encode cell)

    member this.getCellEntities (cell_id:CellId) =
        let rest_uri = this.endpoint + "api/cells/" + (cell_id.ToString()) + "/entities"
        get_entity (Decode.fromString (Decode.array CellEntity.decode) ((this.get_request rest_uri).ToString()))

    member this.updateCellEntities (cell_id:CellId) (entities:CellEntity[])  (addremove:AddRemoveType)=
        let entity_str =
            match addremove with
            | ADD -> "add-entities"
            | REMOVE -> "remove-entities"
        let rest_uri = this.endpoint + "api/cells/" + (cell_id.ToString()) + "/" + entity_str
        let entity_list = (entities |> Array.map (fun e -> CellEntity.encode e)) |> Encode.array
        this.patch_request rest_uri entity_list

    member this.getExperiment (expt_id:ExperimentId) =
        let rest_uri = this.endpoint + "api/experiments/" + expt_id.ToString()
        get_entity (Decode.fromString Experiment.decode ((this.get_request rest_uri).ToString()))

    member this.modifyExperiment (expt:Experiment) (modifyType:RequestType) =
        let rest_uri = this.endpoint + "api/experiments/" + (expt.id.ToString())
        match modifyType with
        | POST -> this.post_request rest_uri (Experiment.encode expt)
        | PATCH -> this.patch_request rest_uri (Experiment.encode expt)

    member this.getExperimentSignals (expt_id:ExperimentId) =
        let rest_uri = this.endpoint + "api/experiments/" + expt_id.ToString() + "/signals"
        get_entity (Decode.fromString (Decode.array Signal.decode) ((this.get_request rest_uri).ToString()))

    member this.updateSignals (expt_id:ExperimentId) (signals:Signal[]) (addremove:AddRemoveType) =
        let entity_str =
            match addremove with
            | ADD -> "add-signals"
            | REMOVE -> "remove-signals"
        let rest_uri = this.endpoint + "api/experiments/" + (expt_id.ToString()) + "/" + entity_str
        let entity_list = (signals |> Array.map (fun e -> Signal.encode e)) |> Encode.array
        this.patch_request rest_uri entity_list

    member this.getExperimentOperations (expt_id:ExperimentId) =
        let rest_uri = this.endpoint + "api/experiments/" + expt_id.ToString() + "/operations"
        get_entity (Decode.fromString (Decode.array ExperimentOperation.decode) ((this.get_request rest_uri).ToString()))

    member this.updateExperimentOperations (expt_id:ExperimentId) (eo:ExperimentOperation) (addremove:AddRemoveType) =
        let entity_str =
            match addremove with
            | ADD -> "add-operation"
            | REMOVE -> "remove-operation"
        let rest_uri = this.endpoint + "api/experiments/" + (expt_id.ToString()) + "/" + entity_str
        let entity_str = ExperimentOperation.encode eo
        this.patch_request rest_uri entity_str

    member this.addExperimentSamples (expt_id:ExperimentId) (samples: Sample []) =
        let rest_uri = this.endpoint + "api/experiments/" + (expt_id.ToString()) + "/samples"
        let entity_str =  (samples |> Array.map (fun s -> Sample.encode s)) |> Encode.array
        this.post_request rest_uri entity_str

    member this.getExperimentSamples (expt_id:ExperimentId) =
        let rest_uri = this.endpoint + "api/experiments/" + (expt_id.ToString()) + "/samples"
        get_entity (Decode.fromString (Decode.array Sample.decode) ((this.get_request rest_uri).ToString()))

    member this.getSample (sample_id:SampleId) =
        let rest_uri = this.endpoint + "api/samples/" + sample_id.ToString()
        get_entity (Decode.fromString Sample.decode ((this.get_request rest_uri).ToString()))

    member this.addSample (sample:Sample) =
        let rest_uri = this.endpoint + "api/samples/" + sample.id.ToString()
        this.post_request rest_uri (Sample.encode sample)

    member this.getSampleConditions (sample_id:SampleId) =
        let rest_uri = this.endpoint + "api/samples/" + sample_id.ToString() + "/conditions"
        get_entity (Decode.fromString (Decode.array Condition.decode) ((this.get_request rest_uri).ToString()))

    member this.updateSampleConditions (sample_id:SampleId) (conditions:Condition []) (addremove:AddRemoveType) =
        let entity_str =
            match addremove with
            | ADD -> "add-conditions"
            | REMOVE -> "remove-conditions"
        let rest_uri = this.endpoint + "api/samples/" + sample_id.ToString() + "/" + entity_str
        let entity_list = (conditions |> Array.map (fun e -> Condition.encode e)) |> Encode.array
        this.patch_request rest_uri entity_list

    member this.getSampleDevices (sample_id:SampleId) =
        let rest_uri = this.endpoint + "api/samples/" + sample_id.ToString() + "/devices"
        get_entity (Decode.fromString (Decode.array SampleDevice.decode) ((this.get_request rest_uri).ToString()))

    member this.updateSampleDevices (sample_id:SampleId) (devices:SampleDevice []) (addremove:AddRemoveType) =
        let entity_str =
            match addremove with
            | ADD -> "add-devices"
            | REMOVE -> "remove-devices"
        let rest_uri = this.endpoint + "api/samples/" + sample_id.ToString() + "/" + entity_str
        let entity_list = (devices |> Array.map (fun e -> SampleDevice.encode e)) |> Encode.array
        this.patch_request rest_uri entity_list

    member this.getTags (entityId:EventsProcessor.TagSourceId) =
        let rest_uri =
            match entityId with
            | EventsProcessor.PartTag pid ->
                this.endpoint + "api/parts/" + (PartId.GetType pid).ToLower() + "/" + pid.ToString() + "/tags"
            | EventsProcessor.ReagentTag rid ->
                this.endpoint + "api/reagents/" + (ReagentId.GetType rid).ToLower() + "/" + rid.ToString() + "/tags"
            | EventsProcessor.CellTag cid -> this.endpoint + "api/cells/" + (cid.ToString()) + "/tags"
            | EventsProcessor.SampleTag sid -> this.endpoint + "api/samples/" + (sid.ToString()) + "/tags"
            | EventsProcessor.ExperimentTag eid -> this.endpoint + "api/experiments/" + (eid.ToString()) + "/tags"
        (this.get_request rest_uri).AsArray()
        |> Seq.map (fun x -> x.AsString())
        |> Seq.toList
        |> List.map (fun t -> Tag t)

    member this.updateTags (entityId:EventsProcessor.TagSourceId) (tags:Tag []) (addremove:AddRemoveType)=
        let tag_array = tags |> Array.map (fun t ->Encode.string (t.ToString())) |> Encode.array
        let tag_str =
            match addremove with
            | ADD -> "add-tags"
            | REMOVE -> "remove-tags"
        let rest_uri =
            match entityId with
            | EventsProcessor.PartTag pid ->
                this.endpoint + "api/parts/" + (PartId.GetType pid).ToLower() + "/" + pid.ToString() + "/" + tag_str
            | EventsProcessor.ReagentTag rid ->
                this.endpoint + "api/reagents/" + (ReagentId.GetType rid).ToLower() + "/" + rid.ToString() + "/" + tag_str
            | EventsProcessor.CellTag cid -> this.endpoint + "api/cells/" + (cid.ToString()) + "/" + tag_str
            | EventsProcessor.SampleTag sid -> this.endpoint + "api/samples/" + (sid.ToString()) + "/" + tag_str
            | EventsProcessor.ExperimentTag eid -> this.endpoint + "api/experiments/" + (eid.ToString()) + "/" + tag_str
        this.patch_request rest_uri tag_array