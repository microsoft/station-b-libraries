//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Server.Storage

open System
open Config
open BCKG.Domain
open BCKG.Events
open BCKG_REST_Server.Shared.Shared
open BCKG_REST_Server.Server.StorageUtils

let getServer (userId:string) (connectionString:string) : IBCKGApi =
    let connectionSettings = 
        connectionString.Split ';'
        |> Array.map (fun x -> x.Trim())
        |> Array.filter ((<>) "")
        |> Array.map(fun opt -> 
            let fields = opt.Split '='
            fields.[0], fields.[1]) 
        |> Map.ofArray    

    printfn "Using connection settings: %A" connectionString
    let db = BCKG.API.Instance(BCKG.API.InstanceType.CloudInstance(connectionString), userId)

    let bckgApi = {

        postPartsGuidAPI = fun (guid:Guid) (body:Part) -> processPart db (guid, body) RequestType.POST

        patchPartsGuidAPI = fun (guid:Guid) (body:Part) -> processPart db (guid, body) RequestType.PATCH

        getPartsPromoterGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> PromoterId |> PromoterPartId))

        getPartsPromoterGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag)

        patchPartsPromoterGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsPromoterGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsRbsGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> RBSId |> RBSPartId))

        getPartsRbsGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag)

        patchPartsRbsGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsRbsGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsCdsGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> CDSId |> CDSPartId))

        getPartsCdsGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag)

        patchPartsCdsGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsCdsGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsTerminatorGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e  |> TerminatorId |> TerminatorPartId))

        getPartsTerminatorGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag)

        patchPartsTerminatorGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsTerminatorGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsUserdefinedGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> UserDefinedId |> UserDefinedPartId))

        getPartsUserdefinedGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag)

        patchPartsUserdefinedGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsUserdefinedGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsScarGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> ScarId |> ScarPartId))

        getPartsScarGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag)

        patchPartsScarGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsScarGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsBackboneGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> BackboneId |> BackbonePartId))

        getPartsBackboneGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag)

        patchPartsBackboneGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsBackboneGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsOriGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> OriId |> OriPartId))

        getPartsOriGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag)

        patchPartsOriGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsOriGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsLinkerGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> LinkerId |> LinkerPartId))

        getPartsLinkerGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag)

        patchPartsLinkerGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsLinkerGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getPartsRestrictionsiteGuidAPI = unpack1ReturnOption (fun e -> db.TryGetPart (e |> RestrictionSiteId |> RestrictionSitePartId))

        getPartsRestrictionsiteGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag)

        patchPartsRestrictionsiteGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)

        patchPartsRestrictionsiteGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)

        getReagentsDnaGuidAPI = unpack1ReturnOption (fun e -> db.TryGetReagent (e |> DNAId |> DNAReagentId))

        postReagentsDnaGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST

        patchReagentsDnaGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH

        patchReagentsDnaGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> DNAId |> DNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)

        getReagentsDnaGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag)

        patchReagentsDnaGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> DNAId |> DNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)
        getReagentsDnaGuidFileRefsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsDnaGuidFileRefsAPI" }

        getReagentsRnaGuidAPI = unpack1ReturnOption (fun e -> db.TryGetReagent (e |> RNAId |> RNAReagentId))

        postReagentsRnaGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST

        patchReagentsRnaGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH
        getReagentsRnaGuidTagsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsRnaGuidTagsAPI" }

        patchReagentsRnaGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)

        patchReagentsRnaGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)
        getReagentsRnaGuidFileRefsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsRnaGuidFileRefsAPI" }

        getReagentsChemicalGuidAPI = unpack1ReturnOption (fun e -> db.TryGetReagent (e |> ChemicalId |> ChemicalReagentId))

        postReagentsChemicalGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST

        patchReagentsChemicalGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH

        getReagentsChemicalGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag)

        patchReagentsChemicalGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)

        patchReagentsChemicalGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)
        getReagentsChemicalGuidFileRefsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsChemicalGuidFileRefsAPI" }

        getReagentsProteinGuidAPI = unpack1ReturnOption (fun e -> db.TryGetReagent (e |> ProteinId |> ProteinReagentId))

        postReagentsProteinGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST

        patchReagentsProteinGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH

        getReagentsProteinGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag)

        patchReagentsProteinGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)

        patchReagentsProteinGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)
        getReagentsProteinGuidFileRefsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsProteinGuidFileRefsAPI" }

        getReagentsGenericentityGuidAPI = unpack1ReturnOption (fun e -> db.TryGetReagent (e |> GenericEntityId |> GenericEntityReagentId))

        postReagentsGenericentityGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST

        patchReagentsGenericentityGuidAPI = fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH

        getReagentsGenericentityGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag)

        patchReagentsGenericentityGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)

        patchReagentsGenericentityGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)
        getReagentsGenericentityGuidFileRefsAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getReagentsGenericentityGuidFileRefsAPI" }

        getCellsGuidAPI = unpack1ReturnOption (fun e -> db.TryGetCell (e  |> CellId))

        postCellsGuidAPI = fun (guid:Guid) (body:Cell) -> processCell db (guid, body) RequestType.POST

        patchCellsGuidAPI = fun (guid:Guid) (body:Cell) -> processCell db (guid, body) RequestType.PATCH

        getCellsGuidEntitiesAPI = fun (guid:Guid) -> db.GetCellEntities (guid |> CellId)

        patchCellsGuidAddEntitiesAPI = unpack2ReturnBool(fun (guid:Guid) (body:CellEntity[]) -> db.SaveCellEntities (guid |> CellId) (body))

        patchCellsGuidRemoveEntitiesAPI = unpack2ReturnBool(fun (guid:Guid) (body:CellEntity[]) -> db.RemoveCellEntities (guid |> CellId) (body))

        getCellsGuidTagsAPI = fun (guid:Guid) -> getTags db (guid |> CellId |> EventsProcessor.CellTag)

        patchCellsGuidAddTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CellId |> EventsProcessor.CellTag, body) (AddRemoveType.ADD)

        patchCellsGuidRemoveTagsAPI = fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CellId |> EventsProcessor.CellTag, body) (AddRemoveType.REMOVE)

        getSamplesSampleIdAPI = unpack1ReturnOption(fun e-> db.TryGetSample e )

        postSamplesSampleIdAPI = fun (sampleId: SampleId) (body: Sample) -> processSample db (sampleId, body) RequestType.POST

        getSamplesSampleIdDevicesAPI = fun e-> db.GetSampleDevices e
        getSamplesSampleIdDeviceCellIdAPI = fun ((sampleId,cellId):SampleId*CellId) -> async { return failwith "Missing API.fs function getSamplesSampleIdDeviceCellIdAPI" }

        patchSamplesSampleIdAddDevicesAPI = fun (sampleId:SampleId) (body: SampleDevice[]) -> processDeviceEvent db (sampleId , body) (AddRemoveType.ADD)

        patchSamplesSampleIdRemoveDevicesAPI = fun (sampleId:SampleId) (body: SampleDevice[]) -> processDeviceEvent db (sampleId , body) (AddRemoveType.REMOVE)

        getSamplesSampleIdTimeseriesAPI = unpack1ReturnOption db.TryGetTimeSeries

        getSamplesSampleIdConditionsAPI = db.GetSampleConditions

        patchSamplesSampleIdAddConditionsAPI = fun (sampleId: SampleId) (body:Condition[]) -> processSampleConditionEvent db (sampleId, body) (AddRemoveType.ADD)

        patchSamplesSampleIdRemoveConditionsAPI = fun (sampleId: SampleId) (body:Condition[]) -> processSampleConditionEvent db (sampleId, body) (AddRemoveType.REMOVE)

        getSamplesSampleIdObservationsAPI = db.GetSampleObservations

        patchSamplesSampleIdAddObservationsAPI = unpack2ReturnBool (fun (sampleId:SampleId) (body:Observation[]) -> db.SaveObservations body)

        getSamplesSampleIdSignalIdObservationsAPI = fun (sampleId: SampleId, signalId: SignalId) -> db.GetObservations (sampleId, signalId)

        getSamplesSampleIdTagsAPI = fun (guid:SampleId) -> getTags db (guid |> EventsProcessor.SampleTag)

        patchSamplesSampleIdAddTagsAPI = unpackAsyncList (fun e t->db.AddSampleTag(e, (Tag)t))

        patchSamplesSampleIdRemoveTagsAPI = unpackAsyncList (fun e t->db.RemoveSampleTag(e, (Tag)t))

        getExperimentsExperimentIdAPI = unpack1ReturnOption db.TryGetExperiment

        postExperimentsExperimentIdAPI = fun (ExperimentId guid) (body:Experiment) -> processExperiment db (guid, body) RequestType.POST

        patchExperimentsExperimentIdAPI = fun (ExperimentId guid) (body:Experiment) -> processExperiment db (guid, body) RequestType.PATCH

        getExperimentsExperimentIdSamplesAPI = db.GetExperimentSamples

        postExperimentsExperimentIdSamplesAPI = unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Sample[]) -> db.SaveSamples body)

        getExperimentsExperimentIdOperationsAPI = fun (experimentId:ExperimentId) -> db.GetExperimentOperations experimentId

        patchExperimentsExperimentIdAddOperationAPI = fun (experimentId:ExperimentId) (body:ExperimentOperation) -> processExperimentOperation db (experimentId, body) (AddRemoveType.ADD)

        patchExperimentsExperimentIdRemoveOperationAPI = fun (experimentId:ExperimentId) (body:ExperimentOperation) -> processExperimentOperation db (experimentId, body) (AddRemoveType.REMOVE)

        getExperimentsExperimentIdSignalsAPI = fun (experimentId:ExperimentId) -> db.GetExperimentSignals experimentId

        patchExperimentsExperimentIdAddSignalsAPI = unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Signal[]) -> db.SaveExperimentSignals(experimentId,body))

        patchExperimentsExperimentIdRemoveSignalsAPI = unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Signal[]) -> db.RemoveExperimentSignals(experimentId,body))
        getExperimentsExperimentIdObservationsAPI = fun (experimentId:ExperimentId) -> async { return failwith "Missing API.fs function getExperimentsExperimentIdObservationsAPI" }

        getExperimentsExperimentIdTagsAPI = fun (guid:ExperimentId) -> getTags db (guid |> EventsProcessor.ExperimentTag)

        patchExperimentsExperimentIdAddTagsAPI = unpackAsyncList (fun e t->db.AddExperimentTag(e, (Tag)t))

        patchExperimentsExperimentIdRemoveTagsAPI = unpackAsyncList (fun e t->db.RemoveExperimentTag(e, (Tag)t))

        getObservationsObservationIdAPI = unpack1ReturnOption ( fun e -> db.TryGetObservation e)
        postObservationsObservationIdAPI = fun (observationId:ObservationId) (body:Observation) -> async { failwith "Missing API.fs function postObservationsObservationIdAPI" }
        getSignalsSignalIdAPI = fun (signalId:SignalId) -> async { return failwith "Missing API.fs function getSignalsSignalIdAPI" }
        getTagsGuidAPI = fun (guid:Guid) -> async { return failwith "Missing API.fs function getTagsGuidAPI" }
        patchFileRefsGuidLinkAPI = fun (guid:Guid) (body:FileRef[]) -> async { failwith "Missing API.fs function patchFileRefsGuidLinkAPI" }
        patchFileRefsGuidUnlinkAPI = fun (guid:Guid) (body:FileRef[]) -> async { failwith "Missing API.fs function patchFileRefsGuidUnlinkAPI" }
        getFilesFileIdAPI = fun (fileId:FileId) -> async { return failwith "Missing API.fs function getFilesFileIdAPI" }
        postFilesFileIdAPI = fun (fileId:FileId) (body:string) -> async { failwith "Missing API.fs function postFilesFileIdAPI" }

    }
    bckgApi

type BCKGApiConfig(connectionString:string) =
    interface IBCKGApiConfig with
        member _.GetApi (userId:string) : IBCKGApi =
            getServer userId connectionString
