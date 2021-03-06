//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Server.RouteTable

open Microsoft.AspNetCore.Http
open Giraffe

open BCKG.Domain

open BCKG_REST_Server.Shared.ClientPaths
open BCKG_REST_Server.Shared.Codec
open BCKG_REST_Server.Shared.Shared

open HandlerUtils

let routeTable (checkPermitted:HttpHandler) (lookupAPI:HttpContext->IBCKGApi) : HttpHandler =
    choose [
        checkPermitted
        POST >=> routef clientPaths.postPartsGuid (post1Handler lookupAPI guidFromString Decoders.getPart (fun iApi -> iApi.postPartsGuidAPI))
        PATCH >=> routef clientPaths.patchPartsGuid (post1Handler lookupAPI guidFromString Decoders.getPart (fun iApi -> iApi.patchPartsGuidAPI))
        GET >=> routef clientPaths.getPartsPromoterGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsPromoterGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsPromoterGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsPromoterGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsPromoterGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsPromoterGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsPromoterGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsPromoterGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsRbsGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsRbsGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsRbsGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsRbsGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsRbsGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsRbsGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsRbsGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsRbsGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsCdsGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsCdsGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsCdsGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsCdsGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsCdsGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsCdsGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsCdsGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsCdsGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsTerminatorGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsTerminatorGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsTerminatorGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsTerminatorGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsTerminatorGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsTerminatorGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsTerminatorGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsTerminatorGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsUserdefinedGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsUserdefinedGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsUserdefinedGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsUserdefinedGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsUserdefinedGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsUserdefinedGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsUserdefinedGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsUserdefinedGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsScarGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsScarGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsScarGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsScarGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsScarGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsScarGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsScarGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsScarGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsBackboneGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsBackboneGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsBackboneGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsBackboneGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsBackboneGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsBackboneGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsBackboneGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsBackboneGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsOriGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsOriGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsOriGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsOriGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsOriGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsOriGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsOriGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsOriGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsLinkerGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsLinkerGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsLinkerGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsLinkerGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsLinkerGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsLinkerGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsLinkerGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsLinkerGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getPartsRestrictionsiteGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsRestrictionsiteGuidAPI) Encoders.setPart)
        GET >=> routef clientPaths.getPartsRestrictionsiteGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getPartsRestrictionsiteGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchPartsRestrictionsiteGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsRestrictionsiteGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchPartsRestrictionsiteGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchPartsRestrictionsiteGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsDnaGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsDnaGuidAPI) Encoders.setReagent)
        POST >=> routef clientPaths.postReagentsDnaGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.postReagentsDnaGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsDnaGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.patchReagentsDnaGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsDnaGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsDnaGuidAddTagsAPI))
        GET >=> routef clientPaths.getReagentsDnaGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsDnaGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchReagentsDnaGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsDnaGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsDnaGuidFileRefs (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsDnaGuidFileRefsAPI) Encoders.setFileRefArray)
        GET >=> routef clientPaths.getReagentsRnaGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsRnaGuidAPI) Encoders.setReagent)
        POST >=> routef clientPaths.postReagentsRnaGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.postReagentsRnaGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsRnaGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.patchReagentsRnaGuidAPI))
        GET >=> routef clientPaths.getReagentsRnaGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsRnaGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchReagentsRnaGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsRnaGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchReagentsRnaGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsRnaGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsRnaGuidFileRefs (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsRnaGuidFileRefsAPI) Encoders.setFileRefArray)
        GET >=> routef clientPaths.getReagentsChemicalGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsChemicalGuidAPI) Encoders.setReagent)
        POST >=> routef clientPaths.postReagentsChemicalGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.postReagentsChemicalGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsChemicalGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.patchReagentsChemicalGuidAPI))
        GET >=> routef clientPaths.getReagentsChemicalGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsChemicalGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchReagentsChemicalGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsChemicalGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchReagentsChemicalGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsChemicalGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsChemicalGuidFileRefs (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsChemicalGuidFileRefsAPI) Encoders.setFileRefArray)
        GET >=> routef clientPaths.getReagentsProteinGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsProteinGuidAPI) Encoders.setReagent)
        POST >=> routef clientPaths.postReagentsProteinGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.postReagentsProteinGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsProteinGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.patchReagentsProteinGuidAPI))
        GET >=> routef clientPaths.getReagentsProteinGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsProteinGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchReagentsProteinGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsProteinGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchReagentsProteinGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsProteinGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsProteinGuidFileRefs (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsProteinGuidFileRefsAPI) Encoders.setFileRefArray)
        GET >=> routef clientPaths.getReagentsGenericentityGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsGenericentityGuidAPI) Encoders.setReagent)
        POST >=> routef clientPaths.postReagentsGenericentityGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.postReagentsGenericentityGuidAPI))
        PATCH >=> routef clientPaths.patchReagentsGenericentityGuid (post1Handler lookupAPI guidFromString Decoders.getReagent (fun iApi -> iApi.patchReagentsGenericentityGuidAPI))
        GET >=> routef clientPaths.getReagentsGenericentityGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsGenericentityGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchReagentsGenericentityGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsGenericentityGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchReagentsGenericentityGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchReagentsGenericentityGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getReagentsGenericentityGuidFileRefs (get1Handler lookupAPI guidFromString (fun iApi->iApi.getReagentsGenericentityGuidFileRefsAPI) Encoders.setFileRefArray)
        GET >=> routef clientPaths.getCellsGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getCellsGuidAPI) Encoders.setCell)
        POST >=> routef clientPaths.postCellsGuid (post1Handler lookupAPI guidFromString Decoders.getCell (fun iApi -> iApi.postCellsGuidAPI))
        PATCH >=> routef clientPaths.patchCellsGuid (post1Handler lookupAPI guidFromString Decoders.getCell (fun iApi -> iApi.patchCellsGuidAPI))
        GET >=> routef clientPaths.getCellsGuidEntities (get1Handler lookupAPI guidFromString (fun iApi->iApi.getCellsGuidEntitiesAPI) Encoders.setCellEntityArray)
        PATCH >=> routef clientPaths.patchCellsGuidAddEntities (post1Handler lookupAPI guidFromString Decoders.getCellEntityArray (fun iApi -> iApi.patchCellsGuidAddEntitiesAPI))
        PATCH >=> routef clientPaths.patchCellsGuidRemoveEntities (post1Handler lookupAPI guidFromString Decoders.getCellEntityArray (fun iApi -> iApi.patchCellsGuidRemoveEntitiesAPI))
        GET >=> routef clientPaths.getCellsGuidTags (get1Handler lookupAPI guidFromString (fun iApi->iApi.getCellsGuidTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchCellsGuidAddTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchCellsGuidAddTagsAPI))
        PATCH >=> routef clientPaths.patchCellsGuidRemoveTags (post1Handler lookupAPI guidFromString Decoders.getstringArray (fun iApi -> iApi.patchCellsGuidRemoveTagsAPI))
        GET >=> routef clientPaths.getSamplesSampleId (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdAPI) Encoders.setSample)
        POST >=> routef clientPaths.postSamplesSampleId (post1Handler lookupAPI SampleId.fromString Decoders.getSample (fun iApi -> iApi.postSamplesSampleIdAPI))
        GET >=> routef clientPaths.getSamplesSampleIdDevices (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdDevicesAPI) Encoders.setSampleDeviceArray)
        GET >=> routef clientPaths.getSamplesSampleIdDeviceCellId (get2Handler lookupAPI SampleId.fromString CellId.fromString (fun iApi->iApi.getSamplesSampleIdDeviceCellIdAPI) Encoders.setSampleDevice)
        PATCH >=> routef clientPaths.patchSamplesSampleIdAddDevices (post1Handler lookupAPI SampleId.fromString Decoders.getSampleDeviceArray (fun iApi -> iApi.patchSamplesSampleIdAddDevicesAPI))
        PATCH >=> routef clientPaths.patchSamplesSampleIdRemoveDevices (post1Handler lookupAPI SampleId.fromString Decoders.getSampleDeviceArray (fun iApi -> iApi.patchSamplesSampleIdRemoveDevicesAPI))
        GET >=> routef clientPaths.getSamplesSampleIdTimeseries (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdTimeseriesAPI) Encoders.setstring)
        GET >=> routef clientPaths.getSamplesSampleIdConditions (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdConditionsAPI) Encoders.setConditionArray)
        PATCH >=> routef clientPaths.patchSamplesSampleIdAddConditions (post1Handler lookupAPI SampleId.fromString Decoders.getConditionArray (fun iApi -> iApi.patchSamplesSampleIdAddConditionsAPI))
        PATCH >=> routef clientPaths.patchSamplesSampleIdRemoveConditions (post1Handler lookupAPI SampleId.fromString Decoders.getConditionArray (fun iApi -> iApi.patchSamplesSampleIdRemoveConditionsAPI))
        GET >=> routef clientPaths.getSamplesSampleIdObservations (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdObservationsAPI) Encoders.setObservationArray)
        PATCH >=> routef clientPaths.patchSamplesSampleIdAddObservations (post1Handler lookupAPI SampleId.fromString Decoders.getObservationArray (fun iApi -> iApi.patchSamplesSampleIdAddObservationsAPI))
        GET >=> routef clientPaths.getSamplesSampleIdSignalIdObservations (get2Handler lookupAPI SampleId.fromString SignalId.fromString (fun iApi->iApi.getSamplesSampleIdSignalIdObservationsAPI) Encoders.setObservationArray)
        GET >=> routef clientPaths.getSamplesSampleIdTags (get1Handler lookupAPI SampleId.fromString (fun iApi->iApi.getSamplesSampleIdTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchSamplesSampleIdAddTags (post1Handler lookupAPI SampleId.fromString Decoders.getstringArray (fun iApi -> iApi.patchSamplesSampleIdAddTagsAPI))
        PATCH >=> routef clientPaths.patchSamplesSampleIdRemoveTags (post1Handler lookupAPI SampleId.fromString Decoders.getstringArray (fun iApi -> iApi.patchSamplesSampleIdRemoveTagsAPI))
        GET >=> routef clientPaths.getExperimentsExperimentId (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdAPI) Encoders.setExperiment)
        POST >=> routef clientPaths.postExperimentsExperimentId (post1Handler lookupAPI ExperimentId.fromString Decoders.getExperiment (fun iApi -> iApi.postExperimentsExperimentIdAPI))
        PATCH >=> routef clientPaths.patchExperimentsExperimentId (post1Handler lookupAPI ExperimentId.fromString Decoders.getExperiment (fun iApi -> iApi.patchExperimentsExperimentIdAPI))
        GET >=> routef clientPaths.getExperimentsExperimentIdSamples (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdSamplesAPI) Encoders.setSampleArray)
        POST >=> routef clientPaths.postExperimentsExperimentIdSamples (post1Handler lookupAPI ExperimentId.fromString Decoders.getSampleArray (fun iApi -> iApi.postExperimentsExperimentIdSamplesAPI))
        GET >=> routef clientPaths.getExperimentsExperimentIdOperations (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdOperationsAPI) Encoders.setExperimentOperationArray)
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdAddOperation (post1Handler lookupAPI ExperimentId.fromString Decoders.getExperimentOperation (fun iApi -> iApi.patchExperimentsExperimentIdAddOperationAPI))
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdRemoveOperation (post1Handler lookupAPI ExperimentId.fromString Decoders.getExperimentOperation (fun iApi -> iApi.patchExperimentsExperimentIdRemoveOperationAPI))
        GET >=> routef clientPaths.getExperimentsExperimentIdSignals (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdSignalsAPI) Encoders.setSignalArray)
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdAddSignals (post1Handler lookupAPI ExperimentId.fromString Decoders.getSignalArray (fun iApi -> iApi.patchExperimentsExperimentIdAddSignalsAPI))
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdRemoveSignals (post1Handler lookupAPI ExperimentId.fromString Decoders.getSignalArray (fun iApi -> iApi.patchExperimentsExperimentIdRemoveSignalsAPI))
        GET >=> routef clientPaths.getExperimentsExperimentIdObservations (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdObservationsAPI) Encoders.setObservationArray)
        GET >=> routef clientPaths.getExperimentsExperimentIdTags (get1Handler lookupAPI ExperimentId.fromString (fun iApi->iApi.getExperimentsExperimentIdTagsAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdAddTags (post1Handler lookupAPI ExperimentId.fromString Decoders.getstringArray (fun iApi -> iApi.patchExperimentsExperimentIdAddTagsAPI))
        PATCH >=> routef clientPaths.patchExperimentsExperimentIdRemoveTags (post1Handler lookupAPI ExperimentId.fromString Decoders.getstringArray (fun iApi -> iApi.patchExperimentsExperimentIdRemoveTagsAPI))
        GET >=> routef clientPaths.getObservationsObservationId (get1Handler lookupAPI observationIdFromString (fun iApi->iApi.getObservationsObservationIdAPI) Encoders.setObservation)
        POST >=> routef clientPaths.postObservationsObservationId (post1Handler lookupAPI observationIdFromString Decoders.getObservation (fun iApi -> iApi.postObservationsObservationIdAPI))
        GET >=> routef clientPaths.getSignalsSignalId (get1Handler lookupAPI SignalId.fromString (fun iApi->iApi.getSignalsSignalIdAPI) Encoders.setSignalArray)
        GET >=> routef clientPaths.getTagsGuid (get1Handler lookupAPI guidFromString (fun iApi->iApi.getTagsGuidAPI) Encoders.setstringArray)
        PATCH >=> routef clientPaths.patchFileRefsGuidLink (post1Handler lookupAPI guidFromString Decoders.getFileRefArray (fun iApi -> iApi.patchFileRefsGuidLinkAPI))
        PATCH >=> routef clientPaths.patchFileRefsGuidUnlink (post1Handler lookupAPI guidFromString Decoders.getFileRefArray (fun iApi -> iApi.patchFileRefsGuidUnlinkAPI))
        GET >=> routef clientPaths.getFilesFileId (get1Handler lookupAPI FileId.fromString (fun iApi->iApi.getFilesFileIdAPI) Encoders.setstring)
        POST >=> routef clientPaths.postFilesFileId (post1Handler lookupAPI FileId.fromString Decoders.getstring (fun iApi -> iApi.postFilesFileIdAPI))

        RequestErrors.NOT_FOUND "Not Found"
]