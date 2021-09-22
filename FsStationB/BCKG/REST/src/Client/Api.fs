// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG_REST_Server.Client.Api

open BCKG.Domain

open BCKG_REST_Server.Shared.ClientPaths
open BCKG_REST_Server.Shared.Codec
open BCKG_REST_Server.Shared.Shared

open HandlerUtils

let bckgApi : IBCKGApi = {
        postPartsGuidAPI = post1Handler guidToString clientPaths.postPartsGuid Encoders.setPart
        patchPartsGuidAPI = post1Handler guidToString clientPaths.patchPartsGuid Encoders.setPart
        getPartsPromoterGuidAPI = get1Handler guidToString clientPaths.getPartsPromoterGuid Decoders.getPart
        getPartsPromoterGuidTagsAPI = get1Handler guidToString clientPaths.getPartsPromoterGuidTags Decoders.getstringArray
        patchPartsPromoterGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsPromoterGuidAddTags Encoders.setstringArray
        patchPartsPromoterGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsPromoterGuidRemoveTags Encoders.setstringArray
        getPartsRbsGuidAPI = get1Handler guidToString clientPaths.getPartsRbsGuid Decoders.getPart
        getPartsRbsGuidTagsAPI = get1Handler guidToString clientPaths.getPartsRbsGuidTags Decoders.getstringArray
        patchPartsRbsGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsRbsGuidAddTags Encoders.setstringArray
        patchPartsRbsGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsRbsGuidRemoveTags Encoders.setstringArray
        getPartsCdsGuidAPI = get1Handler guidToString clientPaths.getPartsCdsGuid Decoders.getPart
        getPartsCdsGuidTagsAPI = get1Handler guidToString clientPaths.getPartsCdsGuidTags Decoders.getstringArray
        patchPartsCdsGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsCdsGuidAddTags Encoders.setstringArray
        patchPartsCdsGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsCdsGuidRemoveTags Encoders.setstringArray
        getPartsTerminatorGuidAPI = get1Handler guidToString clientPaths.getPartsTerminatorGuid Decoders.getPart
        getPartsTerminatorGuidTagsAPI = get1Handler guidToString clientPaths.getPartsTerminatorGuidTags Decoders.getstringArray
        patchPartsTerminatorGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsTerminatorGuidAddTags Encoders.setstringArray
        patchPartsTerminatorGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsTerminatorGuidRemoveTags Encoders.setstringArray
        getPartsUserdefinedGuidAPI = get1Handler guidToString clientPaths.getPartsUserdefinedGuid Decoders.getPart
        getPartsUserdefinedGuidTagsAPI = get1Handler guidToString clientPaths.getPartsUserdefinedGuidTags Decoders.getstringArray
        patchPartsUserdefinedGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsUserdefinedGuidAddTags Encoders.setstringArray
        patchPartsUserdefinedGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsUserdefinedGuidRemoveTags Encoders.setstringArray
        getPartsScarGuidAPI = get1Handler guidToString clientPaths.getPartsScarGuid Decoders.getPart
        getPartsScarGuidTagsAPI = get1Handler guidToString clientPaths.getPartsScarGuidTags Decoders.getstringArray
        patchPartsScarGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsScarGuidAddTags Encoders.setstringArray
        patchPartsScarGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsScarGuidRemoveTags Encoders.setstringArray
        getPartsBackboneGuidAPI = get1Handler guidToString clientPaths.getPartsBackboneGuid Decoders.getPart
        getPartsBackboneGuidTagsAPI = get1Handler guidToString clientPaths.getPartsBackboneGuidTags Decoders.getstringArray
        patchPartsBackboneGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsBackboneGuidAddTags Encoders.setstringArray
        patchPartsBackboneGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsBackboneGuidRemoveTags Encoders.setstringArray
        getPartsOriGuidAPI = get1Handler guidToString clientPaths.getPartsOriGuid Decoders.getPart
        getPartsOriGuidTagsAPI = get1Handler guidToString clientPaths.getPartsOriGuidTags Decoders.getstringArray
        patchPartsOriGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsOriGuidAddTags Encoders.setstringArray
        patchPartsOriGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsOriGuidRemoveTags Encoders.setstringArray
        getPartsLinkerGuidAPI = get1Handler guidToString clientPaths.getPartsLinkerGuid Decoders.getPart
        getPartsLinkerGuidTagsAPI = get1Handler guidToString clientPaths.getPartsLinkerGuidTags Decoders.getstringArray
        patchPartsLinkerGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsLinkerGuidAddTags Encoders.setstringArray
        patchPartsLinkerGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsLinkerGuidRemoveTags Encoders.setstringArray
        getPartsRestrictionsiteGuidAPI = get1Handler guidToString clientPaths.getPartsRestrictionsiteGuid Decoders.getPart
        getPartsRestrictionsiteGuidTagsAPI = get1Handler guidToString clientPaths.getPartsRestrictionsiteGuidTags Decoders.getstringArray
        patchPartsRestrictionsiteGuidAddTagsAPI = post1Handler guidToString clientPaths.patchPartsRestrictionsiteGuidAddTags Encoders.setstringArray
        patchPartsRestrictionsiteGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchPartsRestrictionsiteGuidRemoveTags Encoders.setstringArray
        getReagentsDnaGuidAPI = get1Handler guidToString clientPaths.getReagentsDnaGuid Decoders.getReagent
        postReagentsDnaGuidAPI = post1Handler guidToString clientPaths.postReagentsDnaGuid Encoders.setReagent
        patchReagentsDnaGuidAPI = post1Handler guidToString clientPaths.patchReagentsDnaGuid Encoders.setReagent
        patchReagentsDnaGuidAddTagsAPI = post1Handler guidToString clientPaths.patchReagentsDnaGuidAddTags Encoders.setstringArray
        getReagentsDnaGuidTagsAPI = get1Handler guidToString clientPaths.getReagentsDnaGuidTags Decoders.getstringArray
        patchReagentsDnaGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchReagentsDnaGuidRemoveTags Encoders.setstringArray
        getReagentsDnaGuidFileRefsAPI = get1Handler guidToString clientPaths.getReagentsDnaGuidFileRefs Decoders.getFileRefArray
        getReagentsRnaGuidAPI = get1Handler guidToString clientPaths.getReagentsRnaGuid Decoders.getReagent
        postReagentsRnaGuidAPI = post1Handler guidToString clientPaths.postReagentsRnaGuid Encoders.setReagent
        patchReagentsRnaGuidAPI = post1Handler guidToString clientPaths.patchReagentsRnaGuid Encoders.setReagent
        getReagentsRnaGuidTagsAPI = get1Handler guidToString clientPaths.getReagentsRnaGuidTags Decoders.getstringArray
        patchReagentsRnaGuidAddTagsAPI = post1Handler guidToString clientPaths.patchReagentsRnaGuidAddTags Encoders.setstringArray
        patchReagentsRnaGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchReagentsRnaGuidRemoveTags Encoders.setstringArray
        getReagentsRnaGuidFileRefsAPI = get1Handler guidToString clientPaths.getReagentsRnaGuidFileRefs Decoders.getFileRefArray
        getReagentsChemicalGuidAPI = get1Handler guidToString clientPaths.getReagentsChemicalGuid Decoders.getReagent
        postReagentsChemicalGuidAPI = post1Handler guidToString clientPaths.postReagentsChemicalGuid Encoders.setReagent
        patchReagentsChemicalGuidAPI = post1Handler guidToString clientPaths.patchReagentsChemicalGuid Encoders.setReagent
        getReagentsChemicalGuidTagsAPI = get1Handler guidToString clientPaths.getReagentsChemicalGuidTags Decoders.getstringArray
        patchReagentsChemicalGuidAddTagsAPI = post1Handler guidToString clientPaths.patchReagentsChemicalGuidAddTags Encoders.setstringArray
        patchReagentsChemicalGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchReagentsChemicalGuidRemoveTags Encoders.setstringArray
        getReagentsChemicalGuidFileRefsAPI = get1Handler guidToString clientPaths.getReagentsChemicalGuidFileRefs Decoders.getFileRefArray
        getReagentsProteinGuidAPI = get1Handler guidToString clientPaths.getReagentsProteinGuid Decoders.getReagent
        postReagentsProteinGuidAPI = post1Handler guidToString clientPaths.postReagentsProteinGuid Encoders.setReagent
        patchReagentsProteinGuidAPI = post1Handler guidToString clientPaths.patchReagentsProteinGuid Encoders.setReagent
        getReagentsProteinGuidTagsAPI = get1Handler guidToString clientPaths.getReagentsProteinGuidTags Decoders.getstringArray
        patchReagentsProteinGuidAddTagsAPI = post1Handler guidToString clientPaths.patchReagentsProteinGuidAddTags Encoders.setstringArray
        patchReagentsProteinGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchReagentsProteinGuidRemoveTags Encoders.setstringArray
        getReagentsProteinGuidFileRefsAPI = get1Handler guidToString clientPaths.getReagentsProteinGuidFileRefs Decoders.getFileRefArray
        getReagentsGenericentityGuidAPI = get1Handler guidToString clientPaths.getReagentsGenericentityGuid Decoders.getReagent
        postReagentsGenericentityGuidAPI = post1Handler guidToString clientPaths.postReagentsGenericentityGuid Encoders.setReagent
        patchReagentsGenericentityGuidAPI = post1Handler guidToString clientPaths.patchReagentsGenericentityGuid Encoders.setReagent
        getReagentsGenericentityGuidTagsAPI = get1Handler guidToString clientPaths.getReagentsGenericentityGuidTags Decoders.getstringArray
        patchReagentsGenericentityGuidAddTagsAPI = post1Handler guidToString clientPaths.patchReagentsGenericentityGuidAddTags Encoders.setstringArray
        patchReagentsGenericentityGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchReagentsGenericentityGuidRemoveTags Encoders.setstringArray
        getReagentsGenericentityGuidFileRefsAPI = get1Handler guidToString clientPaths.getReagentsGenericentityGuidFileRefs Decoders.getFileRefArray
        getCellsGuidAPI = get1Handler guidToString clientPaths.getCellsGuid Decoders.getCell
        postCellsGuidAPI = post1Handler guidToString clientPaths.postCellsGuid Encoders.setCell
        patchCellsGuidAPI = post1Handler guidToString clientPaths.patchCellsGuid Encoders.setCell
        getCellsGuidEntitiesAPI = get1Handler guidToString clientPaths.getCellsGuidEntities Decoders.getCellEntityArray
        patchCellsGuidAddEntitiesAPI = post1Handler guidToString clientPaths.patchCellsGuidAddEntities Encoders.setCellEntityArray
        patchCellsGuidRemoveEntitiesAPI = post1Handler guidToString clientPaths.patchCellsGuidRemoveEntities Encoders.setCellEntityArray
        getCellsGuidTagsAPI = get1Handler guidToString clientPaths.getCellsGuidTags Decoders.getstringArray
        patchCellsGuidAddTagsAPI = post1Handler guidToString clientPaths.patchCellsGuidAddTags Encoders.setstringArray
        patchCellsGuidRemoveTagsAPI = post1Handler guidToString clientPaths.patchCellsGuidRemoveTags Encoders.setstringArray
        getSamplesSampleIdAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleId Decoders.getSample
        postSamplesSampleIdAPI = post1Handler SampleId.toString clientPaths.postSamplesSampleId Encoders.setSample
        getSamplesSampleIdDevicesAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleIdDevices Decoders.getSampleDeviceArray
        getSamplesSampleIdDeviceCellIdAPI = get2Handler SampleId.toString CellId.toString clientPaths.getSamplesSampleIdDeviceCellId Decoders.getSampleDevice
        patchSamplesSampleIdAddDevicesAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdAddDevices Encoders.setSampleDeviceArray
        patchSamplesSampleIdRemoveDevicesAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdRemoveDevices Encoders.setSampleDeviceArray
        getSamplesSampleIdTimeseriesAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleIdTimeseries Decoders.getstring
        getSamplesSampleIdConditionsAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleIdConditions Decoders.getConditionArray
        patchSamplesSampleIdAddConditionsAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdAddConditions Encoders.setConditionArray
        patchSamplesSampleIdRemoveConditionsAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdRemoveConditions Encoders.setConditionArray
        getSamplesSampleIdObservationsAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleIdObservations Decoders.getObservationArray
        patchSamplesSampleIdAddObservationsAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdAddObservations Encoders.setObservationArray
        getSamplesSampleIdSignalIdObservationsAPI = get2Handler SampleId.toString SignalId.toString clientPaths.getSamplesSampleIdSignalIdObservations Decoders.getObservationArray
        getSamplesSampleIdTagsAPI = get1Handler SampleId.toString clientPaths.getSamplesSampleIdTags Decoders.getstringArray
        patchSamplesSampleIdAddTagsAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdAddTags Encoders.setstringArray
        patchSamplesSampleIdRemoveTagsAPI = post1Handler SampleId.toString clientPaths.patchSamplesSampleIdRemoveTags Encoders.setstringArray
        getExperimentsExperimentIdAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentId Decoders.getExperiment
        postExperimentsExperimentIdAPI = post1Handler ExperimentId.toString clientPaths.postExperimentsExperimentId Encoders.setExperiment
        patchExperimentsExperimentIdAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentId Encoders.setExperiment
        getExperimentsExperimentIdSamplesAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentIdSamples Decoders.getSampleArray
        postExperimentsExperimentIdSamplesAPI = post1Handler ExperimentId.toString clientPaths.postExperimentsExperimentIdSamples Encoders.setSampleArray
        getExperimentsExperimentIdOperationsAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentIdOperations Decoders.getExperimentOperationArray
        patchExperimentsExperimentIdAddOperationAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdAddOperation Encoders.setExperimentOperation
        patchExperimentsExperimentIdRemoveOperationAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdRemoveOperation Encoders.setExperimentOperation
        getExperimentsExperimentIdSignalsAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentIdSignals Decoders.getSignalArray
        patchExperimentsExperimentIdAddSignalsAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdAddSignals Encoders.setSignalArray
        patchExperimentsExperimentIdRemoveSignalsAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdRemoveSignals Encoders.setSignalArray
        getExperimentsExperimentIdObservationsAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentIdObservations Decoders.getObservationArray
        getExperimentsExperimentIdTagsAPI = get1Handler ExperimentId.toString clientPaths.getExperimentsExperimentIdTags Decoders.getstringArray
        patchExperimentsExperimentIdAddTagsAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdAddTags Encoders.setstringArray
        patchExperimentsExperimentIdRemoveTagsAPI = post1Handler ExperimentId.toString clientPaths.patchExperimentsExperimentIdRemoveTags Encoders.setstringArray
        getObservationsObservationIdAPI = get1Handler observationIdToString clientPaths.getObservationsObservationId Decoders.getObservation
        postObservationsObservationIdAPI = post1Handler observationIdToString clientPaths.postObservationsObservationId Encoders.setObservation
        getSignalsSignalIdAPI = get1Handler SignalId.toString clientPaths.getSignalsSignalId Decoders.getSignalArray
        getTagsGuidAPI = get1Handler guidToString clientPaths.getTagsGuid Decoders.getstringArray
        patchFileRefsGuidLinkAPI = post1Handler guidToString clientPaths.patchFileRefsGuidLink Encoders.setFileRefArray
        patchFileRefsGuidUnlinkAPI = post1Handler guidToString clientPaths.patchFileRefsGuidUnlink Encoders.setFileRefArray
        getFilesFileIdAPI = get1Handler FileId.toString clientPaths.getFilesFileId Decoders.getstring
        postFilesFileIdAPI = post1Handler FileId.toString clientPaths.postFilesFileId Encoders.setstring

}
