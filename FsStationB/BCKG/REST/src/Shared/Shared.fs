//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Shared.Shared

open System
open BCKG.Domain

let guidFromString (x:string) : Option<Guid> = 
    try 
        Some (System.Guid.Parse x)
    with 
        | _  -> None

let guidToString (g:Guid) : string = g.ToString()

let observationIdFromString (x:string) : Option<ObservationId> = failwith "Unimplemented conversion from string to ObservationId"

let observationIdToString (o:ObservationId) : string = failwith "Unimplemented conversion from string to ObservationId"

let partIdFromString (x:string) : Option<PartId> = failwith "Unimplemented conversion from string to PartId"

let partIdToString (p:PartId) : string = failwith "Unimplemented conversion from string to PartId"

let reagentIdFromString (x:string) : Option<ReagentId> = failwith "Unimplemented conversion from string to ReagentId"

let reagentIdToString (r:ReagentId) : string = failwith "Unimplemented conversion from string to ReagentId"

type IBCKGApi = {
    //postPartsGuidAPI : guid -> body -> Async<unit>
    postPartsGuidAPI : Guid -> Part -> Async<unit>
    //patchPartsGuidAPI : guid -> body -> Async<unit>
    patchPartsGuidAPI : Guid -> Part -> Async<unit>
    //getPartsPromoterGuidAPI : guid -> Async<Part>
    getPartsPromoterGuidAPI : Guid -> Async<Part>
    //getPartsPromoterGuidTagsAPI : guid -> Async<string[]>
    getPartsPromoterGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsPromoterGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsPromoterGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsPromoterGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsPromoterGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsRbsGuidAPI : guid -> Async<Part>
    getPartsRbsGuidAPI : Guid -> Async<Part>
    //getPartsRbsGuidTagsAPI : guid -> Async<string[]>
    getPartsRbsGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsRbsGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsRbsGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsRbsGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsRbsGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsCdsGuidAPI : guid -> Async<Part>
    getPartsCdsGuidAPI : Guid -> Async<Part>
    //getPartsCdsGuidTagsAPI : guid -> Async<string[]>
    getPartsCdsGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsCdsGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsCdsGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsCdsGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsCdsGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsTerminatorGuidAPI : guid -> Async<Part>
    getPartsTerminatorGuidAPI : Guid -> Async<Part>
    //getPartsTerminatorGuidTagsAPI : guid -> Async<string[]>
    getPartsTerminatorGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsTerminatorGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsTerminatorGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsTerminatorGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsTerminatorGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsUserdefinedGuidAPI : guid -> Async<Part>
    getPartsUserdefinedGuidAPI : Guid -> Async<Part>
    //getPartsUserdefinedGuidTagsAPI : guid -> Async<string[]>
    getPartsUserdefinedGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsUserdefinedGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsUserdefinedGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsUserdefinedGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsUserdefinedGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsScarGuidAPI : guid -> Async<Part>
    getPartsScarGuidAPI : Guid -> Async<Part>
    //getPartsScarGuidTagsAPI : guid -> Async<string[]>
    getPartsScarGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsScarGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsScarGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsScarGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsScarGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsBackboneGuidAPI : guid -> Async<Part>
    getPartsBackboneGuidAPI : Guid -> Async<Part>
    //getPartsBackboneGuidTagsAPI : guid -> Async<string[]>
    getPartsBackboneGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsBackboneGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsBackboneGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsBackboneGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsBackboneGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsOriGuidAPI : guid -> Async<Part>
    getPartsOriGuidAPI : Guid -> Async<Part>
    //getPartsOriGuidTagsAPI : guid -> Async<string[]>
    getPartsOriGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsOriGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsOriGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsOriGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsOriGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsLinkerGuidAPI : guid -> Async<Part>
    getPartsLinkerGuidAPI : Guid -> Async<Part>
    //getPartsLinkerGuidTagsAPI : guid -> Async<string[]>
    getPartsLinkerGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsLinkerGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsLinkerGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsLinkerGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsLinkerGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getPartsRestrictionsiteGuidAPI : guid -> Async<Part>
    getPartsRestrictionsiteGuidAPI : Guid -> Async<Part>
    //getPartsRestrictionsiteGuidTagsAPI : guid -> Async<string[]>
    getPartsRestrictionsiteGuidTagsAPI : Guid -> Async<string[]>
    //patchPartsRestrictionsiteGuidAddTagsAPI : guid -> body -> Async<unit>
    patchPartsRestrictionsiteGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchPartsRestrictionsiteGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchPartsRestrictionsiteGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsDnaGuidAPI : guid -> Async<Reagent>
    getReagentsDnaGuidAPI : Guid -> Async<Reagent>
    //postReagentsDnaGuidAPI : guid -> body -> Async<unit>
    postReagentsDnaGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsDnaGuidAPI : guid -> body -> Async<unit>
    patchReagentsDnaGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsDnaGuidAddTagsAPI : guid -> body -> Async<unit>
    patchReagentsDnaGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsDnaGuidTagsAPI : guid -> Async<string[]>
    getReagentsDnaGuidTagsAPI : Guid -> Async<string[]>
    //patchReagentsDnaGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchReagentsDnaGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsDnaGuidFileRefsAPI : guid -> Async<FileRef[]>
    getReagentsDnaGuidFileRefsAPI : Guid -> Async<FileRef[]>
    //getReagentsRnaGuidAPI : guid -> Async<Reagent>
    getReagentsRnaGuidAPI : Guid -> Async<Reagent>
    //postReagentsRnaGuidAPI : guid -> body -> Async<unit>
    postReagentsRnaGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsRnaGuidAPI : guid -> body -> Async<unit>
    patchReagentsRnaGuidAPI : Guid -> Reagent -> Async<unit>
    //getReagentsRnaGuidTagsAPI : guid -> Async<string[]>
    getReagentsRnaGuidTagsAPI : Guid -> Async<string[]>
    //patchReagentsRnaGuidAddTagsAPI : guid -> body -> Async<unit>
    patchReagentsRnaGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchReagentsRnaGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchReagentsRnaGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsRnaGuidFileRefsAPI : guid -> Async<FileRef[]>
    getReagentsRnaGuidFileRefsAPI : Guid -> Async<FileRef[]>
    //getReagentsChemicalGuidAPI : guid -> Async<Reagent>
    getReagentsChemicalGuidAPI : Guid -> Async<Reagent>
    //postReagentsChemicalGuidAPI : guid -> body -> Async<unit>
    postReagentsChemicalGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsChemicalGuidAPI : guid -> body -> Async<unit>
    patchReagentsChemicalGuidAPI : Guid -> Reagent -> Async<unit>
    //getReagentsChemicalGuidTagsAPI : guid -> Async<string[]>
    getReagentsChemicalGuidTagsAPI : Guid -> Async<string[]>
    //patchReagentsChemicalGuidAddTagsAPI : guid -> body -> Async<unit>
    patchReagentsChemicalGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchReagentsChemicalGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchReagentsChemicalGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsChemicalGuidFileRefsAPI : guid -> Async<FileRef[]>
    getReagentsChemicalGuidFileRefsAPI : Guid -> Async<FileRef[]>
    //getReagentsProteinGuidAPI : guid -> Async<Reagent>
    getReagentsProteinGuidAPI : Guid -> Async<Reagent>
    //postReagentsProteinGuidAPI : guid -> body -> Async<unit>
    postReagentsProteinGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsProteinGuidAPI : guid -> body -> Async<unit>
    patchReagentsProteinGuidAPI : Guid -> Reagent -> Async<unit>
    //getReagentsProteinGuidTagsAPI : guid -> Async<string[]>
    getReagentsProteinGuidTagsAPI : Guid -> Async<string[]>
    //patchReagentsProteinGuidAddTagsAPI : guid -> body -> Async<unit>
    patchReagentsProteinGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchReagentsProteinGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchReagentsProteinGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsProteinGuidFileRefsAPI : guid -> Async<FileRef[]>
    getReagentsProteinGuidFileRefsAPI : Guid -> Async<FileRef[]>
    //getReagentsGenericentityGuidAPI : guid -> Async<Reagent>
    getReagentsGenericentityGuidAPI : Guid -> Async<Reagent>
    //postReagentsGenericentityGuidAPI : guid -> body -> Async<unit>
    postReagentsGenericentityGuidAPI : Guid -> Reagent -> Async<unit>
    //patchReagentsGenericentityGuidAPI : guid -> body -> Async<unit>
    patchReagentsGenericentityGuidAPI : Guid -> Reagent -> Async<unit>
    //getReagentsGenericentityGuidTagsAPI : guid -> Async<string[]>
    getReagentsGenericentityGuidTagsAPI : Guid -> Async<string[]>
    //patchReagentsGenericentityGuidAddTagsAPI : guid -> body -> Async<unit>
    patchReagentsGenericentityGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchReagentsGenericentityGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchReagentsGenericentityGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getReagentsGenericentityGuidFileRefsAPI : guid -> Async<FileRef[]>
    getReagentsGenericentityGuidFileRefsAPI : Guid -> Async<FileRef[]>
    //getCellsGuidAPI : guid -> Async<Cell>
    getCellsGuidAPI : Guid -> Async<Cell>
    //postCellsGuidAPI : guid -> body -> Async<unit>
    postCellsGuidAPI : Guid -> Cell -> Async<unit>
    //patchCellsGuidAPI : guid -> body -> Async<unit>
    patchCellsGuidAPI : Guid -> Cell -> Async<unit>
    //getCellsGuidEntitiesAPI : guid -> Async<CellEntity[]>
    getCellsGuidEntitiesAPI : Guid -> Async<CellEntity[]>
    //patchCellsGuidAddEntitiesAPI : guid -> body -> Async<unit>
    patchCellsGuidAddEntitiesAPI : Guid -> CellEntity[] -> Async<unit>
    //patchCellsGuidRemoveEntitiesAPI : guid -> body -> Async<unit>
    patchCellsGuidRemoveEntitiesAPI : Guid -> CellEntity[] -> Async<unit>
    //getCellsGuidTagsAPI : guid -> Async<string[]>
    getCellsGuidTagsAPI : Guid -> Async<string[]>
    //patchCellsGuidAddTagsAPI : guid -> body -> Async<unit>
    patchCellsGuidAddTagsAPI : Guid -> string[] -> Async<unit>
    //patchCellsGuidRemoveTagsAPI : guid -> body -> Async<unit>
    patchCellsGuidRemoveTagsAPI : Guid -> string[] -> Async<unit>
    //getSamplesSampleIdAPI : sampleId -> Async<Sample>
    getSamplesSampleIdAPI : SampleId -> Async<Sample>
    //postSamplesSampleIdAPI : sampleId -> body -> Async<unit>
    postSamplesSampleIdAPI : SampleId -> Sample -> Async<unit>
    //getSamplesSampleIdDevicesAPI : sampleId -> Async<SampleDevice[]>
    getSamplesSampleIdDevicesAPI : SampleId -> Async<SampleDevice[]>
    //getSamplesSampleIdDeviceCellIdAPI : (sampleId,cellId) -> Async<SampleDevice>
    getSamplesSampleIdDeviceCellIdAPI : SampleId*CellId -> Async<SampleDevice>
    //patchSamplesSampleIdAddDevicesAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdAddDevicesAPI : SampleId -> SampleDevice[] -> Async<unit>
    //patchSamplesSampleIdRemoveDevicesAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdRemoveDevicesAPI : SampleId -> SampleDevice[] -> Async<unit>
    //getSamplesSampleIdTimeseriesAPI : sampleId -> Async<string>
    getSamplesSampleIdTimeseriesAPI : SampleId -> Async<string>
    //getSamplesSampleIdConditionsAPI : sampleId -> Async<Condition[]>
    getSamplesSampleIdConditionsAPI : SampleId -> Async<Condition[]>
    //patchSamplesSampleIdAddConditionsAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdAddConditionsAPI : SampleId -> Condition[] -> Async<unit>
    //patchSamplesSampleIdRemoveConditionsAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdRemoveConditionsAPI : SampleId -> Condition[] -> Async<unit>
    //getSamplesSampleIdObservationsAPI : sampleId -> Async<Observation[]>
    getSamplesSampleIdObservationsAPI : SampleId -> Async<Observation[]>
    //patchSamplesSampleIdAddObservationsAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdAddObservationsAPI : SampleId -> Observation[] -> Async<unit>
    //getSamplesSampleIdSignalIdObservationsAPI : (sampleId,signalId) -> Async<Observation[]>
    getSamplesSampleIdSignalIdObservationsAPI : SampleId*SignalId -> Async<Observation[]>
    //getSamplesSampleIdTagsAPI : sampleId -> Async<string[]>
    getSamplesSampleIdTagsAPI : SampleId -> Async<string[]>
    //patchSamplesSampleIdAddTagsAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdAddTagsAPI : SampleId -> string[] -> Async<unit>
    //patchSamplesSampleIdRemoveTagsAPI : sampleId -> body -> Async<unit>
    patchSamplesSampleIdRemoveTagsAPI : SampleId -> string[] -> Async<unit>
    //getExperimentsExperimentIdAPI : experimentId -> Async<Experiment>
    getExperimentsExperimentIdAPI : ExperimentId -> Async<Experiment>
    //postExperimentsExperimentIdAPI : experimentId -> body -> Async<unit>
    postExperimentsExperimentIdAPI : ExperimentId -> Experiment -> Async<unit>
    //patchExperimentsExperimentIdAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdAPI : ExperimentId -> Experiment -> Async<unit>
    //getExperimentsExperimentIdSamplesAPI : experimentId -> Async<Sample[]>
    getExperimentsExperimentIdSamplesAPI : ExperimentId -> Async<Sample[]>
    //postExperimentsExperimentIdSamplesAPI : experimentId -> body -> Async<unit>
    postExperimentsExperimentIdSamplesAPI : ExperimentId -> Sample[] -> Async<unit>
    //getExperimentsExperimentIdOperationsAPI : experimentId -> Async<ExperimentOperation[]>
    getExperimentsExperimentIdOperationsAPI : ExperimentId -> Async<ExperimentOperation[]>
    //patchExperimentsExperimentIdAddOperationAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdAddOperationAPI : ExperimentId -> ExperimentOperation -> Async<unit>
    //patchExperimentsExperimentIdRemoveOperationAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdRemoveOperationAPI : ExperimentId -> ExperimentOperation -> Async<unit>
    //getExperimentsExperimentIdSignalsAPI : experimentId -> Async<Signal[]>
    getExperimentsExperimentIdSignalsAPI : ExperimentId -> Async<Signal[]>
    //patchExperimentsExperimentIdAddSignalsAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdAddSignalsAPI : ExperimentId -> Signal[] -> Async<unit>
    //patchExperimentsExperimentIdRemoveSignalsAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdRemoveSignalsAPI : ExperimentId -> Signal[] -> Async<unit>
    //getExperimentsExperimentIdObservationsAPI : experimentId -> Async<Observation[]>
    getExperimentsExperimentIdObservationsAPI : ExperimentId -> Async<Observation[]>
    //getExperimentsExperimentIdTagsAPI : experimentId -> Async<string[]>
    getExperimentsExperimentIdTagsAPI : ExperimentId -> Async<string[]>
    //patchExperimentsExperimentIdAddTagsAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdAddTagsAPI : ExperimentId -> string[] -> Async<unit>
    //patchExperimentsExperimentIdRemoveTagsAPI : experimentId -> body -> Async<unit>
    patchExperimentsExperimentIdRemoveTagsAPI : ExperimentId -> string[] -> Async<unit>
    //getObservationsObservationIdAPI : observationId -> Async<Observation>
    getObservationsObservationIdAPI : ObservationId -> Async<Observation>
    //postObservationsObservationIdAPI : observationId -> body -> Async<unit>
    postObservationsObservationIdAPI : ObservationId -> Observation -> Async<unit>
    //getSignalsSignalIdAPI : signalId -> Async<Signal[]>
    getSignalsSignalIdAPI : SignalId -> Async<Signal[]>
    //getTagsGuidAPI : guid -> Async<string[]>
    getTagsGuidAPI : Guid -> Async<string[]>
    //patchFileRefsGuidLinkAPI : guid -> body -> Async<unit>
    patchFileRefsGuidLinkAPI : Guid -> FileRef[] -> Async<unit>
    //patchFileRefsGuidUnlinkAPI : guid -> body -> Async<unit>
    patchFileRefsGuidUnlinkAPI : Guid -> FileRef[] -> Async<unit>
    //getFilesFileIdAPI : fileId -> Async<string>
    getFilesFileIdAPI : FileId -> Async<string>
    //postFilesFileIdAPI : fileId -> body -> Async<unit>
    postFilesFileIdAPI : FileId -> string -> Async<unit>
}
