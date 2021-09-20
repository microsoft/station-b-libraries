module BCKG.API
open BCKG.Events
open BCKG.Domain
open Thoth.Json.Net
open System.IO.Compression



type EventStorageType =
    | TableEvent of Event
    | BlobEvent of (Event* byte [])
            

type private IStorage = 
    //Indexing functions
    abstract member IndexFiles                : unit -> Async<Map<System.Guid, FileRef[]>> 
    abstract member IndexTags                 : unit -> Async<Map<System.Guid, Tag[]>>
    abstract member IndexExperimentOperations : unit -> Async<Map<ExperimentId, ExperimentOperation[]>>
    abstract member IndexExperimentSignals    : unit -> Async<Map<ExperimentId, Signal[]>>
    abstract member IndexSampleConditions     : unit -> Async<Map<SampleId, Condition[]>>
    abstract member IndexSampleCells          : unit -> Async<Map<SampleId, (CellId * (float option * float option))[]>>
    abstract member IndexExperimentSamples    : unit -> Async<Map<ExperimentId, Sample[]>>
    abstract member IndexCellEntities         : unit -> Async<Map<CellId, CellEntity[]>>
    abstract member IndexObservations         : unit -> Async<Map<SampleId*SignalId, Observation[]>>


    //batch getters
    abstract member GetParts:  unit -> Async<Part[]>
    abstract member GetReagents: unit -> Async<Reagent[]>
    abstract member GetExperiments: unit -> Async<Experiment[]>
    abstract member GetSamples: unit -> Async<Sample[]>
    abstract member GetEvents: unit -> Async<Event[]>
    abstract member GetCells: unit -> Async<Cell[]>
    abstract member GetInteractions: unit -> Async<Interaction[]>

    //abstract member GetInteractionsBetween: Entity[] -> Async<Interaction[]>


    //singleton getters
    abstract member TryGetPart:  PartId -> Async<Part option>                
    abstract member GetPartTags:  PartId -> Async<Tag[]>    
    abstract member TryGetInteraction: InteractionId -> Async<Interaction option> 

    abstract member TryGetExperimentEvent: ExperimentOperationId -> Async<ExperimentOperation option>    
    abstract member TryGetExperimentSignal: ExperimentId * SignalId -> Async<Signal option>                     //TODO: Is this needed? Just checks whether it exists?
    abstract member TryGetSampleCellStore: SampleId * CellId -> Async<BCKG.Entities.SampleCellStore option>     //TODO: Is this needed? Just checks whether it exists?
    abstract member TryGetSampleCondition: SampleId * ReagentId  -> Async<Condition option>                 //TODO: Is this needed? Just checks whether it exists?
    abstract member TryGetSampleDevice: SampleId * CellId -> Async<SampleDevice option>
    abstract member GetSampleDevices: SampleId -> Async<SampleDevice[]>
    abstract member TryGetFileRef: FileId -> Async<FileRef option>                                              //TODO: Is this needed? Just checks whether it exists?        

    //abstract member TryGetInteractionProperties:InteractionId -> Async<(InteractionProperties*string) option>     
           
    abstract member TryGetExperiment:        ExperimentId -> Async<Experiment option>
    abstract member GetExperimentSamples:    ExperimentId -> Async<Sample[]>    
    abstract member GetExperimentOperations: ExperimentId -> Async<ExperimentOperation[]>
    abstract member GetExperimentSignals:    ExperimentId -> Async<Signal[]>    
    abstract member GetExperimentFiles:      ExperimentId -> Async<FileRef[]>    
    abstract member GetExperimentTags:       ExperimentId -> Async<Tag[]>

    abstract member TryGetReagent:   ReagentId -> Async<Reagent option>
    abstract member GetReagentTags:  ReagentId -> Async<Tag[]>    
    abstract member GetReagentFiles: ReagentId -> Async<FileRef[]>    

    abstract member TryGetCell:        CellId -> Async<Cell option> 
    abstract member GetCellTags:       CellId -> Async<Tag[]>    
    abstract member GetEntitiesOfCell: CellId -> Async<CellEntity[]>

    abstract member TryGetSample:   SampleId -> Async<Sample option>
    abstract member GetSampleCells: SampleId -> Async<(CellId*(float option * float option))[]>    
    abstract member GetSampleFiles: SampleId -> Async<FileRef[]>    
    abstract member GetSampleConditions: SampleId -> Async<Condition[]>    
    abstract member GetSampleTags:  SampleId -> Async<Tag[]>
    abstract member GetSampleObservations: SampleId -> Async<Observation[]>
    abstract member TryGetObservation: ObservationId -> Async<Observation option>
    abstract member GetObservations: SampleId * SignalId -> Async<Observation[]> 

    abstract member GetSignals: SignalId -> Async<(ExperimentId * Signal)[]>


    abstract member TryGetCellParent:            CellId -> Async<CellId option>
    abstract member TryGetCellSourceExperiment:  CellId  -> Async<ExperimentId option>
    abstract member TryGetDeviceSourceExperiment: DNAId -> Async<ExperimentId option>
    abstract member GetDeviceComponents:       DNAId -> Async<DNAId[]>

    //file access 
    //TODO: Should these be TryGet... -> option?    
    abstract member GetFile: FileId -> Async<string>                
    abstract member GetFileLink: FileId * string -> Async<string>   
    abstract member GetTimeSeriesFile: FileId -> Async<string>
    abstract member TryGetTimeSeries: SampleId -> Async<string option>
    abstract member GetTimeSeriesLink: FileId * string -> Async<string>    
    abstract member GetAllFileLinks: FileId[] -> Async<Map<FileId,string>>
    abstract member GetBlobsZip: string -> Async<byte[]>    //returns a zip file (as bytes) of all blobs in the container (TODO: Admin API?)
        
    abstract member GetLogJson: unit -> Async<string>
    abstract member GetEntityEvents: System.Guid -> Async<TempEventTuple[]> //TODO: return Events rather than strings to enable custom UI elements (need types in Fable clients)
    
    abstract member ListPlateReaderUniformFileBetween: System.DateTime * System.DateTime -> seq<Storage.UniformFile> //TODO: Why not Async?
    abstract member LoadPlateReaderUniformFile: string -> string //TODO: Why not Async?
    
    abstract member SaveEvent: Event -> unit

    //this functions should not be used externally    
    abstract member _TryParseExperimentLayout: ExperimentId -> Async<Sample[] option>
    abstract member _SavePart: Part -> Async<bool>
    abstract member _SaveReagent: Reagent -> Async<bool>
    abstract member _SaveExperiment: Experiment -> Async<bool>
    abstract member _SaveExperimentEvent: ExperimentId * ExperimentOperation -> Async<bool>   
    abstract member _SaveCell:Cell -> Async<bool>
    abstract member _SaveCellEntity:CellEntity -> Async<bool>    
    abstract member _RemoveCellEntity:CellEntity -> Async<bool>
    abstract member _SaveInteractionConstituents:InteractionProperties*string*((InteractionId*InteractionNodeType*System.Guid*string*int)list) -> Async<bool>    
    abstract member _SaveDerivedFrom: DerivedFrom -> Async<bool>
    abstract member _TryGetDerivedFrom : DerivedFrom -> Async<DerivedFrom option>
    abstract member _DeleteDerivedFrom : DerivedFrom -> Async<bool>
    
    abstract member _DeleteExperimentEvent: ExperimentId * ExperimentOperation -> Async<bool>  //TODO: is this allowed? Do we need the event?
    abstract member _SaveExperimentSignal: ExperimentId * Signal -> Async<bool>   
    abstract member _DeleteExperimentSignal: ExperimentId * Signal -> Async<bool>  //TODO: is this allowed? Do we need the signal?
    abstract member _SaveSample: Sample -> Async<bool>    

    abstract member _SaveSampleDevice: SampleId * CellId * (float option) * (float option) -> Async<bool>  //TODO: rethink implmentation
    abstract member _DeleteSampleDeviceStore: BCKG.Entities.SampleCellStore -> Async<bool>  //TODO: is this allowed?    
    abstract member _SaveReplicate: SampleId * ReplicateId -> Async<bool>
    abstract member _UnlinkReplicate: ReplicateId -> Async<bool>
    
    
    abstract member _UploadTimeSeries: FileId * byte[] -> Async<unit>
    abstract member _UploadFile: FileId * byte[] -> Async<unit>
    abstract member _SaveCondition: SampleId * ReagentId * Concentration * (Time option)-> Async<bool> //TODO: Is this needed? 
    abstract member _DeleteCondition: SampleId * ReagentId  -> Async<bool> 
    abstract member _SaveFileRef: System.Guid * FileRef -> Async<bool> // Source * FileRef
    abstract member _DeleteFileRef: System.Guid * FileRef -> Async<bool> // Source * FileRef
    abstract member _SaveTag: System.Guid * Tag -> Async<bool> //Source * Tag
    abstract member _DeleteTag: System.Guid * Tag -> Async<bool> //Source * Tag

    abstract member _SaveObservation: Observation -> Async<bool> 
    

type private CloudStorage (connectionString) = 
    let tryGetExperiment experimentId = Storage.getExperiment connectionString experimentId
    let getFile fileId = Storage.getFile connectionString fileId    
    let getFileRefs guid = Storage.getFileRefs connectionString guid
    let getSamplesForExperiment experimentId = Storage.getSamplesForExperiment connectionString experimentId

    interface IStorage
        with 
        member _.IndexFiles()                = Storage.IndexFiles connectionString
        member _.IndexTags()                 = Storage.IndexTags connectionString
        member _.IndexExperimentOperations() = Storage.IndexExperimentOperations connectionString
        member _.IndexExperimentSignals()    = Storage.IndexExperimentSignals connectionString
        member _.IndexSampleConditions()     = Storage.IndexSampleConditions connectionString
        member _.IndexSampleCells()          = Storage.IndexSampleCells connectionString
        member _.IndexExperimentSamples()    = Storage.IndexExperimentSamples connectionString
        member _.IndexCellEntities()         = Storage.IndexCellEntities connectionString
        member _.IndexObservations()         = Storage.IndexObservations connectionString

        member _.GetParts() = Storage.getAllParts connectionString
        member _.GetReagents() = Storage.getAllReagents connectionString
        member _.GetExperiments() = Storage.getAllExperiments connectionString 
        member _.GetSamples() = Storage.getAllSamples connectionString
        member _.GetInteractions() = Storage.getAllInteractions connectionString

        member _.GetEvents () = Storage.GetEvents connectionString None
        member _.GetCells() = Storage.getAllCells connectionString
        
        member _.GetPartTags        (partId:PartId)                = Storage.getTags connectionString (partId.guid)
        member _.GetExperimentTags  (ExperimentId experimentGuid)  = Storage.getTags connectionString experimentGuid
        member _.GetReagentTags     (reagentId:ReagentId)          = Storage.getTags connectionString (reagentId.guid)
        member _.GetCellTags        (CellId cellGuid)              = Storage.getTags connectionString cellGuid
        member _.GetSampleTags      (SampleId sampleGuid)          = Storage.getTags connectionString sampleGuid
        
        member _.GetExperimentFiles (ExperimentId experimentGuid)  = Storage.getFileRefs connectionString experimentGuid
        member _.GetReagentFiles    (reagentId:ReagentId)          = Storage.getFileRefs connectionString (reagentId.guid)
        member _.GetSampleFiles     (SampleId sampleGuid)          = Storage.getFileRefs connectionString sampleGuid
        member _.GetSampleCells      sampleId                      = Storage.getSampleCells connectionString sampleId
        member _.GetSampleConditions sampleId                      = Storage.getSampleConditions connectionString sampleId
        
        member _.GetSignals (signalId: SignalId)                   = Storage.getSignals connectionString signalId

        //member _.GetInteractionsBetween entities = failwith "TODO"
        //member _.GetFileRefs guid = Storage.getFileRefs connectionString guid
        member _.GetExperimentOperations experientId = Storage.getExperimentOperations connectionString experientId
        member _.GetExperimentSignals experientId = Storage.getExperimentSignals connectionString experientId
        
        member _.GetEntitiesOfCell cellId = Storage.getEntitiesOfCell connectionString cellId
        member _.TryGetExperiment experimentId = tryGetExperiment experimentId
        member _.TryGetReagent reagentId = Storage.getReagent connectionString reagentId
        member _.TryGetPart partId = Storage.getPart connectionString partId
        member _.TryGetSample sampleId = Storage.getSample connectionString sampleId        
        member _.TryGetCell cellId = Storage.getCell connectionString cellId
        //member _.TryGetInteractionProperties interactionId = Storage.getInteractionProperties connectionString interactionId
        member _.TryGetInteraction interactionId = Storage.getInteraction connectionString interactionId
        member _._SaveDerivedFrom derivedFrom = Storage.saveDerivedFrom connectionString derivedFrom
        member _._TryGetDerivedFrom derivedFrom = Storage.getDerivedFrom connectionString derivedFrom
        member _._DeleteDerivedFrom derivedFrom = Storage.deleteDerivedFrom connectionString derivedFrom
        member _.TryGetCellParent             cellId    = Storage.tryGetCellParent connectionString cellId
        member _.TryGetCellSourceExperiment   cellId    = Storage.tryGetCellSourceExperiment connectionString cellId
        member _.TryGetDeviceSourceExperiment deviceId  = Storage.tryGetDeviceSourceExperiment connectionString deviceId
        member _.GetDeviceComponents          deviceId  = Storage.getDeviceComponents connectionString deviceId
        
        member _.GetExperimentSamples experimentId = getSamplesForExperiment experimentId
        member _.GetFile fileId = getFile fileId
        member _.GetFileLink (fileId, name) = Storage.generateFileDownloadLink connectionString fileId name
        member _.GetTimeSeriesFile fileId = Storage.getTimeSeriesFile connectionString fileId
        member _.TryGetTimeSeries sampleId = Storage.getTimeSeries connectionString sampleId
        member _.GetTimeSeriesLink (fileId, name)  = Storage.generateTimeSeriesFileDownloadLink connectionString fileId name
        member _.GetAllFileLinks fileIds = Storage.generateAllDownloadLinks  connectionString fileIds
        member _.GetBlobsZip clobContainerId = Storage.dowloadBlobs connectionString clobContainerId    
        member _.GetLogJson () = Storage.GetEventsLog connectionString    
        member _.GetEntityEvents entityGuid = 
            async {
                let! events = Storage.GetEvents connectionString (Some entityGuid)
                let result = events |> Array.map Event.ToUiString
                return result
                }
        member _.ListPlateReaderUniformFileBetween (from, until) = Storage.ListPlateReaderUniformFileBetween connectionString from until
        member _.LoadPlateReaderUniformFile path = Storage.LoadPlateReaderUniformFile connectionString path
        member _._TryParseExperimentLayout experimentId = Storage.DataProcessor.TryParseExperimentLayout tryGetExperiment getFileRefs getFile getSamplesForExperiment experimentId
        member _._SavePart part = Storage.savePart connectionString part
        member _._SaveReagent reagent = Storage.saveReagent connectionString reagent
        member _._SaveExperiment experiment = Storage.saveExperiment connectionString experiment
        member _._SaveExperimentEvent (experimentId, event) = Storage.storeExperimentEvent connectionString experimentId event
        member _._SaveCell cell = Storage.saveCell connectionString cell
        member _._SaveCellEntity cellentity = Storage.saveCellEntity connectionString cellentity
        member _._RemoveCellEntity cellentity = Storage.deleteCellEntity connectionString cellentity
        member _._SaveInteractionConstituents(iprops,iType,entities) = Storage.saveInteractionConstituents connectionString iprops iType entities
        member _._DeleteExperimentEvent (experimentId, event) = Storage.deleteExperimentEvent connectionString experimentId event
        member _._SaveExperimentSignal (experimentId, signal) = Storage.storeExperimentSignal connectionString experimentId signal
        member _._DeleteExperimentSignal (experimentId, signal) = Storage.deleteExperimentSignal connectionString experimentId signal
        member _._SaveSample sample = Storage.saveSample connectionString sample
        member _._SaveSampleDevice (sampleId, cellId, cellDensity, cellPreSeeding) = Storage.saveSampleDevice connectionString sampleId cellId cellDensity cellPreSeeding
        member _._DeleteSampleDeviceStore store = Storage.deletesampleDeviceStore connectionString store    
        member _._UploadTimeSeries (fileId, content) = Storage.safeUploadFileBytesToTimeSeriesContainer connectionString fileId content
        member _._UploadFile (fileId, content) = Storage.uploadFileBytesToFilesContainer connectionString fileId content
        member _._SaveCondition (sampleId, reagentId, value, time) = Storage.saveSampleCondition connectionString sampleId reagentId value time
        member _._DeleteCondition (sampleId, reagentId) = Storage.deletesampleConditionStore connectionString sampleId reagentId
        member _._SaveFileRef (sourceGuid, fileRef) = Storage.storeFileRef connectionString sourceGuid fileRef
        member _._DeleteFileRef (sourceGuid, fileRef) =Storage.deleteFileRef connectionString sourceGuid fileRef
        member _._SaveTag (sourceGuid,tag) = Storage.storeTag connectionString sourceGuid tag
        member _._DeleteTag (sourceGuid,tag) = Storage.deleteTag connectionString sourceGuid tag
        member _.SaveEvent event = Storage.saveEvent connectionString event    
        member _.TryGetExperimentEvent experiemntEventId = Storage.getExperimentEvent connectionString experiemntEventId
        member _.TryGetExperimentSignal (experimentId, signalId) = Storage.getExperimentSignal connectionString experimentId signalId
        member _.TryGetSampleCellStore (sampleId, cellId) = Storage.getSampleCellStore connectionString sampleId cellId
        member _.GetSampleDevices sampleId = Storage.getSampleDevices connectionString sampleId
        member _.TryGetSampleDevice (sampleId, cellId) = Storage.getSampleDevice connectionString sampleId cellId
        member _.TryGetSampleCondition (sampleId, reagentId) = Storage.getSampleCondition connectionString sampleId reagentId
        member _.TryGetFileRef fileId = Storage.getFileRef connectionString fileId
        member _._SaveObservation observation = Storage.saveObservation connectionString observation
        member _._SaveReplicate (sampleId,replicateId) = Storage.saveReplicate connectionString sampleId replicateId
        member _._UnlinkReplicate replicateId = Storage.unlinkReplicate connectionString replicateId
        
        member _.GetSampleObservations (sampleId) = Storage.getSampleObservations connectionString sampleId
        member _.TryGetObservation (observationId) = Storage.tryGetObservation connectionString observationId
        member _.GetObservations (sampleId, signalId) = Storage.getObservations connectionString sampleId signalId

let parseOrFail f x = match f x with | Some y -> y | None -> failwithf "Could not parse %s" x
type private MemoryStorage = 
    //Entities
    val mutable partsMap             : Map<System.Guid, Part> 
    val mutable experimentsMap       : Map<ExperimentId, Experiment>
    val mutable reagentsMap          : Map<System.Guid, Reagent>       
    val mutable samplesMap           : Map<SampleId, Sample>    
    val mutable signalsMap           : Map<SignalId, Signal>
    val mutable operationsMap        : Map<ExperimentOperationId, ExperimentOperation>    
    val mutable conditionsMap        : Map<SampleId * ReagentId, Condition>
    val mutable cellsMap             : Map<CellId, Cell> 
    val mutable sampleDevicesMap     : Map<SampleId, SampleDevice[]>
    val mutable sampleObservationsMap : Map<SampleId, Observation[]>
    val mutable observationsMap      : Map<ObservationId, Observation>

    //Files
    val mutable files                : Map<System.Guid, Set<FileRef>>  
    val mutable fileRefs             : Map<FileId, FileRef>
    val mutable fileContent          : Map<FileId, byte[]>
    val mutable dataContent          : Map<FileId, byte[]> //timeSeries
    
    val mutable tags                 : Map<System.Guid, Set<Tag>>
    val mutable cellEntities         : Map<CellId, Set<CellEntity>>

    //Relations
    val mutable experimentOperations : Map<ExperimentId, Set<ExperimentOperationId>>
    val mutable experimentSignals    : Map<ExperimentId, Set<SignalId>>
    val mutable sampleToCells        : Map<SampleId, Map<CellId, (float option * float option)>> //SampleID -> (CellID -> Cell Density * PreSeeding)
    val mutable experimentToSamples  : Map<ExperimentId, Set<SampleId>>
    val mutable replicatesMap           : Map<ReplicateId, SampleId>

    //Properties
    val mutable derivedFroms         : Set<DerivedFrom>
    val mutable interactions         : Map<InteractionId,Interaction>

    //Events log
    val mutable events               : Event list

    val mutable observations : Map<(SampleId*SignalId), Observation[]>

    new () = 
        {   partsMap = Map.empty
            experimentsMap = Map.empty
            reagentsMap = Map.empty                    
            signalsMap = Map.empty
            samplesMap = Map.empty
            cellsMap = Map.empty
            sampleDevicesMap = Map.empty
            observationsMap = Map.empty
            sampleObservationsMap = Map.empty

            derivedFroms = Set.empty
            interactions = Map.empty
             
            operationsMap = Map.empty
            conditionsMap = Map.empty
            experimentOperations = Map.empty
            experimentSignals = Map.empty
            sampleToCells = Map.empty
            experimentToSamples = Map.empty
            replicatesMap = Map.empty
            files = Map.empty
            fileContent = Map.empty
            dataContent = Map.empty
            events = List.empty
            tags = Map.empty
            cellEntities = Map.empty
            fileRefs = Map.empty
            observations = Map.empty
        }                

    member private this.experiments = this.experimentsMap |> Map.toArray |> Array.map snd
           
    member private this.parts = this.partsMap |> Map.toArray |> Array.map snd
           
    member private this.reagents = this.reagentsMap |> Map.toArray |> Array.map snd
      
    member private this.samples = this.samplesMap |> Map.toArray |> Array.map snd

    member private this.cells = this.cellsMap |> Map.toArray |> Array.map snd

    member private this._GetTimeSeriesFile fileId = async {return (System.Text.Encoding.ASCII.GetString this.dataContent.[fileId])}
    
    interface IStorage
        with 
        member this.IndexFiles()                = async { return (Map.map (fun _ x -> Set.toArray x) this.files) }    
        member this.IndexTags()                = async { return (Map.map (fun _ x -> Set.toArray x) this.tags) }    
        member this.IndexExperimentOperations() = async { return (Map.map (fun _ x -> Set.toArray x |> Array.map(fun y -> this.operationsMap.[y])) this.experimentOperations) }    
        member this.IndexExperimentSignals()    = async { return (Map.map (fun _ x -> Set.toArray x |> Array.map(fun y -> this.signalsMap.[y])) this.experimentSignals) }    
        member this.IndexSampleConditions()     = 
            async { 
                let index = 
                    this.conditionsMap
                    |> Map.toArray
                    |> Array.map(fun ((sampleId, reagentId), condition) -> 
                        sampleId, condition
                        )
                    |> Array.groupBy fst
                    |> Array.map(fun (key, L) -> key, Array.map snd L)
                    |> Map.ofArray
                return index                
                }    
        member this.IndexSampleCells()          = async { return (Map.map (fun _ x -> Map.toArray x) this.sampleToCells) }    
        member this.IndexExperimentSamples()    = async { return (Map.map (fun _ x -> Set.toArray x |> Array.map ( fun y -> this.samplesMap.[y])) this.experimentToSamples) }    
        member this.IndexCellEntities()         = async { return (Map.map (fun _ x -> Set.toArray x) this.cellEntities) }   
        member this.IndexObservations()         = async { return this.observations}

        member this.GetParts() = async {return this.parts}
        member this.GetReagents() = async {return this.reagents}
        member this.GetExperiments() = async {return this.experiments}
        member this.GetSamples() = async {return this.samples}
        member this.GetEvents() = async {return Array.ofList this.events}
        member this.GetCells()   = async {return this.cells}
        member _.GetInteractions() = failwith "TODO"

        member _.GetSignals                         _  = failwith "TODO"
        member _.TryGetCellParent                   _  = failwith "TODO"
        member _.TryGetCellSourceExperiment         _  = failwith "TODO"
        member _.TryGetDeviceSourceExperiment       _  = failwith "TODO"
        member _.GetDeviceComponents                _  = failwith "TODO"

        member this.TryGetObservation (observationId) =  async {return Map.tryFind observationId this.observationsMap}
        member this.GetObservations (sampleId, signalId) = async {return (let key = (sampleId, signalId) in if this.observations.ContainsKey key then this.observations.[key] else Array.empty)}
        member this.GetSampleObservations (sampleId) = async {return (let key = (sampleId) in if this.sampleObservationsMap.ContainsKey key then this.sampleObservationsMap.[key] else Array.empty)}

        member this.GetSampleDevices (sampleId) = async {return (let key = sampleId in if this.sampleDevicesMap.ContainsKey key then this.sampleDevicesMap.[key] else Array.empty)}

        member this._SaveCellEntity cellEntity = 
            async {
                if this.cellEntities.ContainsKey cellEntity.cellId then 
                    printfn "WARNING: Modifying cell entity %A" cellEntity.cellId
                let currentEntities = 
                    if this.cellEntities.ContainsKey cellEntity.cellId then
                        this.cellEntities.[cellEntity.cellId]
                    else
                        Set.empty
                this.cellEntities <- this.cellEntities.Add(cellEntity.cellId, currentEntities.Add cellEntity)
                return true
                }  
        
        member this._RemoveCellEntity cellEntity = failwith "Not implemented yet."

        member this.GetEntitiesOfCell cellId = async {if this.cellEntities.ContainsKey(cellId) then return Set.toArray this.cellEntities.[cellId] else return [||]}
        
        member this._SaveReplicate (sampleId,replicateId) = 
            async {
                if this.replicatesMap.ContainsKey replicateId then 
                    printfn "WARNING: Replacing replicate %s" (replicateId.ToString())
                    this.replicatesMap <- this.replicatesMap.Add(replicateId, sampleId)
                return true
                }
        
        member this._UnlinkReplicate (replicateId) = 
            async {
                if this.replicatesMap.ContainsKey replicateId then 
                    this.replicatesMap <- this.replicatesMap.Remove(replicateId)
                return true
                }

        member this._SaveDerivedFrom derivedFrom = 
            async{
                if not (this.derivedFroms.Contains derivedFrom) then 
                    this.derivedFroms <- this.derivedFroms.Add(derivedFrom)
                else printfn "WARNING: Derived From property already exists."
                return true
            }
        
        member this._SaveCell cell = 
            async {
                if this.cellsMap.ContainsKey cell.id then 
                    printfn "WARNING: Replacing cell %A" cell.id
                this.cellsMap <- this.cellsMap.Add(cell.id, cell)
                return true
                }    
        
        member this._SaveTag(guid,tag) =
            async {
                let currentTags = 
                    if this.tags.ContainsKey guid then 
                        this.tags.[guid]
                    else
                        Set.empty
                this.tags <- this.tags.Add(guid, currentTags.Add tag)
                return true
                }    

        member this._DeleteTag(guid,tag) = 
            async {
                if this.tags.ContainsKey guid then 
                    let newTags = this.tags.[guid].Remove tag
                    this.tags <- this.tags.Add(guid, newTags)
                return true
                }    

        member this.GetPartTags (partId:PartId) = async {if this.tags.ContainsKey(partId.guid) then return Set.toArray this.tags.[partId.guid] else return [||]}
        member this.GetReagentTags (reagentId:ReagentId) = async {if this.tags.ContainsKey(reagentId.guid) then return Set.toArray this.tags.[reagentId.guid] else return [||]}
        member this.GetCellTags (CellId cellGuid)  = async {if this.tags.ContainsKey(cellGuid) then return Set.toArray this.tags.[cellGuid] else return [||]}
        member this.GetExperimentTags (ExperimentId experimentGuid) = async {if this.tags.ContainsKey(experimentGuid) then return Set.toArray this.tags.[experimentGuid] else return [||]}
        
        member this.GetExperimentFiles (ExperimentId experimentGuid) = async {if this.files.ContainsKey(experimentGuid) then return Set.toArray this.files.[experimentGuid] else return [||]}
        member this.GetReagentFiles (reagentId:ReagentId) = async {if this.files.ContainsKey(reagentId.guid) then return Set.toArray this.files.[reagentId.guid] else return [||]}
        member this.GetSampleFiles (SampleId sampleGuid) = async {if this.files.ContainsKey(sampleGuid) then return Set.toArray this.files.[sampleGuid] else return [||]}
        
        member this.GetSampleConditions (sampleId:SampleId) = 
            async {
                return this.conditionsMap |> Map.toArray |> Array.map (fun ((sid,rid),condition) -> condition) |> Array.filter (fun s -> s.sampleId = sampleId) }
        
        member this.GetSampleTags (SampleId sampleGuid)  = async {if this.tags.ContainsKey(sampleGuid) then return Set.toArray this.tags.[sampleGuid] else return [||]}        
        member this.GetSampleCells sampleId     = async {if this.sampleToCells.ContainsKey(sampleId) then return (Map.toArray this.sampleToCells.[sampleId]) else return [||]}
        
        member this.TryGetCell cellId   = async {return (if this.cellsMap.ContainsKey cellId then (Some this.cellsMap.[cellId]) else None)}
        
        member this.GetExperimentOperations experimentId = async {if this.experimentOperations.ContainsKey(experimentId) then return (this.experimentOperations.[experimentId] |> Set.toArray |> Array.map(fun opId -> this.operationsMap.[opId])) else return [||]}
        member this.GetExperimentSignals experimentId = async {if this.experimentSignals.ContainsKey(experimentId) then return (this.experimentSignals.[experimentId] |> Set.toArray |> Array.map(fun signalId -> if this.signalsMap.ContainsKey signalId then this.signalsMap.[signalId] else failwithf "Signal not found %A" signalId)) else return [||]}
        
        member this.TryGetTimeSeries (SampleId sampleGuid) = 
            async {
                let files = this.files.[sampleGuid] |> Set.toList 
                if List.isEmpty files then 
                    return None
                else
                    let! data = this._GetTimeSeriesFile (List.head files).fileId
                    return (Some data)
                }

        member this._TryGetDerivedFrom derivedFrom = 
            async{
                if this.derivedFroms.Contains derivedFrom then 
                    return derivedFrom |> Some
                else 
                    return None
            }
        member this._DeleteDerivedFrom derivedFrom = 
            async{
                if this.derivedFroms.Contains derivedFrom then 
                    this.derivedFroms <- this.derivedFroms.Remove(derivedFrom)
                else 
                    printfn "[WARNING] Derived From specified does not exist in memory. No further action will be taken."
                return true
            }
        

        member this.TryGetInteraction _  = failwith "TODO"
        
        member this.TryGetSampleCondition (sampleId, reagentId) = async {return (if this.conditionsMap.ContainsKey(sampleId,reagentId) then (Some this.conditionsMap.[sampleId,reagentId]) else None)}
        member this._TryParseExperimentLayout experimentId = 
            //Storage.DataProcessor.TryParseExperimentLayout tryGetxperiment getFileRefs getFile getSamplesForExperiment experimentId
            failwith "Not currently implemented"

        member this._SaveInteractionConstituents(iprops,iType,entities) = failwith "TODO"
        
        member this.TryGetExperiment experimentId = async {return Map.tryFind experimentId this.experimentsMap}
        member this.TryGetReagent reagentId = async {return Map.tryFind reagentId.guid this.reagentsMap}
        member this.TryGetPart partId = async {return Map.tryFind (partId.ToString() |> System.Guid) this.partsMap}
        member this.TryGetSample sampleId = async {return Map.tryFind sampleId this.samplesMap}
                
        
        member this.GetExperimentSamples experimentId = async {return this.experimentToSamples.[experimentId] |> Set.map (fun sampleId -> this.samplesMap.[sampleId]) |> Set.toArray}        

        member this.GetFile fileId = async {return (System.Text.Encoding.ASCII.GetString this.fileContent.[fileId])}    
        member this.GetTimeSeriesFile fileId = this._GetTimeSeriesFile fileId
        
        member this.TryGetExperimentEvent experimentEventId = 
            async {
                let result = 
                    if this.operationsMap.ContainsKey experimentEventId then
                        Some this.operationsMap.[experimentEventId] 
                    else 
                        None

                return result
            }
    
        member this.TryGetExperimentSignal (experimentId, signalId) = 
            async {
                let result = 
                    if this.experimentSignals.ContainsKey experimentId && this.experimentSignals.[experimentId].Contains signalId then
                        Some this.signalsMap.[signalId]
                    else 
                        None

                return result
            }
    
        member this.TryGetSampleCellStore (sampleId, cellId) = 
            async {
                let key = sampleId
                let result = 
                    if this.sampleToCells.ContainsKey key && this.sampleToCells.[key].ContainsKey cellId then 
                        let cellDensity, cellPreSeeding = this.sampleToCells.[key].[cellId]
                        let store = Entities.SampleCellStore(sampleId, cellId, cellDensity, cellPreSeeding)
                        Some store
                    else
                        None
                return result
            }
        
        member this.TryGetSampleDevice (sampleId, cellId) = 
            async {
                let key = sampleId
                let result = 
                    if this.sampleToCells.ContainsKey key && this.sampleToCells.[key].ContainsKey cellId then 
                        let cellDensity, cellPreSeeding = this.sampleToCells.[key].[cellId]
                        let sampleDevice = {sampleId=sampleId; cellId=cellId; cellDensity=cellDensity; cellPreSeeding=cellPreSeeding}
                        Some sampleDevice
                    else
                        None
                return result
            }

        member this.TryGetFileRef fileId = 
            async {
                let result = 
                    if this.fileRefs.ContainsKey fileId then 
                        Some this.fileRefs.[fileId]
                    else
                        None
                return result
            }
   
        member this.GetLogJson () = async { return (this.events |> List.rev |> List.map Event.encode |> String.concat "," |> sprintf "[%s]")}

        member this.GetEntityEvents entityGuid = 
            async {        
                let result = 
                    this.events 
                    |> List.rev 
                    |> List.filter (fun e -> 
                        let guid = 
                            e.target
                            |> EventTarget.toTargetId
                            |> System.Guid.Parse  
                        guid = entityGuid    
                        )
                    |> Array.ofList
                    |> Array.map Event.ToUiString                                        
                    
                return result
                }

        member this.ListPlateReaderUniformFileBetween (from, until) = failwith "Plate reader data is currently not available in a BCKG memory instance"
        member this.LoadPlateReaderUniformFile x = failwith "Plate reader data is currently not available in a BCKG memory instance"
        member this.GetBlobsZip clobContainerId = failwith "Downloading of blob containers is not supported in a BCKG Memory Instance"
        member this.GetFileLink (fileId, name) = failwith "File download links are not supported in a BCKG Memory Instance"
        member this.GetTimeSeriesLink (fileId, name)  = failwith "File download links are not supported in a BCKG Memory Instance"
        member this.GetAllFileLinks fileIds =  failwith "File download links are not supported in a BCKG Memory Instance"

        //NOTE: linked fields (e.g. operations from experiments) are ignored from the partent entity and stored through separate events
        member this._SavePart part = 
            async {
                if this.partsMap.ContainsKey (part.id.ToString() |> System.Guid) then 
                    printfn "WARNING: Replacing part %A" part.id
                this.partsMap <- this.partsMap.Add(part.guid, part)
                return true
                }    
    
        member this._SaveReagent reagent = 
            async {
                if this.reagentsMap.ContainsKey reagent.id.guid then 
                    printfn "WARNING: Replacing reagent %A" reagent.id
                this.reagentsMap <- this.reagentsMap.Add(reagent.id.guid, reagent)
                return true
                }    
    
        member this._SaveExperiment experiment = 
            async {
                if this.experimentsMap.ContainsKey experiment.id then 
                    printfn "WARNING: Replacing experiment %A" experiment.id
                this.experimentsMap <- this.experimentsMap.Add(experiment.id, experiment)
                return true
                }
    
        member this._SaveExperimentEvent (experimentId, event) = 
            async {
                let operations =
                    if this.experimentOperations.ContainsKey experimentId then 
                        this.experimentOperations.[experimentId]
                    else
                        Set.empty
                this.experimentOperations <- this.experimentOperations.Add(experimentId, operations.Add event.id)
            
                if this.operationsMap.ContainsKey event.id then 
                    printfn "WARNING: Replacing experiment operation %A" event.id
                this.operationsMap <- this.operationsMap.Add(event.id, event)
                return true
                }

        //TODO: is matching by id sufficient for removal? If so, should we change the signature of the method?
        member this._DeleteExperimentEvent (experimentId, event) = 
            async {            
                if this.experimentOperations.ContainsKey experimentId then                 
                    let operations' = this.experimentOperations.[experimentId].Remove event.id
                    this.experimentOperations <- this.experimentOperations.Add(experimentId, operations')
                
                return true 
                }        

        member this._SaveExperimentSignal (experimentId, signal) = 
            async {
                if this.signalsMap.ContainsKey signal.id then 
                    printfn "WARNING: Replacing signal %A" signal.id
                this.signalsMap <- this.signalsMap.Add(signal.id, signal)
                let signals = 
                    if this.experimentSignals.ContainsKey experimentId then 
                        this.experimentSignals.[experimentId]
                    else
                        Set.empty
                this.experimentSignals <- this.experimentSignals.Add(experimentId, signals.Add signal.id)

                return true
            }

        //TODO: is matching by id sufficient for removal? If so, should we change the signature of the method?
        member this._DeleteExperimentSignal (experimentId, signal) =
            async {            
                if this.experimentSignals.ContainsKey experimentId then                 
                    let signals' = this.experimentSignals.[experimentId].Remove signal.id
                    this.experimentSignals <- this.experimentSignals.Add(experimentId, signals')
                       
                return true 
                }        

        member this._SaveSample sample = 
            async {
                if this.samplesMap.ContainsKey sample.id then 
                    printfn "WARNING: Replacing sample %A" sample.id
                this.samplesMap <- this.samplesMap.Add(sample.id, sample)
                return true
            }

        member this._SaveSampleDevice (sampleId, reagentId, cellDensity, cellPreSeeding) = 
            async { 
                let key = sampleId
                let devices = 
                    if this.sampleToCells.ContainsKey key then 
                        this.sampleToCells.[key]
                    else
                        Map.empty
                this.sampleToCells <- this.sampleToCells.Add(key, devices.Add(reagentId, (cellDensity, cellPreSeeding)))
                return true
            }

        member this._DeleteSampleDeviceStore (store:BCKG.Entities.SampleCellStore) =
            async {                 
                let sampleId = parseOrFail SampleId.fromString store.sampleId
                let cellId = parseOrFail CellId.fromString store.cellId
                let key = sampleId            
                if this.sampleToCells.ContainsKey key then 
                    let devices' = this.sampleToCells.[key].Remove cellId
                    this.sampleToCells <- this.sampleToCells.Add(key, devices')                                
            
                return true
            }

        member this.SaveEvent event = this.events <- event::this.events          

        member this._UploadTimeSeries (fileId, content) = 
            async {
                if this.dataContent.ContainsKey fileId then 
                    printfn "WARNING: Replacing data content for %A" fileId
                this.dataContent <- this.dataContent.Add(fileId, content)
                return ()
            }
    
        member this._UploadFile (fileId, content) = 
            async {
                if this.fileContent.ContainsKey fileId then 
                    printfn "WARNING: Replacing file content for %A" fileId
                this.fileContent <- this.fileContent.Add(fileId, content)
                return ()
            }
    
        member this._SaveCondition (sampleId, reagentId, value, time) = 
            async {
                let key = (sampleId, reagentId)
                if this.conditionsMap.ContainsKey key then 
                    printfn "WARNING: Replacing condition %A (%s -> %s)" key (Concentration.toString this.conditionsMap.[key].concentration) (Concentration.toString value)
                this.conditionsMap <- this.conditionsMap.Add(key, Condition.Create(reagentId,sampleId, value, time))
                return true
            }
    
        member this._DeleteCondition (sampleId, reagentId) = 
            async {
                let key = (sampleId, reagentId)
                this.conditionsMap <- this.conditionsMap.Remove key
                return true
            }
    
        member this._SaveFileRef (sourceGuid, fileRef) =         
            async {
                if this.fileRefs.ContainsKey fileRef.fileId then 
                    printfn "WARNING: Replacing file %A's ref" fileRef.fileId
                this.fileRefs <- this.fileRefs.Add(fileRef.fileId, fileRef)


                let currentFiles = 
                    if this.files.ContainsKey sourceGuid then    
                        this.files.[sourceGuid]
                    else
                        Set.empty
                this.files <- this.files.Add(sourceGuid, currentFiles.Add fileRef)
                return true
            }
    
        //TODO: this only workes with a 1 file to 1 source model
        member this._DeleteFileRef (sourceGuid, fileRef) =
            async {                    
                if this.files.ContainsKey sourceGuid then
                    let fileRefs = this.files.[sourceGuid].Remove fileRef 
                    this.files <- this.files.Add(sourceGuid, fileRefs)
                return true
            }            

        member this._SaveObservation observation = 
              async {
                  let key = (observation.sampleId, observation.signalId)
                  let obs = 
                    if this.observations.ContainsKey key then 
                      this.observations.[key]
                    else
                      Array.empty

                  let obs' = Array.append obs [|observation|]
                  this.observations <- this.observations.Add(key, obs')
                  return true
              }


type InstanceType =
    | CloudInstance of string
    | MemoryInstance

module EP = BCKG.Events.EventsProcessor
type Instance = 
    val private storage : IStorage
    val private user : string
    
    new (t:InstanceType, user:string) = 
        let storage = 
            match t with 
            | CloudInstance connectionString -> 
                Storage.initialiseDatabase connectionString |> Async.AwaitTask |> Async.RunSynchronously |> ignore
                CloudStorage(connectionString) :> IStorage
            | MemoryInstance -> MemoryStorage() :>IStorage        
        {storage = storage; user = user}
        
    member this.GetPartTags                        = this.storage.GetPartTags
    member this.GetReagentTags                     = this.storage.GetReagentTags
    member this.GetCellTags                        = this.storage.GetCellTags
    member this.GetSampleTags                      = this.storage.GetSampleTags
    member this.GetExperimentTags                  = this.storage.GetExperimentTags

    member this.GetSampleCells                     = this.storage.GetSampleCells
    member this.GetSampleFiles                     = this.storage.GetSampleFiles
    member this.GetSampleConditions                = this.storage.GetSampleConditions

    member this.TryGetCell                         = this.storage.TryGetCell
    member this.TryGetPart                         = this.storage.TryGetPart
    member this.TryGetReagent                      = this.storage.TryGetReagent
    member this.TryGetExperiment                   = this.storage.TryGetExperiment 
    
    member this.GetSignals                         = this.storage.GetSignals

    member this.GetExperimentOperations            = this.storage.GetExperimentOperations
    member this.GetExperimentSamples               = this.storage.GetExperimentSamples
    member this.GetExperimentSignals               = this.storage.GetExperimentSignals
    member this.GetExperimentFiles                 = this.storage.GetExperimentFiles
    member this.GetReagentFiles                    = this.storage.GetReagentFiles
    member this.GetFileLink                        = this.storage.GetFileLink
    member this.GetTimeSeriesFile                  = this.storage.GetTimeSeriesFile
    member this.TryGetTimeSeries                   = this.storage.TryGetTimeSeries
    member this.GetTimeSeriesLink                  = this.storage.GetTimeSeriesLink
    member this.GetAllFileLinks                    = this.storage.GetAllFileLinks
    member this.GetEntityEvents                    = this.storage.GetEntityEvents

    member this.TryGetCellParent                   = this.storage.TryGetCellParent             
    member this.TryGetCellSourceExperiment         = this.storage.TryGetCellSourceExperiment   
    member this.TryGetDeviceSourceExperiment       = this.storage.TryGetDeviceSourceExperiment 
    member this.GetDeviceComponents                = this.storage.GetDeviceComponents       

    member this.TryGetObservation                  = this.storage.TryGetObservation
    member this.GetObservations                    = this.storage.GetObservations
    member this.GetSampleObservations              = this.storage.GetSampleObservations
    //member this.GetInteractionsBetween             = this.storage.GetInteractionsBetween
    member this.TryGetSample                       = this.storage.TryGetSample
    //member this.TryGetExperimentEvent              = this.storage.TryGetExperimentEvent
    //member this.TryGetExperimentSignal             = this.storage.TryGetExperimentSignal 
    //member this.TryGetSampleCellStore              = this.storage.TryGetSampleCellStore
    member this.TryGetSampleDevice                 = this.storage.TryGetSampleDevice
    member this.GetSampleDevices                   = this.storage.GetSampleDevices
    member this.TryGetSampleCondition              = this.storage.TryGetSampleCondition
    //member this.TryGetFileRef                      = this.storage.TryGetFileRef
    //member this.GetFile                            = this.storage.GetFile       
    //member this.MkFilesIndex                       = this.storage.MkFilesIndex
    

    //Functions used in admin functionality only
    member this.GetBlobsZip                        = this.storage.GetBlobsZip
    member this.GetLogJson                         = this.storage.GetLogJson

    member this.GetParts                           = this.storage.GetParts
    member this.GetReagents                        = this.storage.GetReagents
    member this.GetExperiments                     = this.storage.GetExperiments
    member this.GetSamples                         = this.storage.GetSamples
    member this.GetConditionsOfSample              = this.storage.GetSampleConditions
    member this.GetEvents                          = this.storage.GetEvents
    member this.GetCells                           = this.storage.GetCells
    member this.GetInteractions                    = this.storage.GetInteractions
    
    member this.GetCellEntities                    = this.storage.GetEntitiesOfCell


    member this.ListPlateReaderUniformFileBetween  = this.storage.ListPlateReaderUniformFileBetween
    member this.LoadPlateReaderUniformFile         = this.storage.LoadPlateReaderUniformFile

    member this.ProcessEvent (processEvent:EventStorageType) =         
        match processEvent with 
        | TableEvent (event) -> 
            this.storage.SaveEvent event
            match event.target with 
            | EventTarget.PartEvent pid -> EP.ProcessPartEvent this.storage._SavePart this.storage.TryGetPart pid event.operation event.change        
            | EventTarget.PartTagEvent pid -> EP.ProcessTagEvent this.storage._SaveTag this.storage._DeleteTag (EP.PartTag(pid)) event.operation event.change 

            | EventTarget.ReagentEvent rid  -> EP.ProcessReagentEvent this.storage._SaveReagent this.storage.TryGetReagent rid event.operation event.change
            | EventTarget.ReagentFileEvent rid -> EP.ProcessAttachedFileEvent this.storage._SaveFileRef this.storage._DeleteFileRef this.storage.TryGetFileRef (EP.ReagentSource rid) event.operation event.change    
            | EventTarget.ReagentTagEvent rid -> EP.ProcessTagEvent this.storage._SaveTag this.storage._DeleteTag (EP.ReagentTag(rid)) event.operation event.change

            | EventTarget.ExperimentEvent exptid -> EP.ProcessExperimentEvent this.storage._SaveExperiment this.storage.TryGetExperiment exptid event.operation event.change
            | EventTarget.ExperimentFileEvent exptid -> EP.ProcessAttachedFileEvent this.storage._SaveFileRef this.storage._DeleteFileRef this.storage.TryGetFileRef  (EP.ExperimentSource exptid) event.operation event.change
            | EventTarget.ExperimentOperationEvent exptid -> EP.ProcessExperimentOperationEvent this.storage._SaveExperimentEvent this.storage._DeleteExperimentEvent this.storage.TryGetExperimentEvent exptid event.operation event.change
            | EventTarget.ExperimentSignalEvent exptid -> EP.ProcessExperimentSignalEvent this.storage._SaveExperimentSignal this.storage._DeleteExperimentSignal this.storage.TryGetExperimentSignal exptid event.operation event.change
            | EventTarget.ExperimentTagEvent exptid -> EP.ProcessTagEvent this.storage._SaveTag this.storage._DeleteTag (EP.ExperimentTag(exptid)) event.operation event.change

            | EventTarget.SampleEvent sid -> EP.ProcessSampleEvent this.storage._SaveSample this.storage.TryGetSample sid event.operation event.change
            | EventTarget.SampleDataEvent sid -> EP.ProcessAttachedFileEvent this.storage._SaveFileRef this.storage._DeleteFileRef this.storage.TryGetFileRef (EP.SampleSource sid) event.operation event.change
            | EventTarget.SampleDeviceEvent sid -> EP.ProcessSampleDeviceEvent this.storage._SaveSampleDevice this.storage._DeleteSampleDeviceStore this.storage.TryGetSampleCellStore sid event.operation event.change
            | EventTarget.SampleConditionEvent sid -> EP.ProcessSampleConditionEvent this.storage._SaveCondition this.storage._DeleteCondition this.storage.TryGetSampleCondition sid event.operation event.change
            | EventTarget.SampleTagEvent sid -> EP.ProcessTagEvent this.storage._SaveTag this.storage._DeleteTag (EP.SampleTag(sid)) event.operation event.change
            | EventTarget.SampleReplicateEvent sid -> EP.ProcessSampleReplicateEvent this.storage._SaveReplicate this.storage._UnlinkReplicate sid event.operation event.change

            | EventTarget.CellEvent cid -> EP.ProcessCellEvent this.storage._SaveCell this.storage.TryGetCell cid event.operation event.change
            | EventTarget.CellFileEvent cid -> EP.ProcessAttachedFileEvent this.storage._SaveFileRef this.storage._DeleteFileRef this.storage.TryGetFileRef (EP.CellSource cid) event.operation event.change
            | EventTarget.CellEntityEvent cid -> EP.ProcessCellEntityEvent this.storage._SaveCellEntity this.storage._RemoveCellEntity this.storage.GetEntitiesOfCell cid event.operation event.change
            | EventTarget.CellTagEvent cid -> EP.ProcessTagEvent this.storage._SaveTag this.storage._DeleteTag (EP.CellTag(cid)) event.operation event.change
            
            | EventTarget.InteractionEvent ied -> EP.ProcessInteractionEvent this.storage._SaveInteractionConstituents ied event.operation event.change

            | EventTarget.DerivedFromEvent -> EP.ProcessDerivedFromEvent this.storage._SaveDerivedFrom this.storage._TryGetDerivedFrom this.storage._DeleteDerivedFrom event.operation event.change

            | EventTarget.StartLogEvent _ -> EventResult.EmptyEventResult
            | EventTarget.FinishLogEvent _ -> EventResult.EmptyEventResult
    
            | EventTarget.ProcessDataEvent _ -> EventResult.EmptyEventResult
            | EventTarget.ParseLayoutEvent _ -> EventResult.EmptyEventResult

            | EventTarget.FileEvent _           -> failwithf "%s cannot be a Table Event Target." (EventTarget.toTargetTypeString event.target)
            | EventTarget.TimeSeriesFileEvent _ -> failwithf "%s cannot be a Table Event Target." (EventTarget.toTargetTypeString event.target)
            | EventTarget.BundleFileEvent _     -> failwithf "%s cannot be a Table Event Target." (EventTarget.toTargetTypeString event.target)
            
            | EventTarget.ObservationEvent oid -> EP.ProcessObservationEvent this.storage._SaveObservation event.operation oid event.change
            
        | BlobEvent(event,filecontent) -> 
            this.storage.SaveEvent event
            match event.target with 
            | EventTarget.FileEvent(fid) -> EP.ProcessUploadFileEvent this.storage._UploadFile fid event.operation filecontent
            | EventTarget.TimeSeriesFileEvent(fid) -> EP.ProcessTimeSeriesFileEvent this.storage._UploadTimeSeries fid event.operation filecontent
            | EventTarget.BundleFileEvent(fid) -> EP.ProcessBundleEvent this.storage._UploadFile fid event.operation filecontent
            | _ -> failwithf "%s cannot be a Blob Event Target." (EventTarget.toTargetTypeString event.target)
    

    //BCKG Modify methods (must generate events for tracking)
    
    member this.SaveDerivedFrom (derivedFrom) = 
        let dfExists = this.storage._TryGetDerivedFrom derivedFrom |> Async.RunSynchronously
        let dfEvent = 
            match dfExists with 
            | Some(currentdf) -> 
                printfn "[WARNING] This Derived From property already exists. No action will be taken."
                None
            | None -> (BCKG.Events.addDerivedFrom "" derivedFrom) |> Some
        match dfEvent with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | DerivedFromEventResult b -> b
            | EmptyEventResult -> async{return true}
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}
    
    member this.RemoveDerivedFrom (derivedFrom) = 
        let dfExists = this.storage._TryGetDerivedFrom derivedFrom |> Async.RunSynchronously
        let dfEvent = 
            match dfExists with 
            | Some(currentdf) -> 
                (BCKG.Events.removeDerivedFrom "" derivedFrom) |> Some
            | None -> 
                printfn "[WARNING] This Derived From property was not found. No action will be taken."
                None
        match dfEvent with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | DerivedFromEventResult b -> b
            | EmptyEventResult -> async{return true}
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}

    member this.SaveCell (cell:Cell) = 
        let cellExists = this.storage.TryGetCell cell.id |> Async.RunSynchronously
        let cellEvents = 
            match cellExists with 
            | Some(currentCell) -> BCKG.Events.modifyCell "" currentCell cell
            | None -> BCKG.Events.addCell "" cell |> Some
        match cellEvents with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | CellEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}

    member this.SaveCellEntity (entity:CellEntity) = 
        let existingCellEntities = this.storage.GetEntitiesOfCell entity.cellId  |> Async.RunSynchronously
        let event = BCKG.Events.addCellEntity "" (existingCellEntities |> Array.toList) entity
        match event with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | CellEntityEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}

    member this.SaveCellEntities (cellId:CellId) (entities:CellEntity[]) = 
        async {
            let! existingCellEntities = this.storage.GetEntitiesOfCell cellId 
            let! results = 
                entities 
                |> Array.toList 
                |> List.map (fun entity -> 
                    let entityEvent = BCKG.Events.addCellEntity "" (existingCellEntities |> Array.toList) entity
                    match entityEvent with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | CellEntityEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            return (Array.forall id results)
        }

    member this.RemoveCellEntity (entity:CellEntity) = 
        let existingCellEntities = this.storage.GetEntitiesOfCell entity.cellId  |> Async.RunSynchronously
        let event = BCKG.Events.removeCellEntity "" (existingCellEntities |> Array.toList) entity
        match event with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | CellEntityEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}
    
    member this.RemoveCellEntities (cellId:CellId) (entities:CellEntity[]) = 
        async {
            let! existingCellEntities = this.storage.GetEntitiesOfCell cellId 
            let! results = 
                entities 
                |> Array.toList 
                |> List.map (fun entity -> 
                    let entityEvent = BCKG.Events.removeCellEntity "" (existingCellEntities |> Array.toList) entity
                    match entityEvent with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | CellEntityEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            return (Array.forall id results)
        }

    member this.SaveSampleConditions (sampleId:SampleId, conditions: Condition[]) =
        async {
            let! existingSampleConditions = this.storage.GetSampleConditions sampleId
            let! results = 
                conditions 
                |> Array.toList 
                |> List.map (fun conc -> 
                    let concEvent = BCKG.Events.addCondition "" (existingSampleConditions |> Array.toList) conc
                    match concEvent with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | SampleConditionEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> 
                        async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
          
    member this.RemoveSampleConditions (sampleId: SampleId, conditions: Condition[]) =
        let uniqueConditions =
            match conditions.Length with
            | 0 -> this.storage.GetSampleConditions sampleId |> Async.RunSynchronously |> Array.distinctBy (fun x -> x.ToString())
            | _ -> conditions |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! results =
                uniqueConditions
                |> Array.toList
                |> List.map (fun condition ->
                    let conditionEventOpt = BCKG.Events.removeCondition "" condition
                    let e = {conditionEventOpt with user = this.user}
                    let result = this.ProcessEvent (e |> TableEvent)
                    match result with
                    | SampleConditionEventResult b -> b
                    | _ -> failwith "Unexpected Result"
                    )
                |> Async.Parallel

            return (Array.forall id results)
          }

    member this.SaveSampleCellStores (sampleId:SampleId, cellIds:(CellId * (float option * float option))[]) = 
        async {
            let! existingSampleCells = this.storage.GetSampleCells sampleId 
            let! results = 
                cellIds 
                |> Array.toList 
                |> List.map (fun cell -> 
                    let cellEvent = BCKG.Events.addSampleCellStore "" sampleId (existingSampleCells |> Array.toList) cell
                    match cellEvent with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | SampleDeviceEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }

    member this.SaveSampleDevices (sampleId:SampleId, devices:SampleDevice[]) =
        async {
            let! existingSampleDevices = this.storage.GetSampleDevices sampleId
            let! results = 
                devices
                |> Array.toList
                |> List.map (fun device ->
                    let deviceEvent = BCKG.Events.addSampleDevice "" sampleId (existingSampleDevices |> Array.toList) device
                    match deviceEvent with 
                    | Some(e) ->
                        let e  = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with
                        | SampleDeviceEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                 |> Async.Parallel
            return (Array.forall id results)
        }
    
    member this.RemoveSampleDevices (sampleId: SampleId, devices: SampleDevice[]) =
        let uniqueDevices =
            match devices.Length with
            | 0 -> this.storage.GetSampleDevices sampleId |> Async.RunSynchronously |> Array.distinctBy (fun x -> x.ToString())
            | _ -> devices |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! results =
                uniqueDevices
                |> Array.toList
                |> List.map (fun device ->
                    let deviceEventOpt = BCKG.Events.removeSampleDevice "" device.sampleId device.cellId
                    let e = {deviceEventOpt with user = this.user}
                    let result = this.ProcessEvent (e |> TableEvent)
                    match result with
                    | SampleDeviceEventResult b -> b
                    | _ -> failwith "Unexpected Result"
                    )
                |> Async.Parallel

            return (Array.forall id results)
          }

    member this.AddSampleReplicate (sampleId:SampleId, replicateId:ReplicateId) = 
        let replicateEvent = BCKG.Events.addSampleReplicate "" sampleId replicateId
        let e = {replicateEvent with user = this.user}
        let result = this.ProcessEvent (e |> TableEvent)
        match result with 
        | SampleReplicateEventResult b -> b
        | _ -> failwith "Unexpected Result"
    
    member this.UnlinkSampleReplicate (sampleId:SampleId, replicateId:ReplicateId) = 
        let replicateEvent = BCKG.Events.unlinkSampleReplicate "" sampleId replicateId
        let e = {replicateEvent with user = this.user}
        let result = this.ProcessEvent (e |> TableEvent)
        match result with 
        | SampleReplicateEventResult b -> b
        | _ -> failwith "Unexpected Result"

    member this.SaveExperimentSignals (experimentId:ExperimentId,signals:Signal[]) = 
        async {
            let! existingExperimentSignals = this.storage.GetExperimentSignals experimentId 
            let! results = 
                signals 
                |> Array.toList 
                |> List.map (fun signal -> 
                    let event = BCKG.Events.addExperimentSignal "" experimentId (existingExperimentSignals |> Array.toList) signal
                    match event with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ExperimentSignalEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }   
    
    member this.RemoveExperimentSignals (experimentId:ExperimentId,signals:Signal[]) = 
        async {
            let! existingExperimentSignals = this.storage.GetExperimentSignals experimentId 
            let! results = 
                signals 
                |> Array.toList 
                |> List.map (fun signal -> 
                    let event = BCKG.Events.removeExperimentSignal "" experimentId (existingExperimentSignals |> Array.toList) signal
                    match event with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ExperimentSignalEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }   
    

    member this.SaveExperiment (experiment:Experiment) =
        let experimentExists = this.storage.TryGetExperiment experiment.id |> Async.RunSynchronously
        let experimentEvents = 
            match experimentExists with 
            | Some(currentExperiment) -> BCKG.Events.modifyExperiment "" currentExperiment experiment
            | None -> BCKG.Events.addExperiment "" experiment |> Some
        match experimentEvents with 
               | Some event -> 
                   let event = {event with user = this.user}
                   let result = this.ProcessEvent (event |> TableEvent)
                   match result with 
                   | ExperimentEventResult b -> b
                   | _ -> failwith "Unexpected Result"
               | None -> async{return true}
    
    member this.AddExperimentFile (experimentId:ExperimentId, fileRef:FileRef) = 
        let existingExperimentFiles = this.GetExperimentFiles experimentId |> Async.RunSynchronously
        let experimentFileEvent = BCKG.Events.addExperimentFile "" experimentId (List.ofSeq existingExperimentFiles) fileRef
        match experimentFileEvent with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | ExperimentFileEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}

    member this.UnlinkExperimentFile (experimentId:ExperimentId, fileRef:FileRef) = 
        let event = BCKG.Events.unlinkExperimentFile "" experimentId fileRef
        let event = {event with user = this.user}
        let result = this.ProcessEvent (event |> TableEvent)
        match result with 
        | ExperimentFileEventResult b -> b
        | _ -> failwith "Unexpected Result"
    
    member this.AddPartTag (partId:PartId, tag:Tag) = 
        let existingTags = this.storage.GetPartTags partId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.addPartTag "" partId (existingTags |> List.ofArray) (tag)
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | PartTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
       
    member this.AddPartTags (partId:PartId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetPartTags partId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.addPartTag "" partId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | PartTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.RemovePartTag (partId:PartId, tag:Tag) = 
        let existingTags = this.storage.GetPartTags partId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.removePartTag "" partId (existingTags |> List.ofArray)   tag
        
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | PartTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.RemovePartTags (partId:PartId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetPartTags partId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.removePartTag "" partId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | PartTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }

    member this.AddReagentTag (reagentId:ReagentId, tag:Tag) = 
        let existingTags = this.storage.GetReagentTags reagentId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.addReagentTag "" reagentId (existingTags |> List.ofArray) (tag)
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | ReagentTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.AddReagentTags (reagentId:ReagentId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetReagentTags reagentId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.addReagentTag "" reagentId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ReagentTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.RemoveReagentTag (reagentId:ReagentId, tag:Tag) = 
        let existingTags = this.storage.GetReagentTags reagentId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.removeReagentTag "" reagentId (existingTags |> List.ofArray)   tag
        
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | ReagentTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}

    member this.RemoveReagentTags (reagentId:ReagentId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetReagentTags reagentId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.removeReagentTag "" reagentId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ReagentTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.AddCellTag (cellId:CellId,  tag:Tag) = 
        let existingTags = this.storage.GetCellTags cellId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.addCellTag "" cellId (existingTags |> List.ofArray) (tag)
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | CellTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.AddCellTags (cellId:CellId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetCellTags cellId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.addCellTag "" cellId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | CellTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.RemoveCellTag (cellId:CellId, tag:Tag) = 
        let existingTags = this.storage.GetCellTags cellId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.removeCellTag "" cellId (existingTags |> List.ofArray) tag
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | CellTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.RemoveCellTags (cellId:CellId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetCellTags cellId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.removeCellTag "" cellId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | CellTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.AddExperimentTag (experimentId:ExperimentId,  tag:Tag) = 
        let existingTags = this.storage.GetExperimentTags experimentId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.addExperimentTag "" experimentId (existingTags |> List.ofArray) (tag)
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | ExperimentTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.AddExperimentTags (experimentId:ExperimentId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetExperimentTags experimentId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.addExperimentTag "" experimentId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ExperimentTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.RemoveExperimentTag (experimentId:ExperimentId, tag:Tag) = 
        let existingTags = this.storage.GetExperimentTags experimentId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.removeExperimentTag "" experimentId (existingTags |> List.ofArray) tag
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | ExperimentTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.RemoveExperimentTags (experimentId:ExperimentId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetExperimentTags experimentId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.removeExperimentTag "" experimentId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | ExperimentTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.AddSampleTag (sampleId:SampleId,  tag:Tag) = 
        let existingTags = this.storage.GetSampleTags sampleId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.addSampleTag "" sampleId (existingTags |> List.ofArray) (tag)
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | SampleTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}
    
    member this.AddSampleTags (sampleId:SampleId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetSampleTags sampleId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.addSampleTag "" sampleId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | SampleTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }

    member this.RemoveSampleTag (sampleId:SampleId, tag:Tag) = 
        let existingTags = this.storage.GetSampleTags sampleId |> Async.RunSynchronously
        let tagEventOpt = BCKG.Events.removeSampleTag "" sampleId (existingTags |> List.ofArray) tag
        match tagEventOpt with 
        | Some(e) -> 
            let e = {e with user = this.user}
            let result = this.ProcessEvent (e |> TableEvent)
            match result with 
            | SampleTagEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async {return true}

    member this.RemoveSampleTags (sampleId:SampleId, tags:Tag[]) =
        let uniqueTags = tags |> Array.distinctBy (fun x -> x.ToString())
        async {
            let! existingTags = this.storage.GetSampleTags sampleId 
            let! results = 
                uniqueTags 
                |> Array.toList 
                |> List.map (fun tag -> 
                    let tagEventOpt = BCKG.Events.removeSampleTag "" sampleId (existingTags |> List.ofArray) (tag)
                    match tagEventOpt with 
                    | Some(e) -> 
                        let e = {e with user = this.user}
                        let result = this.ProcessEvent (e |> TableEvent)
                        match result with 
                        | SampleTagEventResult b -> b
                        | _ -> failwith "Unexpected Result"
                    | None -> async{return true}
                    )
                |> Async.Parallel
            
            return (Array.forall id results)
        }
    
    member this.SaveReagent (reagent:Reagent) =
        let reagentExists = this.storage.TryGetReagent reagent.id |> Async.RunSynchronously
        let reagentEvent = 
            match reagentExists with 
            | Some(currentReagent) -> BCKG.Events.modifyReagent "" currentReagent reagent
            | None -> (BCKG.Events.addReagent "" reagent) |> Some
        match reagentEvent with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | ReagentEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}
    
    member this.AddReagentFile (reagentId:ReagentId, fileRef:FileRef) = 
        let existingReagentFiles = this.GetReagentFiles reagentId |> Async.RunSynchronously
        let reagentFileEvent = BCKG.Events.addReagentFile "" reagentId (List.ofSeq existingReagentFiles) fileRef
        match reagentFileEvent with 
        | Some event -> 
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | ReagentFileEventResult b -> b
            | _ -> failwith "Unexpected Result"
        | None -> async{return true}

    member this.UnlinkReagentFile (reagentId:ReagentId, fileRef:FileRef) = 
        let event = BCKG.Events.unlinkReagentFile "" reagentId fileRef
        let event = {event with user = this.user}
        let result = this.ProcessEvent (event |> TableEvent)
        match result with 
        | ReagentFileEventResult b -> b
        | _ -> failwith "Unexpected Result"
        
    member this.SavePart (part:Part)=
        let partExists = this.storage.TryGetPart part.id |> Async.RunSynchronously
        let partEvent = 
            match partExists with 
            | Some(currentPart) -> BCKG.Events.modifyPart "" currentPart part
            | None -> Some(BCKG.Events.addPart "" part)
        match partEvent with 
        | Some event ->
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | PartEventResult b -> b
            | _ -> failwith "Unexpected Result."
        | None -> async {return true} 
    
    member this.SaveExperimentOperation (experimentId:ExperimentId, experimentOperation:ExperimentOperation)=
        
        let experimentOps = this.storage.GetExperimentOperations (experimentId) |> Async.RunSynchronously
        let experimentOpEvent = BCKG.Events.addExperimentOperation "" experimentId experimentOps experimentOperation
        match experimentOpEvent with 
        | Some event ->
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | ExperimentEventEventResult b -> b
            | _ -> failwith "Unexpected Result."
        | None -> async {return true}

    member this.RemoveExperimentOperation (experimentId:ExperimentId) (experimentOperation:ExperimentOperation)=
        
        let experimentOps = this.storage.GetExperimentOperations (experimentId) |> Async.RunSynchronously
        let experimentOpEvent = BCKG.Events.removeExperimentOperation "" experimentId experimentOps experimentOperation
        match experimentOpEvent with 
        | Some event ->
            let event = {event with user = this.user}
            let result = this.ProcessEvent (event |> TableEvent)
            match result with 
            | ExperimentEventEventResult b -> b
            | _ -> failwith "Unexpected Result."
        | None -> async {return true}
    
    member this.SaveSamples (samples:Sample []) =
        async{
            let sampleEvents = 
                samples 
                |> Array.toList
                |> List.map(fun s -> 
                    let sampleExists = this.storage.TryGetSample s.id |> Async.RunSynchronously
                    let sevents = 
                        match sampleExists with 
                        | Some(currentSample) -> BCKG.Events.modifySample "" currentSample s
                        | None -> BCKG.Events.addSample "" s |> Some
                    sevents)
                |> List.choose (fun x -> x)
                
            let! result = 
                sampleEvents 
                |> List.map (fun e ->
                    let e = {e with user = this.user}
                    let result = this.ProcessEvent (e |> TableEvent)
                    match result with 
                    | SampleEventResult b -> b
                    | _ -> failwith "Unexpected Result"
                    )        
                |> Async.Parallel
            return (Array.forall id result)
        }
    
    member this.UploadFile (fid:FileId,content:string) =
       let bytes = content |> System.Text.Encoding.UTF8.GetBytes
       //Check if the file exists?
       let fileEvent = BCKG.Events.uploadFile "" fid (fid.ToString())
       let fileEvent = {fileEvent with user = this.user}
       let result = this.ProcessEvent ((fileEvent,bytes) |> BlobEvent)
       match result with 
       | FileEventResult r -> r
       | _ -> failwith "Unexpected Result."        
        
    member this.UploadBundle (fid:FileId,(files:((string * string)[]))) =
        let filenames = files |> Array.toList |> List.unzip |> fst

        use memstream = new System.IO.MemoryStream()
        use zip = new ZipArchive(memstream, ZipArchiveMode.Create, true)
        files 
        |> Seq.iter(fun (name,dataString) ->         
            let entry = zip.CreateEntry name
            use entryStream = entry.Open()        
            let data = System.Text.Encoding.UTF8.GetBytes dataString        
            entryStream.Write(data,0,data.Length)
            )
        zip.Dispose()
        let bytes = (memstream.ToArray())
        let bundleEvent = BCKG.Events.uploadBundle "" fid filenames
        let bundleEvent = {bundleEvent with user = this.user}
        let result = this.ProcessEvent ((bundleEvent,bytes)|>BlobEvent)
        match result with 
        | BundleFileEventResult r -> r
        | _ -> failwith "Unexpected Result."
    
    member this.ParseLayout (experimentId:ExperimentId) =
        
        let event = BCKG.Events.parseLayoutEvent "" experimentId
        let event = {event with user = this.user}
        async{
            let! samplesoption = this.storage._TryParseExperimentLayout experimentId
            let sampleEvents = 
                match samplesoption with 
                | Some(samples) ->
                    let sevents = 
                       samples 
                       |> Array.toList
                       |> List.map (fun s -> 
                             let sampleExists = this.storage.TryGetSample s.id |> Async.RunSynchronously
                             match sampleExists with 
                             | Some(currentSample) -> BCKG.Events.modifySample "" currentSample s
                             | None -> (BCKG.Events.addSample "" s) |> Some) 
                       |> List.choose (fun x -> x)
                    sevents |> List.map (fun s -> {s with user = this.user; triggeredBy = event.id |> Some})
                | None -> []
            
            this.storage.SaveEvent event
            let! result = 
                sampleEvents 
                |> List.map (fun e -> 
                    let result = this.ProcessEvent (e |> TableEvent)
                    match result with 
                    | SampleEventResult b -> b
                    | SampleConditionEventResult b -> b
                    | SampleDeviceEventResult b -> b
                    | SampleDataEventResult b -> b
                    | _ -> failwith "Unexpected Result"
                    )        
                |> Async.Parallel
            return (Array.forall id result)
        }            
    
    member this.ProcessData (experimentId:ExperimentId) =
        let event = BCKG.Events.processDataEvent "" experimentId
        let event = {event with user = this.user}
        async{
            let! processResults = Storage.DataProcessor.ProcessExperimentData this.storage.TryGetExperiment this.storage.GetExperimentOperations this.storage.GetExperimentSignals this.storage.GetExperimentSamples this.storage.ListPlateReaderUniformFileBetween this.storage.LoadPlateReaderUniformFile experimentId
            
            printfn "Experiment ID = %s" (experimentId.ToString())
            let pdataevents = 
                match processResults with 
                | Some(sampleMap) ->
                    let samples,tsfiles = sampleMap |> Array.unzip
                    let tsevents = 
                        tsfiles |> Array.toList |> List.map (fun (fid,content) -> 
                            let e = BCKG.Events.uploadTimeSeriesFile "" fid (fid.ToString()) 
                            let e = {e with user = this.user; triggeredBy = Some(event.id)}
                            (e,(content|> System.Text.Encoding.UTF8.GetBytes)) |> BlobEvent)
                    let sampleEvents =                           
                        samples 
                        |> Array.toList
                        |> List.map (fun s -> 
                              let sampleExists = this.storage.TryGetSample s.id |> Async.RunSynchronously
                              match sampleExists with 
                              | Some(currentSample) -> BCKG.Events.modifySample "" currentSample s
                              | None -> (BCKG.Events.addSample "" s) |> Some)
                        |> List.choose (fun x -> x)
                        |> List.map (fun s -> (({s with user = this.user; triggeredBy = Some(event.id)}) |> TableEvent) )
                    let sampleDataEvents = 
                        sampleMap
                        |> Array.toList
                        |> List.choose (fun (s,(fid,_)) ->
                            let fileref = { fileName = fid.ToString(); fileId = fid; Type = FileType.CharacterizationData}
                            let sampleFiles = 
                                this.storage.GetSampleFiles s.id |> Async.RunSynchronously
                                |> Array.toList
                            let event_option = BCKG.Events.addSampleFile "" s.id sampleFiles fileref
                            match event_option with 
                            | Some (e) -> ({e with user = this.user; triggeredBy = Some(event.id)} |> TableEvent) |> Some
                            | None -> None)
                    (tsevents@sampleEvents@sampleDataEvents)                         
                | None -> []
            this.storage.SaveEvent event
            let! result = 
                pdataevents 
                |> List.map (fun e -> 
                    let result = this.ProcessEvent e
                    match result with 
                    | SampleEventResult b -> b
                    | SampleConditionEventResult b -> b
                    | SampleDeviceEventResult b -> b
                    | SampleDataEventResult b -> b
                    | TimeSeriesFileEventResult b -> async {return true} //is this the best way to do this?
                    | _ -> failwith "Unexpected Result"
                    )        
                |> Async.Parallel
            return (Array.forall id result)
        
        }

    member this.SaveObservation (obs:Observation) = 
        //TODO: check if observation exists?
        
        let event = BCKG.Events.addObservation "" obs
        let event' = {event with user = this.user}
        let result = this.ProcessEvent (event' |> TableEvent)
        match result with 
        | ObservationEventResult b -> b
        | _ -> failwith "Unexpected Result"

    member this.SaveObservations (observations:Observation[]) = 
        //TODO: check if observation exists?
        async{
            let obsEvents =
                observations
                |> Array.toList
                |> List.map (fun obs -> BCKG.Events.addObservation "" obs)

            let! result = 
                obsEvents 
                |> List.map (fun e ->
                    let e = {e with user = this.user}
                    let result = this.ProcessEvent (e |> TableEvent)
                    match result with 
                    | ObservationEventResult b -> b
                    | _ -> failwith "Unexpected Result"
                    )        
                |> Async.Parallel
            return (Array.forall id result)
        }
    
    member this.GetKnowledgeGraph() = 
       async {                   
           let! tagsIndex = this.storage.IndexTags() 
           let! operationsIndex = this.storage.IndexExperimentOperations()
           let! filesIndex = this.storage.IndexFiles()
           let! signalsIndex = this.storage.IndexExperimentSignals() 
           let! conditionsIndex = this.storage.IndexSampleConditions() 
           let! cellsIndex = this.storage.IndexSampleCells()  
           let! entitiesIndex = this.storage.IndexCellEntities()
           let! observationsIndex = this.storage.IndexObservations()
           let! parts = this.storage.GetParts()                      
           let! reagents = this.storage.GetReagents()                      
           let! experiments = this.storage.GetExperiments()                                         
           let! cells = this.storage.GetCells()   
           let! samples = this.storage.GetSamples()

           printf "Preparing Knowledge Graph structure..."
           let graph = 
              { partsMap       = parts |> Array.map (fun x -> x.id, x) |> Map.ofArray
                experimentsMap = experiments |> Array.map (fun x -> x.id, x) |> Map.ofArray
                reagentsMap    = reagents |> Array.map (fun x -> x.id, x) |> Map.ofArray
                cellsMap       = cells |> Array.map (fun x -> x.id, x) |> Map.ofArray
                tags           = tagsIndex                
                experimentOperations = operationsIndex |> Map.map(fun _ ops -> ops |> Array.map (fun op -> op.id))
                operationsMap        = operationsIndex |> Map.toArray |> Array.collect snd |> Array.map (fun op -> op.id, op) |> Map.ofArray
                samplesMap = samples |> Array.map (fun x -> x.id, x) |> Map.ofArray                        
                signalsMap = signalsIndex |> Map.toArray |> Array.collect snd |> Array.map (fun op -> op.id, op) |> Map.ofArray
                experimentSignals = signalsIndex |> Map.map (fun _ x -> Array.map (fun (y:Signal) -> y.id) x)           
                sampleConditions = conditionsIndex
                sampleCells = cellsIndex                              
                sampleFiles = samples |> Array.map(fun x -> x.id, if filesIndex.ContainsKey x.id.guid then filesIndex.[x.id.guid] else Array.empty) |> Map.ofArray
                cellEntities = entitiesIndex
                observations = observationsIndex
              }  
           printfn "done"
           return graph
           }
        
    member this.PrintStats() =
           async {        
                 let! knowledgeGraph = this.GetKnowledgeGraph()
                 let parts = knowledgeGraph.parts
                 let reagents = knowledgeGraph.reagents
                 let experiments = knowledgeGraph.experiments
                 let samples = knowledgeGraph.samples
       
                 printfn "\n\n\n\n\n---------------------------------------------------------\nBCKG Statistics:"
                 printfn "- %i genetic parts" parts.Length
                 printfn "- %i reagents" reagents.Length
                 printfn "\t- %i chemicals" (reagents |> Array.filter (fun r -> match r with | Chemical _ -> true | _ -> false)) .Length
                 //printfn "\t- %i cell devices" (reagents |> Array.filter (fun r -> r.sequence.IsSome && r.context.IsSome)) .Length
                 //printfn "\t- %i DNA devices" (reagents |> Array.filter (fun r -> r.sequence.IsSome && r.context.IsNone)) .Length
                 printfn "- %i experiments" experiments.Length
                 printfn "- %i samples" samples.Length
               }
