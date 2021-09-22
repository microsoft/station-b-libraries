// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Entities
open BCKG.Domain
open BCKG.Events
open Microsoft.WindowsAzure.Storage.Table


let private tolerantDateTimeParse asString =
    //This is intended to be temporary while we migrate to ISO 8601
    let primaryParseWorked, isoDateTime =
        System.DateTime.TryParseExact(asString, "o", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None)
    if primaryParseWorked then
        isoDateTime
    else
        let backupParseWorked, backupStyleDateTime = System.DateTime.TryParseExact(asString, "dd/MM/yyyy HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None)
        if backupParseWorked then
            backupStyleDateTime
        else
            failwithf "Failed to parse '%s' as a DateTime" asString

type EventStore (event:Event) = 
    inherit TableEntity("",event.id.ToString())
    new() = EventStore(Event.empty)
    member val targetId = EventTarget.toTargetId event.target with get,set
    member val targetType = EventTarget.toTargetTypeString event.target with get,set
    //
    member val operation = EventOperation.toString event.operation with get,set
    member val message = event.message with get,set
    member val change = event.change with get,set
    member val timestamp = event.timestamp.ToUniversalTime().ToString("o") with get,set
    member val triggeredBy = Event.serialize_triggeredByProperty (event.triggeredBy) with get,set
    member val user = event.user with get,set
    static member toEvent(store:EventStore) = 
        {
            Event.id = store.RowKey |> System.Guid |> EventId
            Event.operation = store.operation |> EventOperation.fromString
            Event.message = store.message
            Event.change = store.change
            Event.target = 
                match (store.targetType) with 
                | "PromoterPartEvent" -> store.targetId |> System.Guid |>  PromoterId |> PromoterPartId |> PartEvent
                | "CDSPartEvent" -> store.targetId |> System.Guid |>  CDSId |> CDSPartId |> PartEvent
                | "TerminatorPartEvent" -> store.targetId |> System.Guid |>  TerminatorId |> TerminatorPartId |> PartEvent
                | "UserDefinedPartEvent" -> store.targetId |> System.Guid |>  UserDefinedId |> UserDefinedPartId |> PartEvent
                | "RBSPartEvent" -> store.targetId |> System.Guid |>  RBSId |> RBSPartId |> PartEvent
                | "ScarPartEvent" -> store.targetId |> System.Guid |>  ScarId |> ScarPartId |> PartEvent
                | "BackbonePartEvent" -> store.targetId |> System.Guid |>  BackboneId |> BackbonePartId |> PartEvent
                | "OriPartEvent" -> store.targetId |> System.Guid |>  OriId |> OriPartId |> PartEvent
                | "LinkerPartEvent" -> store.targetId |> System.Guid |>  LinkerId |> LinkerPartId |> PartEvent
                | "RestrictionSitePartEvent" -> store.targetId |> System.Guid |>  RestrictionSiteId |> RestrictionSitePartId |> PartEvent
                
                | "PromoterPartTagEvent" -> store.targetId |> System.Guid |>  PromoterId |> PromoterPartId |> PartTagEvent
                | "CDSPartTagEvent" -> store.targetId |> System.Guid |>  CDSId |> CDSPartId |> PartTagEvent
                | "TerminatorPartTagEvent" -> store.targetId |> System.Guid |>  TerminatorId |> TerminatorPartId |> PartTagEvent
                | "UserDefinedPartTagEvent" -> store.targetId |> System.Guid |>  UserDefinedId |> UserDefinedPartId |> PartTagEvent
                | "RBSPartTagEvent" -> store.targetId |> System.Guid |>  RBSId |> RBSPartId |> PartTagEvent
                | "ScarPartTagEvent" -> store.targetId |> System.Guid |>  ScarId |> ScarPartId |> PartTagEvent
                | "BackbonePartTagEvent" -> store.targetId |> System.Guid |>  BackboneId |> BackbonePartId |> PartTagEvent
                | "OriPartTagEvent" -> store.targetId |> System.Guid |>  OriId |> OriPartId |> PartTagEvent
                | "LinkerPartTagEvent" -> store.targetId |> System.Guid |>  LinkerId |> LinkerPartId |> PartTagEvent
                | "RestrictionSitePartTagEvent" -> store.targetId |> System.Guid |>  RestrictionSiteId |> RestrictionSitePartId |> PartTagEvent
                
                

                | "DNAReagentEvent"         -> store.targetId |> System.Guid |> DNAId |> DNAReagentId |> ReagentEvent
                | "RNAReagentEvent"         -> store.targetId |> System.Guid |> RNAId |> RNAReagentId |> ReagentEvent
                | "ChemicalReagentEvent"         -> store.targetId |> System.Guid |> ChemicalId |> ChemicalReagentId |> ReagentEvent
                | "ProteinReagentEvent"         -> store.targetId |> System.Guid |> ProteinId |> ProteinReagentId |> ReagentEvent
                | "GenericEntityReagentEvent"         -> store.targetId |> System.Guid |> GenericEntityId |> GenericEntityReagentId |> ReagentEvent
                
                | "DNAReagentFileEvent"     -> store.targetId |> System.Guid |> DNAId |> DNAReagentId |> ReagentFileEvent
                | "RNAReagentFileEvent"     -> store.targetId |> System.Guid |> RNAId |> RNAReagentId |> ReagentFileEvent
                | "ChemicalReagentFileEvent"     -> store.targetId |> System.Guid |> ChemicalId |> ChemicalReagentId |> ReagentFileEvent
                | "ProteinReagentFileEvent"     -> store.targetId |> System.Guid |> ProteinId |> ProteinReagentId |> ReagentFileEvent
                | "GenericEntityReagentFileEvent"     -> store.targetId |> System.Guid |> GenericEntityId |> GenericEntityReagentId |> ReagentFileEvent
                
                | "DNAReagentTagEvent"         -> store.targetId |> System.Guid |> DNAId |> DNAReagentId |> ReagentTagEvent
                | "RNAReagentTagEvent"         -> store.targetId |> System.Guid |> RNAId |> RNAReagentId |> ReagentTagEvent
                | "ChemicalReagentTagEvent"         -> store.targetId |> System.Guid |> ChemicalId |> ChemicalReagentId |> ReagentTagEvent
                | "ProteinReagentTagEvent"         -> store.targetId |> System.Guid |> ProteinId |> ProteinReagentId |> ReagentTagEvent
                | "GenericReagentTagEvent"         -> store.targetId |> System.Guid |> GenericEntityId |> GenericEntityReagentId |> ReagentTagEvent
                

                | "ExperimentEvent"      -> store.targetId |> System.Guid |> ExperimentId |> ExperimentEvent
                | "ExperimentFileEvent"  -> store.targetId |> System.Guid |> ExperimentId |> ExperimentFileEvent
                | "ExperimentOperationEvent" -> store.targetId |> System.Guid |> ExperimentId |> ExperimentOperationEvent
                | "ExperimentSignalEvent"-> store.targetId |> System.Guid |> ExperimentId |> ExperimentSignalEvent
                | "ExperimentTagEvent"   -> store.targetId |> System.Guid |> ExperimentId |> ExperimentTagEvent
                

                | "SampleEvent"          -> store.targetId |> System.Guid |> SampleId |> SampleEvent
                | "SampleDeviceEvent"    -> store.targetId |> System.Guid |> SampleId |> SampleDeviceEvent
                | "SampleDataEvent"      -> store.targetId |> System.Guid |> SampleId |> SampleDataEvent
                | "SampleConditionEvent" -> store.targetId |> System.Guid |> SampleId |> SampleConditionEvent
                
                | "CellEvent"            -> store.targetId |> System.Guid |> CellId |> CellEvent
                | "CellEntityEvent"      -> store.targetId |> System.Guid |> CellId |> CellEntityEvent  
                | "CellFileEvent"        -> store.targetId |> System.Guid |> CellId |> CellFileEvent
                | "CellTagEvent"         -> store.targetId |> System.Guid |> CellId |> CellTagEvent
                
                | "FileEvent"            -> store.targetId |> System.Guid |> FileId |> FileEvent
                | "TimeSeriesFileEvent"  -> store.targetId |> System.Guid |> FileId |> TimeSeriesFileEvent
                | "BundleFileEvent"      -> store.targetId |> System.Guid |> FileId |> BundleFileEvent
                | "StartLogEvent"        -> store.targetId |> StartLogEvent
                | "FinishLogEvent"       -> store.targetId |> FinishLogEvent
                | "ProcessDataEvent"     -> store.targetId |> System.Guid |> ExperimentId |> EventTarget.ProcessDataEvent
                | "ParseLayoutEvent"     -> store.targetId |> System.Guid |> ExperimentId |> EventTarget.ParseLayoutEvent

                | "DerivedFromEvent"     -> EventTarget.DerivedFromEvent 

                | _ -> failwithf "%s is an unrecognized Event Target." (store.targetType)
                    
            Event.timestamp = store.timestamp |> System.DateTime.Parse
            Event.triggeredBy = 
                match store.triggeredBy with 
                | "" -> None
                | _ -> store.triggeredBy |> System.Guid |> EventId |> Some
            Event.user = store.user
        }

//Generic File Ref Store
type FileRefStore(source:System.Guid, fileRef:FileRef) =     
   inherit TableEntity("", fileRef.fileId.ToString())
   new() = FileRefStore(System.Guid.NewGuid(), FileRef.empty) //required, don't use directly    
   member val source = source.ToString() with get, set
   //member val fileId = (match fileRef.fileId with FileId guid -> guid.ToString())  with get, set //this is the rowKey
   member val Type = (fileRef.Type |> FileType.toString) with get, set        
   member val fileName = fileRef.fileName with get, set            
   static member toFileRef (store:FileRefStore) = 
       { fileName = store.fileName
         Type = store.Type |> FileType.fromString
         fileId = store.RowKey |> System.Guid |> FileId
       }

//Generic Tag Store
type TagStore(source:System.Guid, tag:Tag) =
    inherit TableEntity("",System.Guid.NewGuid().ToString())
    new() = TagStore(System.Guid.NewGuid(), Tag "")
    member val source = source.ToString() with get,set
    member val tag = Tag.toString tag with get, set


//Part Store
type PartStore(part:Part) =     
   inherit TableEntity("", part.id.ToString())
   new() = PartStore(Part.empty) //required, don't use directly    
   member val Name = part.getProperties.name with get, set    
   member val Type = (part |> Part.GetType) with get, set        
   member val Sequence = part.getProperties.sequence with get, set
   member val Deprecated = part.getProperties.deprecated
   static member toPart (store:PartStore) =
       let partProperties = {
           
           PartProperties.sequence = store.Sequence
           PartProperties.name = store.Name
           PartProperties.deprecated = store.Deprecated
       }
       let guid = store.RowKey |> System.Guid
       Part.FromStringType guid partProperties (store.Type)
       

//Reagent Store
type ReagentStore(reagent:Reagent) =     
    inherit TableEntity("", reagent.id.ToString())
    new() = ReagentStore(Reagent.empty) //required, don't use directly    
    member val Name = reagent.getProperties.name with get, set
    member val Barcode = (match reagent.getProperties.barcode with | Some (Barcode bc) -> bc | None -> "") with get, set
    member val Notes = reagent.getProperties.notes with get, set        
    member val Deprecated  = reagent.getProperties.deprecated with get,set
    member val Type = (reagent.getType) with get, set        
    
    //DNA Specific
    member val DNAType = (match reagent with | DNA dna -> DNAType.ToString dna.Type | _ -> "" ) with get,set
    member val Sequence = 
        let sequence = 
            match reagent with 
            | DNA dna -> dna.sequence 
            | RNA rna -> rna.sequence
            | _ -> ""
        sequence with get, set
    member val Concentration = 
        let conc = 
            match reagent with 
            | DNA dna -> (match dna.concentration with Some conc -> Concentration.toString conc | None -> "")
            | _ -> ""
        conc with get,set
    
    //RNA Specific 
    member val RNAType = (match reagent with | RNA rna -> RNAType.ToString rna.Type | _ -> "" ) with get,set
    (*Also has sequence*)

    //Chemical Specific
    member val ChemicalType = (match reagent with | Chemical chem -> ChemicalType.ToString chem.Type | _ -> "") with get,set

    //Protein Specific
    member val isReporter = (match reagent with | Protein prot -> prot.isReporter.ToString() | _ -> "") with get,set

    static member toReagent(store:ReagentStore) = 
        let reagentProperties = 
            {
                ReagentProperties.barcode = if store.Barcode.Trim() = "" then None else Some (Barcode store.Barcode)
                ReagentProperties.name = store.Name 
                ReagentProperties.notes = store.Notes
                ReagentProperties.deprecated = store.Deprecated
            }
        match store.Type with 
        | "DNA" ->
            let dnaType = DNAType.fromString store.DNAType
            let sequence = store.Sequence
            let concentration = match store.Concentration with | "" -> None | _ -> store.Concentration |> Concentration.Parse |> Some
            {id = store.RowKey |> System.Guid |> DNAId; properties = reagentProperties;Type=dnaType;sequence = sequence;concentration = concentration} |> DNA
        | "RNA" -> 
            let rnaType = RNAType.fromString store.RNAType
            let sequence = store.Sequence
            {id = store.RowKey |> System.Guid |> RNAId; properties = reagentProperties; Type = rnaType; sequence = sequence} |> RNA
        | "Protein" ->
            let isReporter = System.Boolean.Parse(store.isReporter)
            {id = store.RowKey |> System.Guid |> ProteinId; properties = reagentProperties;isReporter = isReporter} |> Protein
        | "Chemical" ->
            let chemicalType = ChemicalType.fromString store.ChemicalType
            {id = store.RowKey |> System.Guid |> ChemicalId; properties = reagentProperties;ChemicalReagent.Type = chemicalType} |> Chemical
        | "Generic Entity" ->
            {id = store.RowKey |> System.Guid |> GenericEntityId; properties = reagentProperties } |> GenericEntity
        | _ -> failwithf "Unknown Reagent type: %s" (store.Type)

//Experiment Stores
type ExperimentStore(experiment:Experiment) =
   inherit TableEntity("", experiment.id.ToString())
   new() = ExperimentStore(Experiment.empty) //required, don't use directly
   member val Name = experiment.name with get, set
   member val Notes = experiment.notes with get, set
   member val Type = (experiment.Type |> ExperimentType.toString) with get, set     
   member val Deprecated = experiment.deprecated with get,set
   static member toExperiment (store:ExperimentStore) = 
       { Experiment.id = store.RowKey |> System.Guid |> ExperimentId
         Experiment.name = store.Name
         Experiment.notes = store.Notes
         Experiment.Type = ExperimentType.fromString store.Type 
         Experiment.deprecated = store.Deprecated
       }

type ExperimentOperationStore(source:ExperimentId, experimentEvent:ExperimentOperation) =     
    inherit TableEntity("", experimentEvent.id.ToString())
    new() = ExperimentOperationStore(ExperimentId.Create(), ExperimentOperation.empty) //required, don't use directly        
    member val source = source.ToString() with get, set
    member val Type = (experimentEvent.Type |> ExperimentOperationType.toString) with get, set        
    member val triggerTime =(experimentEvent.timestamp.ToUniversalTime().ToString("o")) with get, set      
    static member toExperimentEvent (store:ExperimentOperationStore) = 
        { id = store.RowKey |> System.Guid |> ExperimentOperationId
          Type = store.Type |> ExperimentOperationType.fromString
          timestamp = store.triggerTime |> tolerantDateTimeParse
        }

type SignalStore(experimentId: ExperimentId, signal: Signal) = 
    inherit TableEntity("", experimentId.ToString() + signal.id.ToString())
    new() = SignalStore(ExperimentId.Create(), Signal.empty) //required, don't use directly        
    member val id = signal.id.ToString() with get, set
    member val experimentId = experimentId.ToString() with get, set
    member val Type = (signal.settings |> SignalSettings.toTypeString) with get, set
    member val wavelength = (match signal.settings with PlateReaderAbsorbance s -> s.wavelength | _ -> -1.0) with get, set  //TODO: null as negative values?
    member val gain = 
        (   match signal.settings with
            | PlateReaderAbsorbance s -> s.gain
            | PlateReaderFluorescence s -> s.gain
            | _ -> -1.0
        ) with get, set  //TODO: null as negative values?
    member val correction = (match signal.settings with PlateReaderAbsorbance s -> s.correction | _ -> -1.0) with get, set  //TODO: null as negative values?
    member val emission = 
        (   match signal.settings with 
            | PlateReaderFluorescence s -> s.emissionFilter |> PlateReaderFilter.toString 
            | _ -> "" //TODO: null as empty string?
        ) with get, set  
    member val excitation = 
        (   match signal.settings with 
            | PlateReaderFluorescence s -> s.excitationFilter |> PlateReaderFilter.toString 
            | _ -> "" //TODO: null as empty string?
        ) with get, set  
    
    member val units = (match signal.units with Some x -> x | None -> "") with get, set
    static member toSignal(s:SignalStore) = 
        let settings = 
            let t = s.Type |> SignalSettings.fromTypeString
            match t with 
            | PlateReaderFluorescence _ -> PlateReaderFluorescenceSettings.Create(s.excitation |> PlateReaderFilter.fromString, s.emission |> PlateReaderFilter.fromString, s.gain) |> PlateReaderFluorescence
            | PlateReaderAbsorbance _ -> PlateReaderAbsorbanceSettings.Create(s.wavelength, s.gain, s.correction) |> PlateReaderAbsorbance
            | PlateReaderTemperature -> t
            | PlateReaderLuminescence -> t
            | Titre -> t
            | GenericSignal _ -> t

        { Signal.id = s.id |> System.Guid |> SignalId
          settings = settings
          units = (if s.units = "" then None else Some s.units)
        }

//Sample Stores
type SampleStore(sample:Sample) = 
    inherit TableEntity("", sample.id.ToString())
    new() = SampleStore(Sample.empty) //required, don't use directly        
    member val experimentId = sample.experimentId.ToString() with get, set      
    member val MetaType = (sample.meta |> SampleMeta.toStringType) with get, set
    member val virtualWellRow = (match sample.meta with PlateReaderMeta meta -> meta.virtualWell.row | _ -> -1) with get, set //NOTE: -1 as null
    member val virtualWellCol = (match sample.meta with PlateReaderMeta meta -> meta.virtualWell.col | _ -> -1) with get, set //NOTE: -1 as null
    member val physicalWellRow =  //NOTE: -1 as null
        ( match sample.meta with 
         | PlateReaderMeta meta -> 
            match meta.physicalWell with 
            | Some w -> w.row
            | None -> -1
         | _ -> -1
        ) with get, set
    member val physicalWellCol = //NOTE: -1 as null
        ( match sample.meta with 
         | PlateReaderMeta meta -> 
            match meta.physicalWell with 
            | Some w -> w.col
            | None -> -1
         | _ -> -1
        ) with get, set
    member val physicalPlateName = 
        ( match sample.meta with 
         | PlateReaderMeta meta -> 
            match meta.physicalPlateName with 
            | Some w -> w 
            | None -> ""
         | _ -> ""
        ) with get, set    
                        
    static member toSample (store:SampleStore) = 
        let metaType = store.MetaType |> SampleMeta.fromStringType
        let metaData = 
            match metaType with
            | PlateReaderMeta _ -> 
             { virtualWell = 
                { row = store.virtualWellRow
                  col = store.virtualWellCol
                }
               physicalWell = 
                if store.physicalWellCol>=0 && store.physicalWellRow>=0 then 
                    { row = store.physicalWellRow
                      col = store.physicalWellCol
                    }
                    |> Some
                else
                    None
               
               physicalPlateName = if store.physicalPlateName.Trim() = "" then None else store.physicalPlateName.Trim() |> Some
             }    
             |> PlateReaderMeta
            | MissingMeta -> MissingMeta

        {  id = store.RowKey |> System.Guid |> SampleId
           experimentId = store.experimentId |> System.Guid |> ExperimentId
           meta = metaData
           deprecated = false
         }

type ConditionStore(condition:Condition) =    
    inherit TableEntity("", condition.sampleId.ToString() + condition.reagentId.ToString()) //TODO: is it OK to use compound row keys? Does it give faster querying for specific entries?
    new() = ConditionStore(Condition.Create()) //required, don't use directly    
    member val sampleId = condition.sampleId.ToString() with get, set  
    member val reagentId = condition.reagentId.ToString() with get, set
    member val reagentType = (ReagentId.GetType condition.reagentId) with get,set
    member val value = Concentration.getValue condition.concentration with get, set
    member val valueUnits = Concentration.getUnit condition.concentration with get, set
    member val time = (match condition.time with | Some x -> Time.getValue x | None -> -1.0) with get, set
    member val timeUnits = (match condition.time with | Some x -> Time.getUnit x | None -> "") with get, set
    static member toCondition (store:ConditionStore) =      
        { reagentId =  ReagentId.FromType (store.reagentId |> System.Guid) (store.reagentType)                        
          sampleId = store.sampleId |> System.Guid |> SampleId
          concentration = 
            let units = if store.valueUnits = null || store.valueUnits = "" then "uM" else store.valueUnits
            Concentration.Create store.value units
          time = 
            let units = if store.timeUnits = null || store.timeUnits = "" then "h" else store.timeUnits
            if store.time < 0.0 then None else Some (Time.Create store.time units)
        }

type SampleReplicateStore(sampleId: SampleId, replicateId: ReplicateId) = 
    inherit TableEntity("", replicateId.ToString())
    new() = SampleReplicateStore(SampleId.Create(), ReplicateId.Create())
    member val sampleId = sampleId.ToString() with get, set

type SampleCellStore(sampleId: SampleId, cellId:CellId, cellDensity:float option, cellPreSeeding:float option) = 
    inherit TableEntity("", sampleId.ToString() + cellId.ToString()) //TODO: is it OK to use compound row keys? Does it give faster querying for specific entries?
    new() = SampleCellStore(SampleId.Create(), CellId.Create(), None, None) //required, don't use directly    
    member val sampleId = sampleId.ToString() with get, set  
    member val cellId = cellId.ToString() with get, set
    member val cellDensity = (match cellDensity with Some x -> x | None -> -1.0) with get, set
    member val cellPreSeeding = (match cellPreSeeding with Some x -> x | None -> -1.0) with get, set

type SampleDeviceStore(sampleDevice: SampleDevice) = 
    inherit TableEntity("", sampleDevice.sampleId.ToString() + sampleDevice.cellId.ToString())
    new() = SampleDeviceStore(SampleDevice.empty)  
    member val sampleId = sampleDevice.sampleId.ToString() with get, set  
    member val cellId = sampleDevice.cellId.ToString() with get, set
    member val cellDensity = (match sampleDevice.cellDensity with Some x -> x | None -> -1.0) with get, set
    member val cellPreSeeding = (match sampleDevice.cellPreSeeding with Some x -> x | None -> -1.0) with get, set

    static member toSampleDevice(store:SampleDeviceStore) = 
        { SampleDevice.cellId = store.cellId |> System.Guid.Parse |> CellId
          SampleDevice.sampleId = store.sampleId |> System.Guid.Parse |> SampleId
          SampleDevice.cellDensity = match store.cellDensity with | -1.0 -> None | _ -> store.cellDensity |> Some
          SampleDevice.cellPreSeeding = match store.cellPreSeeding with | -1.0 -> None | _ -> store.cellPreSeeding |> Some
        }


//Cell Stores
type CellStore(cell:Cell) =
    inherit TableEntity("", cell.id.ToString())
    new() = CellStore(Cell.empty) //required, don't use directly
    member val Name = cell.name with get, set
    member val Barcode = (match cell.barcode with | Some (Barcode bc) -> bc | None -> "") with get, set
    member val Notes = cell.notes with get, set
    member val Genotype = cell.genotype with get,set
    member val Type = cell.getType with get,set
    member val ProkaryoticType = (match cell with | Prokaryote p -> ProkaryoteType.ToString p.Type | Eukaryote _ -> "") with get,set
    member val Deprecated  = cell.deprecated with get,set
    static member toCell(store:CellStore) = 
        let cellProperties = {
            CellProperties.id = store.RowKey |> System.Guid |> CellId
            CellProperties.barcode = if store.Barcode.Trim() = "" then None else Some (Barcode store.Barcode)
            CellProperties.name = store.Name
            CellProperties.notes = store.Notes
            CellProperties.genotype = store.Genotype
            CellProperties.deprecated = store.Deprecated
        }
        match store.Type with 
        | "Prokaryote" -> 
            let domain = 
                match store.ProkaryoticType with 
                | "Bacteria" -> BCKG.Domain.ProkaryoteType.Bacteria
                | _ -> failwithf "Unkown Prokaryote Domain: %s" store.ProkaryoticType
            {Type = domain;properties = cellProperties} |> Prokaryote
        | _ -> failwithf "Unknown Cell Type: %s" store.Type

type CellEntityStore(cellEntity:CellEntity) = 
    inherit TableEntity("",System.Guid.NewGuid().ToString())
    new() = CellEntityStore(CellEntity.empty)
    member val cellId = cellEntity.cellId.ToString() with get,set
    member val entity = cellEntity.entity.ToString() with get,set
    member val entityType = (ReagentId.GetType cellEntity.entity) with get,set
    member val compartment = (CellCompartment.toString cellEntity.compartment) with get,set
    static member toCellEntity(store:CellEntityStore) = 
        {
            cellId = store.cellId |> System.Guid |> CellId
            compartment = store.compartment |> CellCompartment.fromString
            entity = ReagentId.FromType (store.entity |> System.Guid) (store.entityType) 
        }


//Interaction Stores
type InteractionStore(interaction:InteractionProperties, iType:string) = 
    inherit TableEntity("", interaction.id.ToString())
    new() =  InteractionStore(InteractionProperties.empty,"") //required, don't use directly. 
    member val Type = iType with get,set
    member val notes = interaction.notes with get,set
    member val deprecated = interaction.deprecated with get,set
    

    static member toInteraction (entities:(InteractionNodeType*System.Guid*string*int)list) (store:InteractionStore) =     
        let groupedEntities = 
            entities 
            |> List.groupBy (fun (t,_,_,_) -> t)
            |> List.map (fun (ieType,ielist) -> 
                let comps = ielist |> List.map (fun (_,guid,c,d) -> (guid,c,d))
                (ieType,comps))
            
        let properties = 
            {
                InteractionProperties.id         = store.RowKey |> System.Guid |> InteractionId
                InteractionProperties.notes      = store.notes
                InteractionProperties.deprecated = store.deprecated
            }
        try 
            let FindEntity t = 
                groupedEntities 
                |> List.find (fst >> (=) t)
                |> snd 
                |> List.head

            let TryFindRxnEntity t = 
                let res = groupedEntities |> List.tryFind (fst >> (=) t)
                match res with 
                | Some (_,found) -> 
                    found
                    |> List.map (fun (a,b,c) -> ReagentId.FromType a b)
                | None -> []

            let TryFindAndGroupRxnEntity t = 
                let res = groupedEntities |> List.tryFind (fst >> (=) t)
                match res with 
                | Some (_,found) -> 
                    found
                    |> List.groupBy (fun (_,b,c) -> c)
                    |> List.map (fun (a,(rlist)) -> rlist |> List.map (fun (a,b,c) -> ReagentId.FromType a b) )
                | None -> []


            let fst3  (x,_,_) = x

            match store.Type with 
            | "CodesFor" ->
                {
                    properties = properties
                    cds        = InteractionNodeType.Template |> FindEntity |> fst3 |>  CDSId
                    protein    = InteractionNodeType.Product |> FindEntity |> fst3 |> ProteinId
                } 
                |> CodesFor
            
            | "Genetic Activation" ->
                {
                    properties = properties
                    activator  =  InteractionNodeType.Activator |> FindEntity |> fun (a,b,_) -> [ReagentId.FromType a b]
                    activated  = InteractionNodeType.Activated |> FindEntity |> fst3 |>  PromoterId
                } 
                |> GeneticActivation

                
            | "Genetic Inhibition" -> 
                {
                    properties = properties
                    inhibitor = InteractionNodeType.Inhibitor|> FindEntity |> fun (a,b,_) -> [ReagentId.FromType a b]
                    inhibited = InteractionNodeType.Inhibited |> FindEntity |> fst3 |> PromoterId
                } 
                |> GeneticInhibition

            | "Reaction" ->    
                {
                    properties = properties
                    reactants  = TryFindAndGroupRxnEntity InteractionNodeType.Reactant
                    products   = TryFindAndGroupRxnEntity InteractionNodeType.Product
                    enzyme     = TryFindRxnEntity InteractionNodeType.Enzyme
                } |> Reaction
                
            | _ -> failwithf "Unknown interaction type: %s" (store.Type)
        with
        | _ -> failwithf "Could not load entities for interaction %A of type %A (entities is %A)" store.RowKey store.Type entities

type InteractionEntityStore(interactionId:InteractionId,etype:InteractionNodeType,guid:System.Guid,entityType:string, eindex:int) = 
    inherit TableEntity("", System.Guid.NewGuid().ToString())
    new () = InteractionEntityStore(InteractionId.Create(),InteractionNodeType.Activator,System.Guid.NewGuid(),"",0)
    member val interactionId = interactionId.ToString() with get,set
    member val Type = InteractionNodeType.toString(etype) with get,set
    member val entityId = (guid.ToString()) with get,set
    member val entityType = entityType with get,set
    member val complexIndex = eindex with get,set


type DerivedFromStore(derivedFrom:DerivedFrom) = 
    inherit TableEntity("", System.Guid.NewGuid().ToString())
    new () = DerivedFromStore(DerivedFrom.empty)
    member val source = (DerivedFrom.GetSourceGuid derivedFrom).ToString() with get,set
    member val target = (DerivedFrom.GetTargetGuid derivedFrom).ToString() with get,set
    member val Type = DerivedFrom.GetType derivedFrom with get,set

    static member toDerivedFrom(store:DerivedFromStore) =
        let source = System.Guid.Parse store.source
        let target = System.Guid.Parse store.target
        DerivedFrom.fromType source target store.Type
        

type ObservationStore(observation:Observation) = 
    inherit TableEntity("", observation.id.ToString())
    new () = ObservationStore(Observation.empty)
    member val sample = observation.sampleId.ToString() with get,set
    member val signal = observation.signalId.ToString() with get,set
    member val value  = observation.value with get,set
    member val observedAt = (match observation.timestamp with Some t -> t.ToUniversalTime().ToString("o")  | None -> "") with get, set
    member val replicate = (match observation.replicate with Some r -> r.ToString() | None -> "") with get, set

    static member toObservation(store:ObservationStore) =
        { Observation.id = store.RowKey |> System.Guid.Parse |> ObservationId
          Observation.sampleId = store.sample |> System.Guid.Parse |> SampleId
          Observation.signalId = store.signal |> System.Guid.Parse |> SignalId
          Observation.value = store.value
          Observation.timestamp = if store.observedAt = "" then None else store.observedAt |> tolerantDateTimeParse |> Some
          Observation.replicate = if store.replicate = "" then None else store.replicate |> System.Guid |> ReplicateId |> Some
        }
