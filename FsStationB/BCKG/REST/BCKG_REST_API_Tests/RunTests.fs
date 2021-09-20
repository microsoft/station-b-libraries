module BCKG.Test.RunTests

open BCKG.Test.Entities
open BCKG.Test.APITests
open BCKG.Domain
open BCKG.Events

let checkPart (rest:RestAPI) (part:Part) =
    printfn "Checking %s Part %s" (Part.GetType part) (part.id.ToString())
    let part' = rest.getPart (part.id)
    assert (part'.id = part.id)
    assert (part'.name = part.name)
    //assert (part'.deprecated = part.deprecated) //Part deprecation hasn't been added yet.
    assert (part'.sequence = part.sequence)
    assert ((Part.GetType part')  = (Part.GetType part))

let compareProperties (props:ReagentProperties) (props':ReagentProperties) =
    assert (props.name = props'.name)
    assert (props.notes = props'.notes)
    assert (props.deprecated = props'.deprecated)
    assert (props.barcode = props'.barcode)

let checkDNA (rest:RestAPI) (dna:Reagent) =
    printfn "Checking DNA %s" (dna.id.ToString())
    let dna' = rest.getReagent dna.id
    compareProperties (dna.getProperties) (dna'.getProperties)
    match (dna,dna') with
    | DNA(x),DNA(x') ->
        assert (x.concentration = x'.concentration)
        assert (x.id = x'.id)
        assert (x.sequence = x'.sequence)
        assert (x.Type = x'.Type)
    | _ -> failwithf "ERROR! Both parts must be of type DNA."

let checkRNA (rest:RestAPI) (rna:Reagent) =
    printfn "Checking RNA %s" (rna.id.ToString())
    let rna' = rest.getReagent rna.id
    compareProperties (rna.getProperties) (rna'.getProperties)
    match (rna,rna') with
    | RNA(x),RNA(x') ->
        assert (x.id = x'.id)
        assert (x.sequence = x'.sequence)
        assert (x.Type = x'.Type)
    | _ -> failwithf "ERROR! Both parts must be of type RNA."

let checkChemical (rest:RestAPI) (chemical:Reagent) =
    printfn "Checking Chemical %s" (chemical.id.ToString())
    let chemical' = rest.getReagent chemical.id
    compareProperties (chemical.getProperties) (chemical'.getProperties)
    match (chemical,chemical') with
    | Chemical(x),Chemical(x') ->
        assert (x.id = x'.id)
        assert (x.Type = x'.Type)
    | _ -> failwithf "ERROR! Both parts must be of type Chemical."

let checkProtein (rest:RestAPI) (protein:Reagent) =
    printfn "Checking Protein %s" (protein.id.ToString())
    let protein' = rest.getReagent protein.id
    compareProperties (protein.getProperties) (protein'.getProperties)
    match (protein,protein') with
    | Protein(x),Protein(x') ->
        assert (x.id = x'.id)
        assert (x.isReporter = x'.isReporter)
    | _ -> failwithf "ERROR! Both parts must be of type Protein."

let checkGenericEntity (rest:RestAPI) (ge:Reagent) =
    printfn "Checking Generic Entity %s" (ge.id.ToString())
    let ge' = rest.getReagent ge.id
    compareProperties (ge.getProperties) (ge'.getProperties)
    match (ge,ge') with
    | GenericEntity(x),GenericEntity(x') -> assert (x.id = x'.id)
    | _ -> failwithf "ERROR! Both parts must be of type GenericEntity."

let checkProkaryote (rest:RestAPI) (cell:Cell) =
    printfn "Checking Prokaryotic Cell %s" (cell.id.ToString())
    let cell' = rest.getCell cell.id
    match (cell,cell') with
    | Prokaryote(x),Prokaryote(x') ->
        assert (x.Type = x'.Type)
        assert (x.properties.id = x'.properties.id)
        assert (x.properties.notes = x'.properties.notes)
        assert (x.properties.genotype = x'.properties.genotype)
        assert (x.properties.deprecated = x'.properties.deprecated)
        assert (x.properties.barcode = x'.properties.barcode)
    | _ -> failwithf "ERROR! Both cells must be of type Prokaryote."

let checkCellEntities (rest:RestAPI) (cellId:CellId) (cellEntities:CellEntity[]) =
    printfn "Verifying entities of Cell %s" (cellId.ToString())
    let cellEntities' = rest.getCellEntities cellId
    assert(cellEntities.Length = cellEntities'.Length)
    cellEntities'
    |> Array.iter(fun ce' ->
        let ce =
            cellEntities
            |> Array.tryFind (fun x ->
                (x.entity = ce'.entity) && (x.cellId = ce'.cellId) && (x.compartment = ce'.compartment))
        match ce with
        | Some _ -> ()
        | None -> failwith "Matching cell entity not found")

let checkExperiments (rest:RestAPI) (expt:Experiment) =
    printfn "Checking Experiment %s" (expt.id.ToString())
    let expt' = rest.getExperiment expt.id
    assert (expt.id = expt'.id)
    assert (expt.name = expt'.name)
    assert (expt.notes = expt'.notes)
    assert (expt.deprecated = expt'.deprecated)
    assert (expt.Type = expt'.Type)

let checkSignals (rest:RestAPI) (exptId:ExperimentId) (signals: Signal []) =
    printfn "Checking signals of Experiment %s" (exptId.ToString())
    let signals' = rest.getExperimentSignals exptId
    assert(signals.Length = signals'.Length)
    signals
    |> Array.iter(fun s ->
        let sopt = signals' |> Array.tryFind(fun x -> x.id = s.id)
        match sopt with
        | Some(s') ->
            assert(s.id = s'.id)
            assert(s.units = s'.units)
            match (s.settings,s'.settings) with
            | PlateReaderFluorescence(x),PlateReaderFluorescence(x') ->
                assert((PlateReaderFilter.toString x.emissionFilter) = (PlateReaderFilter.toString x'.emissionFilter))
                assert((PlateReaderFilter.toString x.excitationFilter) = (PlateReaderFilter.toString x'.excitationFilter))
                assert(x.gain = x'.gain)
            | PlateReaderAbsorbance(x),PlateReaderAbsorbance(x') ->
                assert(x.correction = x'.correction)
                assert(x.gain = x'.gain)
                assert(x.wavelength = x'.wavelength)
            | PlateReaderTemperature, PlateReaderTemperature -> ()
            | PlateReaderLuminescence, PlateReaderLuminescence -> ()
            | Titre, Titre -> ()
            | GenericSignal(x),GenericSignal(x') -> assert(x = x')
            | _ -> failwithf "Signal types of signal id: %s does not match" (s.id.ToString())
        | None -> failwithf "ERROR: Signal %s not found." (s.id.ToString()))

let checkExptOperations (rest:RestAPI) (exptId:ExperimentId) (ops:ExperimentOperation []) =
    printfn "Checking operations of Experiment %s" (exptId.ToString())
    let ops' = rest.getExperimentOperations exptId
    assert (ops.Length = ops'.Length)
    ops
    |> Array.iter(fun x ->
        let eopt = ops' |> Array.tryFind (fun eo -> eo.id = x.id)
        match eopt with
        | Some(x') ->
            assert(x.Type = x'.Type)
            assert(x.timestamp = x'.timestamp)
        | None -> failwithf "Experiment Operation %s not found." (x.id.ToString()))

let checkSamples (rest:RestAPI) (exptId:ExperimentId) (samples:Sample []) =
    printfn "Checking samples of Experiment %s" (exptId.ToString())
    let samples' = rest.getExperimentSamples exptId
    assert (samples.Length = samples'.Length)
    samples
    |> Array.iter(fun x ->
        let sopt = samples' |> Array.tryFind (fun s -> s.id = x.id)
        match sopt with
        | Some(x') ->
            assert(x.deprecated = x'.deprecated)
            assert(x.experimentId = x'.experimentId)
            match x.meta,x'.meta with
            | PlateReaderMeta(prm),PlateReaderMeta(prm') ->
                match prm.physicalPlateName, prm'.physicalPlateName with
                | Some(pname), Some(pname') -> assert(pname = pname')
                | None, None -> ()
                | _ -> failwithf "Error! Mismatched Physical plate names for sample %s" (x.id.ToString())
                assert(prm.virtualWell.row = prm'.virtualWell.row)
                assert(prm.virtualWell.col = prm'.virtualWell.col)
                match prm.physicalWell, prm'.physicalWell with
                | Some(pw), Some(pw') ->
                    assert(pw.row = pw'.row)
                    assert(pw.col = pw'.col)
                | None, None -> ()
                | _ -> failwithf "Error! Mismatched Physical well for sample %s" (x.id.ToString())
            | MissingMeta, MissingMeta -> ()
            | _ -> failwithf "ERROR: Sample %s type mismatch" (x.id.ToString())
        | None -> failwithf "ERROR: Sample %s not found" (x.id.ToString())

        )

let checkSampleConditions (rest:RestAPI) (sampleId:SampleId) (conditions:Condition []) =
    printfn "Checking conditions of Sample: %s" (sampleId.ToString())
    let conditions' = rest.getSampleConditions sampleId
    assert (conditions.Length = conditions'.Length)
    let same_time (t:Time option) (t':Time option) =
        match (t,t') with
        | Some(x),Some(x') -> (Time.getHours x) = (Time.getHours x')
        | None, None -> true
        | _ -> false
    conditions
    |> Array.iter(fun x ->
        let copt = conditions' |> Array.tryFind (fun x' ->
            (x.sampleId = x'.sampleId) && (x.reagentId = x'.reagentId) &&
            ((Concentration.toString x.concentration) = (Concentration.toString x'.concentration)) &&
            (same_time x.time x'.time))
        match copt with
        | Some(_) -> ()
        | None -> failwithf "ERROR: Sample %s is missing a condition" (sampleId.ToString()))

let checkSampleDevices (rest:RestAPI) (sampleId:SampleId) (devices:SampleDevice []) =
    printfn "Checking devices of Sample: %s" (sampleId.ToString())
    let devices' = rest.getSampleDevices sampleId
    assert (devices.Length = devices'.Length)
    let check_float (f:float option) (f':float option) =
        match (f,f') with
        | Some(x),Some(x') -> x = x'
        | None,None -> true
        | _ -> false
    devices
    |> Array.iter(fun x ->
        let copt = devices' |> Array.tryFind (fun x' -> (x.cellId.ToString() = x'.cellId.ToString()))
        match copt with
        | Some(x') ->
            assert(check_float x.cellDensity x'.cellDensity)
            assert(check_float x.cellPreSeeding x'.cellPreSeeding)
            assert(x.sampleId = x'.sampleId)
        | None -> failwithf "ERROR: Sample %s is missing a device" (sampleId.ToString()))

let checkTags (rest:RestAPI) (tag_source:EventsProcessor.TagSourceId) (tags:Tag [])=
    printfn "Checking tags associated with Entity %s:" (tag_source.ToString())
    let unique_tags = tags |> Array.distinctBy (fun x -> x.ToString())
    let tags' = (rest.getTags tag_source) |> List.map (fun t -> t.ToString())
    tags |> Array.iter(fun t -> assert (tags' |> List.contains (t.ToString()) ))
    assert (tags'.Length = unique_tags.Length)

let runPartTests (rest:RestAPI) =
    printfn "==========================================================="
    printfn "===============PART TESTS=================================="
    printfn "==========================================================="
    printfn "Adding Promoters:"
    let _ = rest.modifyPart (Entities.promoter1) (RequestType.POST)
    let _ = rest.modifyPart (Entities.promoter2) (RequestType.POST)
    printfn "Adding RBS:"
    let _ = rest.modifyPart (Entities.rbs1) (RequestType.POST)
    printfn "Adding CDS:"
    let _ = rest.modifyPart (Entities.cds1) (RequestType.POST)
    let _ = rest.modifyPart (Entities.cds2) (RequestType.POST)
    printfn "Adding Terminator:"
    let _ = rest.modifyPart (Entities.terminator1) (RequestType.POST)
    printfn "Adding Backbone:"
    let _ = rest.modifyPart (Entities.backbone1) (RequestType.POST)
    printfn "Adding Ori:"
    let _ = rest.modifyPart (Entities.ori1) (RequestType.POST)
    printfn "Adding Scars:"
    let _ = rest.modifyPart (Entities.scar1) (RequestType.POST)
    let _ = rest.modifyPart (Entities.scar2) (RequestType.POST)
    printfn "Adding User Defined Part:"
    let _ = rest.modifyPart (Entities.userdefined1) (RequestType.POST)
    printfn "Adding Linker:"
    let _ = rest.modifyPart (Entities.linker1) (RequestType.POST)
    printfn "Adding Restriction Site:"
    let _ = rest.modifyPart (Entities.restrictionsite1) (RequestType.POST)
    printfn "Finished adding parts"
    printfn "==========================================================="
    let cp = checkPart rest 
    cp Entities.promoter1
    cp Entities.promoter2
    cp Entities.rbs1    
    cp Entities.cds1    
    cp Entities.cds2    
    cp Entities.terminator1    
    cp Entities.backbone1    
    cp Entities.ori1
    cp Entities.scar1
    cp Entities.scar2
    cp Entities.userdefined1
    cp Entities.linker1
    cp Entities.restrictionsite1
    printfn "==========================================================="
    let newuserdefined =
        let newprops = {Entities.userdefined1.getProperties with name = "Deprecated part"; deprecated = true}
        Part.SetProperties Entities.userdefined1 newprops
    let newrestrictionsite =
        let newprops = {Entities.restrictionsite1.getProperties with sequence = "ATGCATGCACACCGTACGTAATGA"}
        Part.SetProperties Entities.restrictionsite1 newprops
    printfn "Modifying User Defined Part %s" (newuserdefined.id.ToString())
    let _ = rest.modifyPart (newuserdefined) (RequestType.PATCH)
    printfn "Modifying Restriction Site Part %s" (newrestrictionsite.id.ToString())
    let _ = rest.modifyPart (newrestrictionsite) (RequestType.PATCH)
    printfn "==========================================================="
    cp newuserdefined
    cp newrestrictionsite
    printfn "==========================================================="
    let terminator1_tags = [|"Untested Terminator";  "Strong Terminator"; "Strong Terminator"; "a"; "b"|] |> Array.map (fun s -> Tag s)
    let terminator1_tagsource = (Entities.terminator1.id |> EventsProcessor.PartTag)
    checkTags rest terminator1_tagsource ([||])
    printfn "Adding tags to Terminator %s" (terminator1_tagsource.ToString())
    let _ = rest.updateTags terminator1_tagsource terminator1_tags (AddRemoveType.ADD)
    checkTags rest terminator1_tagsource terminator1_tags
    let remove_tags = [|"Untested Terminator"; "a"; "b"; "c"|] |> Array.map (fun s -> Tag s)
    printfn "Removing tags from Terminator %s" (terminator1_tagsource.ToString())
    let _ = rest.updateTags terminator1_tagsource remove_tags (AddRemoveType.REMOVE)
    checkTags rest terminator1_tagsource ([|Tag "Strong Terminator"|])
    let add_tags = [|"Tested Terminator"; "Strong Terminator";|] |> Array.map (fun s -> Tag s)
    printfn "Adding tags to Terminator %s" (terminator1_tagsource.ToString())
    let _ = rest.updateTags terminator1_tagsource add_tags (AddRemoveType.ADD)
    checkTags rest terminator1_tagsource add_tags
    printfn "===========================================================\n"

let runReagentTests (rest:RestAPI) =
    printfn "==========================================================="
    printfn "===============REAGENT TESTS==============================="
    printfn "==========================================================="
    printfn "Adding DNA:"
    let _ = rest.modifyReagent (Entities.dna1) (RequestType.POST)
    let _ = rest.modifyReagent (Entities.dna2) (RequestType.POST)
    printfn "Adding RNA:"
    let _ = rest.modifyReagent (Entities.rna1) (RequestType.POST)
    printfn "Adding Chemical:"
    let _ = rest.modifyReagent (Entities.chemical1) (RequestType.POST)
    let _ = rest.modifyReagent (Entities.chemical2) (RequestType.POST)
    printfn "Adding Protein:"
    let _ = rest.modifyReagent (Entities.protein1) (RequestType.POST)
    let _ = rest.modifyReagent (Entities.protein2) (RequestType.POST)
    printfn "Adding Generic Entity:"
    let _ = rest.modifyReagent (Entities.ge1) (RequestType.POST)
    printfn "Finished adding reagents"
    printfn "==========================================================="
    checkDNA rest Entities.dna1
    checkDNA rest Entities.dna2
    checkRNA rest Entities.rna1
    checkChemical rest Entities.chemical1
    checkChemical rest Entities.chemical2
    checkProtein rest Entities.protein1
    checkProtein rest Entities.protein2
    checkGenericEntity rest Entities.ge1
    printfn "==========================================================="
    let newRna =
        match Entities.rna1 with
        | RNA oldrna -> RNA {oldrna with sequence = "UAAGCGCAAUGGUGUGUUUU"; Type = RNAType.GuideRNA}
        | _ -> failwith "Not expected"
    printfn "Modifying RNA Reagent %s" (newRna.id.ToString())
    let _ = rest.modifyReagent (newRna) (RequestType.PATCH)
    checkRNA rest newRna
    printfn "===========================================================\n"

let runCellTests (rest:RestAPI) =
    printfn "==========================================================="
    printfn "===============CELL TESTS=================================="
    printfn "==========================================================="
    printfn "Adding Cells:"
    let _ = rest.modifyCell (Entities.cell0) (RequestType.POST)
    let _ = rest.modifyCell (Entities.cell1) (RequestType.POST)
    printfn "Finished adding cells"
    printfn "==========================================================="
    checkProkaryote rest Entities.cell0
    checkProkaryote rest Entities.cell1
    printfn "==========================================================="
    printfn "Adding Cell Entities:"
    let _ = rest.updateCellEntities cell1.id [|Entities.cell1Entity0|] AddRemoveType.ADD
    checkCellEntities rest cell1.id [|Entities.cell1Entity0|]
    printfn "Adding Cell Entities:"
    let _ = rest.updateCellEntities cell1.id [|Entities.cell1Entity0; Entities.cell1Entity1|] AddRemoveType.ADD
    checkCellEntities rest cell1.id [|Entities.cell1Entity0; Entities.cell1Entity1|]
    printfn "Removing 1 Cell Entity:"
    let _ = rest.updateCellEntities cell1.id [|Entities.cell1Entity0|] AddRemoveType.REMOVE
    checkCellEntities rest cell1.id [|Entities.cell1Entity1|]    
    printfn "===========================================================\n"

let runExperimentTests (rest:RestAPI) =

    printfn "==========================================================="
    printfn "===============EXPERIMENT TESTS============================"
    printfn "==========================================================="
    printfn "Adding Experiments:"
    let _ = rest.modifyExperiment (Entities.buildexpt1) (RequestType.POST)
    let _ = rest.modifyExperiment (Entities.testexpt1) (RequestType.POST)
    printfn "Finished adding cells"
    printfn "==========================================================="
    checkExperiments rest (Entities.buildexpt1)
    checkExperiments rest (Entities.testexpt1)
    printfn "==========================================================="
    //Add Experiment Signals...
    printfn "Adding fluorescence Signals to Experiment: %s" (Entities.testexpt1.id.ToString())
    let _ = rest.updateSignals (Entities.testexpt1.id) Entities.fluorescence_signals (AddRemoveType.ADD)
    checkSignals rest (Entities.testexpt1.id) Entities.fluorescence_signals
    printfn "Adding OD Signals to Experiment: %s" (Entities.testexpt1.id.ToString())
    let _ = rest.updateSignals (Entities.testexpt1.id) Entities.od_signals (AddRemoveType.ADD)
    checkSignals rest (Entities.testexpt1.id) (Entities.od_signals |> Array.append Entities.fluorescence_signals)
    printfn "Removing OD 800 Signal to Experiment: %s" (Entities.testexpt1.id.ToString())
    let _ = rest.updateSignals (Entities.testexpt1.id) [|Entities.od800|] (AddRemoveType.REMOVE)
    checkSignals rest (Entities.testexpt1.id) ([|Entities.mrfp1; Entities.ecfp; Entities.eyfp; Entities.od; Entities.od700|])
    printfn "==========================================================="
    //Add Experiment Operations...
    printfn "Adding Experiment Operations for Build Experiment %s" (Entities.buildexpt1.id.ToString())
    let _ = rest.updateExperimentOperations (Entities.buildexpt1.id) (Entities.build_start) AddRemoveType.ADD
    let _ = rest.updateExperimentOperations (Entities.buildexpt1.id) (Entities.build_end) AddRemoveType.ADD
    checkExptOperations rest Entities.buildexpt1.id Entities.build_ops
    printfn "Adding Experiment Operations for Test Experiment %s" (Entities.testexpt1.id.ToString())
    let _ = rest.updateExperimentOperations (Entities.testexpt1.id) (Entities.test_start) AddRemoveType.ADD
    let _ = rest.updateExperimentOperations (Entities.testexpt1.id) (Entities.test_plate_start) AddRemoveType.ADD
    let _ = rest.updateExperimentOperations (Entities.testexpt1.id) (Entities.test_end) AddRemoveType.ADD
    checkExptOperations rest Entities.testexpt1.id Entities.test_ops
    printfn "==========================================================="
    let expt_tag_source = Entities.testexpt1.id |> EventsProcessor.ExperimentTag
    checkTags rest expt_tag_source ([||])
    let add_tags = [|"Project:BCKG TEST"; "Batch 1"|] |> Array.map (fun s -> Tag s)
    let _ = rest.updateTags expt_tag_source add_tags (AddRemoveType.ADD)
    checkTags rest expt_tag_source add_tags
    let remove_tags = [|"Project:BCKG TEST"|] |> Array.map (fun s -> Tag s)
    printfn "Removing tags from Experiment %s" (expt_tag_source.ToString())
    let _ = rest.updateTags expt_tag_source remove_tags (AddRemoveType.REMOVE)
    checkTags rest expt_tag_source [|Tag "Batch 1"|]
    printfn "===========================================================\n"

let runSampleTests (rest:RestAPI) =
    printfn "==========================================================="
    printfn "===============SAMPLE TESTS================================"
    printfn "==========================================================="

    printfn "Adding samples for Experiment %s" (Entities.testexpt1.id.ToString())
    let _ = rest.addExperimentSamples (Entities.testexpt1.id) (Entities.samples)
    checkSamples rest (Entities.testexpt1.id) (Entities.samples)
    printfn "==========================================================="
    printfn "Adding Sample Conditions"
    Entities.sample_conditions
    |> Array.iter(fun (sid,conditions) ->
        printfn "Adding conditions for Sample %s" (sid.ToString())
        let _ = rest.updateSampleConditions sid conditions AddRemoveType.ADD
        checkSampleConditions rest sid conditions)
    printfn "==========================================================="
    
    printfn "Adding Sample Devices"
    Entities.sample_devices
    |> Array.iter (fun (sid,device) ->
        let _ = rest.updateSampleDevices sid [|device|] AddRemoveType.ADD
        checkSampleDevices rest sid [|device|]
        )
    printfn "===========================================================\n"