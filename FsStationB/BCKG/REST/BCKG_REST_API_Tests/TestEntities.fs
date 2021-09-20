module BCKG.Test.Entities

open BCKG.Domain
open FSharp.Data
open Thoth.Json.Net


let get_entity (res:Result<'A,string>) : 'A =
    match res with
    | Ok(x) -> x
    | Error e -> failwithf "ERROR: %s" e

let getRandomInt (min,max) = System.Random().Next(min,max)
let getRandomValue =
    let a = System.Random().NextDouble() 
    let x = getRandomInt(0,10000) |> float
    a * x

let generateTime: Time option =
    match getRandomInt(0,2) with
    | 0 -> None
    | _ -> getRandomInt(0,20) |> float |> Hours |> Some

let generateFloatOption: float option =
    match getRandomInt(0,2) with
    | 0 -> None
    | _ -> getRandomValue |> float |> Some

let create_meta(i) =
    match i with
    | x when (x <96) ->
        let row = (int i/12)
        let col = i%12
        let pos = {row = row; col = col}
        {virtualWell = pos; physicalWell = pos |> Some; physicalPlateName = "Plate1234" |> Some}
        |> PlateReaderMeta
    | _ -> MissingMeta

let promoter1:Part =
    Promoter({
        id = PromoterId.Create()
        properties = {
            deprecated = false
            sequence = "ATTGGCCCGTACCGAAAGGATATGCCCGAGTT"
            name = "Promoter1"
        }
    })

let promoter2:Part =
    Promoter({
        id = PromoterId.Create()
        properties = {
            deprecated = false
            sequence = "ATTGGGGGGTTTTTTTAAAAAAGGAAGTTGCCGTCATATGCCCGAGTT"
            name = "Promoter2"
        }
    })

let rbs1:Part =
    RBS({
        id = RBSId.Create()
        properties = {
            deprecated = false
            sequence = "TACCTACCCCCTATATATAT"
            name = "RBS1"
        }
    })

let cds1:Part =
    CDS({
        id = CDSId.Create()
        properties = {
            deprecated = false
            sequence = "TTTAAATTTTTTTGGCTATAAGCGCGCCGCCCCATATAAATTTTTTGACGACTTTGTAGCGCGCGCGATCTCTCTCATTTACCTACCCCCTATATATAT"
            name = "CDS1"
        }
    })

let cds2:Part =
    CDS({
        id = CDSId.Create()
        properties = {
            deprecated = false
            sequence = "TTCATATTTGACGACTTTGTAGCGCCTACCCCCTATTTTTTGCGCGGCTATAAGCGGATCTTAAATTAAATTTTTTTCCCTACCCCCTATATATAT"
            name = "CDS2"
        }
    })

let terminator1:Part =
    Terminator({
        id = TerminatorId.Create()
        properties = {
            deprecated = false
            sequence = "GCTACCCCCTATATATATGGATGGGGGGGCC"
            name = "Terminator1"
        }
    })

let scar1:Part =
    Scar({
        id = ScarId.Create()
        properties = {
            deprecated = false
            sequence = "TTATCT"
            name = "Scar1"
        }
    })

let scar2:Part =
    Scar({
        id = ScarId.Create()
        properties = {
            deprecated = false
            sequence = "TCATAT"
            name = "Scar2"
        }
    })

let ori1:Part =
    Ori({
        id = OriId.Create()
        properties = {
            deprecated = false
            sequence = "TTATTGGCCGGGCCGGCGCGTGCCACTCTGACCGTACCGTGACTCAG"
            name = "Ori1"
        }
    })

let backbone1:Part =
    Backbone({
        id = BackboneId.Create()
        properties = {
            deprecated = false
            sequence = "TTCTACCGATTGGCGGTGGTGGACTCTGCCTGATACCGGCCGCCCCACCGGCGCGTGCAG"
            name = "Backbone1"
        }
    })

let linker1:Part =
    Linker({
        id = LinkerId.Create()
        properties = {
            deprecated = false
            sequence = "GACGGCCAGCCAAGCCCCAGACCCAGGGGTGC"
            name = "Linker1"
        }
    })

let restrictionsite1:Part =
    RestrictionSite({
        id = RestrictionSiteId.Create()
        properties = {
            deprecated = false
            sequence = "GGGGGGTTTTTTTT"
            name = "RestrictionSite1"
        }
    })

let userdefined1:Part =
    UserDefined({
        id = UserDefinedId.Create()
        properties = {
            deprecated = false
            sequence = "GAAAGGGAAAGGGAAGGGGT"
            name = "UserDefined1"
        }
    })

let chemical1:Reagent =
    Chemical({
        id = ChemicalId.Create()
        properties = {
            name = "Chemical1"
            notes = ""
            barcode = None
            deprecated = false
        }
        Type = ChemicalType.SmallMolecule
    })

let chemical2:Reagent =
    Chemical({
        id = ChemicalId.Create()
        properties = {
            name = "Chemical2"
            notes = ""
            barcode = None
            deprecated = false
        }
        Type = ChemicalType.SmallMolecule
    })

let dna1:Reagent =
    DNA({
        id = DNAId.Create()
        Type = DNAType.AssembledPlasmidDNA
        concentration = None
        sequence = promoter1.sequence + "ATGC" + promoter2.sequence + rbs1.sequence + scar1.sequence + cds1.sequence + terminator1.sequence + backbone1.sequence
        properties = {
            name = "DNA1"
            notes = ""
            barcode = "FD0001" |> Barcode |> Some
            deprecated = false
        }
    })

let dna2:Reagent =
    DNA({
        id = DNAId.Create()
        Type = DNAType.AssembledPlasmidDNA
        concentration = None
        sequence = promoter2.sequence + "ATGCGGC" + rbs1.sequence +  cds2.sequence + scar2.sequence + terminator1.sequence + scar1.sequence + backbone1.sequence
        properties = {
            name = "DNA2"
            notes = ""
            barcode = "FD0002" |> Barcode |> Some
            deprecated = false
        }
    })

let rna1:Reagent =
    RNA({
        id = RNAId.Create()
        Type = RNAType.SmallRNA
        sequence = "AUGGUGUGUAAGCGCAUUUU"
        properties = {
            name = "RNA1"
            notes = "A completely random RNA."
            barcode = "FD0002" |> Barcode |> Some
            deprecated = false
        }
    })

let protein1:Reagent =
    Protein({
        id = ProteinId.Create()
        isReporter = true
        properties = {
            name = "GFP"
            notes = ""
            barcode = None
            deprecated = false
        }
    })

let protein2:Reagent =
    Protein({
        id = ProteinId.Create()
        isReporter = false
        properties = {
            name = "CDS2-Protein"
            notes = ""
            barcode = None
            deprecated = false
        }
    })

let ge1:Reagent =
    GenericEntity({
        id = GenericEntityId.Create()
        properties = {
            name = "Generic Entity"
            notes = ""
            barcode = None
            deprecated = false
        }
    })

let cell0:Cell =
    Prokaryote({
        properties = {
            id = CellId.Create()
            name = "Cell0"
            notes = ""
            barcode = "FD003" |> Barcode |> Some
            genotype = ""
            deprecated = false
        }
        Type = ProkaryoteType.Bacteria
    })

let cell1:Cell =
    Prokaryote({
        properties = {
            id = CellId.Create()
            name = "Cell1"
            notes = ""
            barcode = "FD004" |> Barcode |> Some
            genotype = ""
            deprecated = false
        }
        Type = ProkaryoteType.Bacteria
    })

let cell1Entity0:CellEntity = {
    cellId = cell1.id
    compartment = CellCompartment.Chromosome
    entity = dna1.id
}

let cell1Entity1:CellEntity = {
    cellId = cell1.id
    compartment = CellCompartment.Plasmid
    entity = dna2.id
}

let create_experiment (name:string) (exptType:ExperimentType) =
    {
        id = ExperimentId.Create()
        name = name
        notes = ""
        deprecated = false
        Type = exptType
    }

let create_experiment_operation (op:ExperimentOperationType,time) :ExperimentOperation=
    {
        id = ExperimentOperationId.Create()
        Type = op
        timestamp = time
    }

let buildexpt1 = create_experiment "Build Experiment 1" ExperimentType.BuildExperiment
let testexpt1 = create_experiment "Test Experiment 1" ExperimentType.TestExperiment

let build_start = create_experiment_operation (ExperimentOperationType.ExperimentStarted, new System.DateTime(2021,1,1,15,30,20))
let build_end = create_experiment_operation (ExperimentOperationType.ExperimentFinished, new System.DateTime(2021,1,1,17,30,8))
let build_ops = [|build_start;build_end|]

let test_start = create_experiment_operation (ExperimentOperationType.ExperimentStarted, new System.DateTime(2021,2,2,15,30,00))
let test_plate_start = create_experiment_operation (ExperimentOperationType.PlateReaderStarted, new System.DateTime(2021,2,2,15,30,00))
let test_end = create_experiment_operation (ExperimentOperationType.ExperimentFinished, new System.DateTime(2021,2,4,10,16,42))
let test_ops = [|test_start; test_plate_start; test_end|]


let create_sample (meta:SampleMeta):Sample=
    {
        id = SampleId.Create()
        experimentId = testexpt1.id
        deprecated = false
        meta = meta
    }

let create_sample_conditions (sampleId:SampleId) =
    let condition1:Condition = {
        reagentId = chemical1.id
        sampleId = sampleId
        concentration = UGML getRandomValue
        time = generateTime
    }
    let condition2:Condition = {
        reagentId = chemical2.id
        sampleId = sampleId
        concentration = M getRandomValue
        time = generateTime
    }
    [|condition1; condition2|]

let create_sample_device (sampleId:SampleId) :SampleDevice=
    {
        cellId = cell1.id
        sampleId = sampleId
        cellDensity = generateFloatOption
        cellPreSeeding = generateFloatOption
    }

let samples =
    [0..100]
    |> List.map(fun i ->  create_sample (create_meta i))
    |> List.toArray

let sample_devices =
    samples
    |> Array.map (fun s ->  (s.id, create_sample_device s.id))

let sample_conditions=
    samples
    |> Array.map (fun s ->  (s.id, create_sample_conditions s.id))

let mrfp1 = {
    id = SignalId.Create();
    settings = PlateReaderFluorescenceSettings.Create(PlateReaderFilter.PlateFilter_550_10,PlateReaderFilter.PlateFilter_610_20,-1.0) |> PlateReaderFluorescence;
    units = None
}

let ecfp = {
    id = SignalId.Create();
    settings = PlateReaderFluorescenceSettings.Create(PlateReaderFilter.PlateFilter_430_10,PlateReaderFilter.PlateFilter_480_10,-1.0) |> PlateReaderFluorescence;
    units = None
}

let eyfp = {
    id = SignalId.Create();
    settings = PlateReaderFluorescenceSettings.Create(PlateReaderFilter.PlateFilter_500_10,PlateReaderFilter.PlateFilter_530,-1.0) |> PlateReaderFluorescence;
    units = None
}

let od = {
    id = SignalId.Create();
    settings = PlateReaderAbsorbanceSettings.Create(600.0,-1.0,-1.0) |> PlateReaderAbsorbance;
    units = None
}

let od700 = {
    id = SignalId.Create();
    settings = PlateReaderAbsorbanceSettings.Create(700.0,-1.0,-1.0) |> PlateReaderAbsorbance;
    units = None
}

let od800 = {
    id = SignalId.Create();
    settings = PlateReaderAbsorbanceSettings.Create(800.0,-1.0,-1.0) |> PlateReaderAbsorbance;
    units = None
}

let fluorescence_signals = [|mrfp1; ecfp; eyfp|]
let od_signals = [|od; od700; od800|]