// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Export

open BCKG.Domain


type KnowledgeGraphSlice = 
    { experimentsMap    : Map<ExperimentId, Experiment>    
      cellsMap          : Map<CellId, Cell>
      signalsMap        : Map<SignalId, Signal>
      experimentSamples : Map<ExperimentId, Sample[]>
      reagentsMap       : Map<ReagentId, Reagent>      
      sampleCells       : Map<SampleId, (CellId * (float option * float option))[]>  //CellDensity * Cell PreSeeding
      sampleConditions  : Map<SampleId, Condition[]>      
      data              : Map<ExperimentId, (Sample*TimeSeries)[]>
     }

     //TODO: improve Async?
     static member Extract (kb:BCKG.API.Instance) (selection: Set<ExperimentId * CellId * SignalId>) =
        async {
            let! expOpts = 
                selection 
                |> Set.toArray                
                |> Array.map (fun (e,_,_) ->  kb.TryGetExperiment e)
                |> Async.Parallel                
            
            let! cellOpts = 
                selection 
                |> Set.toArray
                |> Array.map (fun (_,c,_) ->  kb.TryGetCell c)
                |> Async.Parallel               
                            
            let experimentsMap = 
                expOpts
                |> Array.choose id                
                |> Array.map(fun e -> e.id, e)
                |> Map.ofSeq

            let cellsMap = 
                cellOpts
                |> Array.choose id   
                |> Array.map(fun c -> c.id, c)
                |> Map.ofSeq

            let signalsMap = 
                selection 
                |> Set.map (fun (e,_,_) ->  e |> kb.GetExperimentSignals |> Async.RunSynchronously |> Set.ofArray) 
                |> Set.unionMany
                |> Set.map(fun s -> s.id, s)
                |> Map.ofSeq
       
            let experimentSamples = 
                selection 
                |> Set.map (fun (e,_,_) -> e, kb.GetExperimentSamples e |> Async.RunSynchronously) 
                |> Map.ofSeq

            let samples = 
                experimentSamples
                |> Map.toArray
                |> Array.map snd
                |> Array.concat
                |> Array.distinct

            let sampleCells =
                samples
                |> Array.map(fun sample -> sample.id, kb.GetSampleCells sample.id |> Async.RunSynchronously)
                |> Map.ofSeq

            let sampleConditions = 
                samples
                |> Array.map(fun sample -> sample.id, kb.GetSampleConditions sample.id |> Async.RunSynchronously)
                |> Map.ofSeq

            let reagentsMap = 
                sampleConditions
                |> Map.toArray
                |> Array.map snd
                |> Array.concat
                |> Array.distinct
                |> Array.choose(fun cond -> kb.TryGetReagent cond.reagentId |> Async.RunSynchronously)
                |> Array.map(fun reagent -> reagent.id, reagent)
                |> Map.ofSeq


            let data =
                selection
                |> Set.toArray
                |> Array.map (fun (e, _, _) -> e)
                |> Array.distinct
                |> Array.map (fun e ->
                    let timeSeries = 
                        experimentSamples.[e]
                        |> Array.collect (fun sample ->
                            sample.id 
                            |> kb.GetSampleFiles
                            |> Async.RunSynchronously
                            |> Array.map (fun fid ->
                                let ts =
                                    kb.GetTimeSeriesFile fid.fileId
                                    |> Async.RunSynchronously
                                    |> TimeSeries.fromString
                                (sample, ts)))
                    e,timeSeries)
                |> Map.ofSeq


            let slice = 
                { experimentsMap    = experimentsMap    
                  cellsMap          = cellsMap          
                  signalsMap        = signalsMap        
                  experimentSamples = experimentSamples 
                  reagentsMap       = reagentsMap       
                  sampleCells       = sampleCells       
                  sampleConditions  = sampleConditions  
                  data              = data              
                }
            return slice
            }


let ExportData (knowledgeGraph:BCKG.API.Instance) (selection: Set<ExperimentId * CellId * SignalId>) =
    async {
        let! kb = KnowledgeGraphSlice.Extract knowledgeGraph selection

        let bundle =
            selection
            |> Set.toArray
            |> Array.groupBy (fun (e, _, _) -> e) //experiments are exported as separate CSVs
            |> Array.map (fun (e, L) ->
                let samples = kb.data.[e]

                let signals =
                    L
                    |> Array.map (fun (_, _, s) -> s)
                    |> Array.distinct
                    |> Array.filter (fun s -> samples |> Array.forall (fun (_, D) -> D.observations.ContainsKey s))

                let devices =
                    L
                    |> Array.map (fun (_, d, _) -> d)
                    |> Array.distinct

                let times =
                    Array.replicate signals.Length (snd samples.[0]).times
                    |> Array.concat
                    |> Array.map (Time.getHours >> sprintf "%f")                    
                    |> String.concat ","


                let headers =
                    signals
                    |> Array.collect (fun s ->
                        let name = Signal.toString kb.signalsMap.[s]
                        Array.replicate (snd samples.[0]).times.Length name) //"Raw Data ()"
                    |> String.concat ","

                let exportData =
                    devices
                    |> Array.collect (fun d ->
                        samples
                        |> Array.filter(fun (sample,_) -> Array.exists ((=) d) (Array.map fst kb.sampleCells.[sample.id]))
                        |> Array.map(fun (sample,data) ->                                             
                            let meta = 
                                match sample.meta with 
                                | SampleMeta.PlateReaderMeta meta -> 
                                    match meta.physicalWell with
                                    | Some pos -> sprintf "%c,%i" (System.Convert.ToChar(65 + pos.row)) (pos.col + 1)
                                    | None -> ""
                                | _ -> ""

                            let conditions =
                                kb.sampleConditions.[sample.id]
                                |> Array.map(fun condition -> 
                                    let timing = 
                                        match condition.time with
                                        | Some x -> sprintf "@%s" (Time.toString x)
                                        | None -> ""
                                    sprintf "%s%s=%s" kb.reagentsMap.[condition.reagentId].name timing (Concentration.toString condition.concentration))
                                |> String.concat "; "


                            signals
                            |> Array.map (fun s ->
                                data.observations.[s]
                                |> Array.map (sprintf "%f")
                                |> String.concat ",")
                            |> String.concat ","
                            |> sprintf "%s,,%s,%s,%s" kb.cellsMap.[d].name meta conditions
                            )
                        )                                        
               
                let exportData' = 
                    exportData               
                    |> String.concat "\n"
                    |> sprintf ",,,,,%s\n%s" times
                    |> sprintf "Content,Colony,Well Row,Well Col,Conditions,%s\n%s" headers

                sprintf "%s.csv" (kb.experimentsMap.[e].name.Replace(":", "_").Replace(" ", "_")), exportData')

        return bundle
    }

















(* TODO: 
    - consider devices and map models 
    - Include parts types for fluorescent proteints to account for possible maturation?
    - Generate sets of parameters by applying hypothesis (unique vs repeated between circuits?) (this will replace all the fixed float option parameters)
    - Dilution of species from chemical reactions?
*)

(* - A 'device' is a set of segments;
    - A 'segment' is a set of promoters with the RNA they transcribe
    - RNA transcripts are described as a collection of RBS, PCR pairs
    (PROM * (RBS * PCR) list) list 
    PROM, RBS, PCR are strings (partsID)
*)


(* Notes:
    - Assuming that a (RBS,PCR) pair defines an unique RNA sequence. Is that true?
    - Fixed degratation rate for RNA?
*)

type HillRegulationClass = 
    | Additive
    | Multiplicative
    | Mixed

type ExpressionModelClass = 
    | MassAction 
    | Hill of HillRegulationClass

type GrowthModelClass = NoGrowth | Logistic | LagLogistic 

type HypothesisSettings = 
    { expand_rxn                : bool
    ; fp_maturation             : bool
    ; initials                  : bool
    ; translation               : bool
    ; protein_degradation       : bool
    ; mrna_degradation          : float option
    ; capacity                  : bool
    ; autofluorescence          : bool
    ; backgroundFluorescence    : bool
    ; backgroundAbsorbance      : bool
    ; dilution                  : bool
    ; expression                : ExpressionModelClass   
    ; growth                    : GrowthModelClass
    ; QSSA                      : bool
    }
    static member Default= 
        { expand_rxn          = true
        ; fp_maturation       = true
        ; initials            = true
        ; translation         = false
        ; protein_degradation = true
        ; mrna_degradation    = None
        ; autofluorescence    = false
        ; backgroundFluorescence = true
        ; backgroundAbsorbance   = true
        ; capacity            = true
        ; dilution            = true
        ; expression          = Hill(Mixed)
        ; growth              = LagLogistic
        ; QSSA                = true
        }



let ConditionsToSweep  (conditions:Condition[]) = 
    conditions
    |> Array.map(fun cond -> 
         sprintf "%f" (Concentration.getUM cond.concentration)
        )    



type CrnBuilder = 
    {mutable rates       : Map<string, string>
     mutable parameters  : Map<string, float>
     mutable rxn         : Set<string> 
     mutable species     : Map<string, string>
     mutable paramMap    : Map<string, string>
    }
    static member Create() = 
        { rates       = Map.empty
          parameters  = Map.empty
          rxn         = Set.empty 
          species     = Map.empty
          paramMap    = Map.empty
        }

    member this.CreateParam (name:string)  = 
        if this.paramMap.ContainsKey name then 
            this.paramMap.[name]
        else
            let k = sprintf "k%i" this.paramMap.Count
            this.paramMap <- this.paramMap.Add(name, k)
            this.parameters  <- this.parameters.Add(k,1.0e-3)
            k
    
    member this.CreateSpecies (name:string) = 
        if this.species.ContainsKey name then
            this.species.[name]
        else
            let k = sprintf "s%i" this.species.Count
            this.species <- this.species.Add(name, k)
            k

    member this.AddRxn (rxn:string) = 
        this.rxn <- this.rxn.Add(rxn)

    member this.AddRate (name:string, value:string) = 
        if this.rates.ContainsKey name then 
            if this.rates.[name]<>value then
                failwith "Rate exists with different value"
        else
            this.rates <- this.rates.Add(name, value)
        
    member this.AddParam (name:string, value:float) = 
         if this.parameters.ContainsKey name then 
             if this.parameters.[name]<>value then
                 failwith "Parameter exists with different value"
         else
             this.parameters <- this.parameters.Add(name, value)




type CrnColumn = { name: string; values: float list }
type CrnTable = { times: float list; columns: CrnColumn list }
type Dataset =  { file:string; data:CrnTable list }

let ExportCrn (opt:HypothesisSettings) (kb:BCKG.API.Instance) (cellStrain:CellId) (experiments:ExperimentId[]) = 
    async {    
        let crnBuilder = CrnBuilder.Create()
        let! signals = 
            experiments
            |> Array.map kb.GetExperimentSignals
            |> Async.Parallel
        let selection = 
            Array.zip experiments signals
            |> Array.collect(fun (experimentId, experimentSignals) -> 
                experimentSignals
                |> Array.map(fun signal ->
                    experimentId, cellStrain, signal.id
                    )
                )
            |> Set.ofSeq

        let! kbslice = KnowledgeGraphSlice.Extract kb selection


        //fetch from BCKG
        let samples = kbslice.data |> Map.toArray |> Array.map snd |> Array.concat |> Array.map fst             
        
        let sampleConditions = 
            samples
            |> Array.map(fun sample -> sample.id, kb.GetSampleConditions sample.id |> Async.RunSynchronously)
            |> Map.ofSeq        

        let! interactions = kb.GetInteractions()

        let! cellOpt = kb.TryGetCell cellStrain
        let cell =             
            match cellOpt with
            | Some x -> x
            | None -> failwith "cell strain id not found"

        let! cellEntities = kb.GetCellEntities cellStrain
            
        let! parts = kb.GetParts()
        let partsMap = parts |> Array.map(fun p -> p.id, p) |>  Map.ofSeq 

        //set the order of the samples, which determines the order of the sweeps
        let samplesOrder = sampleConditions |> Map.toArray |> Array.map fst

        let conditionsMap =  //concs in uM
            sampleConditions
            |> Map.toArray
            |> Array.collect(fun (sampleId,conds) -> 
                conds
                |> Array.map(fun cond -> 
                    (sampleId, cond.reagentId), Concentration.getUM cond.concentration
                    )
                )
            |> Map.ofArray

        let condReagents = conditionsMap |> Map.toArray |> Array.map (fst >> snd) |> Array.distinct
        let cellReagents = cellEntities  |> Array.map(fun entity -> entity.entity) |> Array.distinct //TODO: ignoring compartments
        let interReagents = 
            interactions 
            |> Array.map(fun i -> 
                match i with 
                | CodesFor          props -> [ProteinReagentId props.protein]
                | GeneticActivation props -> props.activator
                | GenericActivation props -> props.regulator
                | GeneticInhibition props -> props.inhibitor
                | GenericInhibition props -> props.regulator
                | Reaction          props -> List.concat [props.enzyme; (List.concat props.products); (List.concat props.reactants)]
                )
            |> List.concat
            |> List.toArray
            |> Array.distinct


        //Expand reactions
        let reactions, reagentIds =
            let toSet x = x |> Set.ofSeq |> Set.map Set.ofSeq
            let toSetE x = x |> Set.ofSeq |> Set.singleton
            

            let kgrxn = 
                interactions
                |> Array.choose(fun i -> 
                    match i with
                    | Reaction props -> Some props
                    | _ -> None
                    )
                |> Set.ofSeq

            let mutable rxnReagents =
                kgrxn
                |> Set.map(fun i -> 
                    Set.unionMany [|toSet i.products; toSet i.reactants; toSetE i.enzyme|]
                    )
                |> Set.unionMany
                |> Set.union (Array.concat [condReagents; cellReagents; interReagents] |> Array.map Set.singleton |> Set.ofSeq)


            let canFire (i:ReactionInteraction) = 
                (Set.isSubset (toSet i.reactants) rxnReagents) &&  //all reactants are in the system
                (Set.isSubset (toSetE i.enzyme) rxnReagents)     //all enzymes are in the system


            let mutable sysrxn = kgrxn |> Set.filter canFire

            if opt.expand_rxn then        
                let mutable expanded = false
                while (not expanded) do
                    expanded <- true
                    let newReagents = (sysrxn |> Set.map(fun i -> toSet i.products) |> Set.unionMany) - rxnReagents
                    if not (Set.isEmpty newReagents) then
                        expanded <- false
                        rxnReagents <- rxnReagents + newReagents
                        sysrxn <- kgrxn |> Set.filter canFire
                
            Set.toArray sysrxn, Set.unionMany rxnReagents |> Set.toArray //Note: individual reacents is not the same as complexes (sets of reagents)

        let! reagents = 
            reagentIds
            |> Array.map kb.TryGetReagent
            |> Async.Parallel

        let reagentsMap  = reagents |> Array.choose id |> Array.map (fun r -> r.id, r) |> Map.ofSeq
        let reagentNames = reagentsMap |> Map.map (fun _ r -> r.name)
                     
        let complexToName x = 
            x 
            |> List.map(fun r -> crnBuilder.CreateSpecies reagentNames.[r]) 
            |> List.sortBy id 
            |> String.concat "_"
        

        //Add all reactions 
        reactions
        |> Array.iteri (fun k i ->
            let reactants = i.reactants |> List.map complexToName |> String.concat " + "
            let products  = i.products  |> List.map complexToName |> String.concat " + "

            let enzyme =  if List.isEmpty i.enzyme then "" else sprintf "%s ~" (complexToName i.enzyme) 
            let rate = crnBuilder.CreateParam (sprintf "ReactionRate%i" k)
            crnBuilder.AddRxn (sprintf "%s %s ->{%s} %s" enzyme reactants rate products)
            )
 
        
        //Device is given as a list of parts
        let FindTranslations (promoterId:PromoterId) (device:PartInstance[]) = 
            device
            |> Array.mapi(fun i part ->                                                         
                match partsMap.[part.id] with 
                | RBS rbs -> 
                    device.[i..] 
                    |> Array.choose (fun part' -> 
                        match partsMap.[part'.id] with 
                        | CDS cds -> Some (promoterId, rbs.id, cds.id)
                        | _ -> None
                        )                               
                | _ -> Array.empty
                )
            |> Array.concat
            
        ////work with each segment in isolation first (need to add regulation reactions later)
        ////returns a list of promoter * parts expressed from the promoter (until a terminator)
        let FindTranscriptions (device:PartInstance[]) = 
            device
            |> Array.mapi(fun i part ->                                                         
                match partsMap.[part.id] with 
                | Promoter prom -> 
                    let terIdOpt = device.[i..] |> Array.tryFindIndex (fun part' -> match partsMap.[part'.id] with Terminator _ -> true | _ -> false)
                    match terIdOpt with
                    | Some j -> Some (prom, device.[i+1..i+j-1]) //drop terminators from the list                                
                    | None   -> Some (prom, device.[i+1..])
                | _ -> None
                )
            |> Array.choose id    
            |> Array.map(fun (prom, transcribed) -> FindTranslations prom.id transcribed)
            |> Array.concat
        
        
        let expressionCassettes = 
            cellReagents 
            |> Array.map(fun r ->
                match reagentsMap.[r] with 
                | DNA dna ->                      
                    Reagent.AnnotateSequence partsMap dna.sequence 
                    |> Array.ofSeq                    
                    |> FindTranscriptions                    
                | _ -> Array.empty
                )
            |> Array.concat
        

        let codesFor = 
            interactions 
            |> Array.choose (fun interaction -> 
                match interaction with 
                | CodesFor codes ->  
                    Some (codes.cds, codes.protein)
                | _ -> None
                )
            |> Map.ofSeq
            

        let fluorescentProteins = 
            [ "RFP Protein", "RFP" 
              "eCFP Protein", "CFP"
              "eYFP Protein", "YFP"
            ] 
            |> Map.ofSeq


        let mutable fluoros = Set.empty

        expressionCassettes
        |> Array.iter(fun (prom, rbs, cds) -> 
                let proteinReagent = 
                    if codesFor.ContainsKey cds then 
                        reagentNames.[ProteinReagentId codesFor.[cds]]
                    else
                        "UnknownProtein"

                let rate =  crnBuilder.CreateParam (sprintf "Expression(%A,%A,%A)" partsMap.[PromoterPartId prom].name partsMap.[RBSPartId rbs].name partsMap.[CDSPartId cds].name)
                let rna = crnBuilder.CreateSpecies (sprintf "RNA(%A,%A)" partsMap.[RBSPartId rbs].name partsMap.[CDSPartId cds].name)
                let protein = crnBuilder.CreateSpecies (sprintf "Protein(%A)" proteinReagent)
                    
                if opt.translation then
                    crnBuilder.AddRxn(sprintf "%s ->{%s} %s" rna rate protein)           //Translation of mRNA

                match opt.mrna_degradation with
                | Some k -> crnBuilder.AddRxn(sprintf " ->{%g} %s" k rna)
                | None -> ()

                if opt.protein_degradation then 
                    let deg = crnBuilder.CreateParam (sprintf "Degradation(%A)" proteinReagent)
                    crnBuilder.AddRxn(sprintf "->{%s} %s" deg protein)        

                if opt.dilution then        //Protein dilution
                    crnBuilder.AddRxn(sprintf "%s -> [[growth]*[%s]]" protein protein)
                        
                if opt.dilution && opt.translation then  //mRNA dilution
                    crnBuilder.AddRxn(sprintf "%s -> [[growth]*[%s]]" rna rna)


                if fluorescentProteins.ContainsKey proteinReagent then
                    let fluoro = protein //fluorescentProteins.[proteinReagent] //TODO: use readable names?
                    fluoros <- fluoros.Add(fluoro)

                    if opt.autofluorescence then 
                        let kAuto = crnBuilder.CreateParam (sprintf "Auto-fluorescence(%s)" proteinReagent)
                        if opt.capacity then
                            crnBuilder.AddRxn(sprintf "->[[capacity] * %s] %s" kAuto fluoro)
                        else
                            crnBuilder.AddRxn(sprintf "->{%s} %s" kAuto fluoro)

                        if opt.fp_maturation then                
                            failwith "Not currently implemented"
                        //    rxn <- rxn.Add([],[x],[[fl]],kMat)       //maturation                        
                        //    if opt.protein_degradation then 
                        //        rxn <- rxn.Add([],[[fl]],[],k)       //mature protein degradation (assume same rate as immature fp)     
                        //    p
                        //else                                                
                        //    GecDB.PCR [GecDB.CODES([fl],k)] //replace part so that the mature FP is produced directly
                
                let activators = 
                    interactions
                    |> Array.choose(fun i -> 
                        match i with 
                        | GeneticActivation act -> 
                            if act.activated = prom then
                                Some act.activator
                            else
                                None
                        | _ -> None
                        )
                
                let repressors = 
                    interactions
                    |> Array.choose(fun i -> 
                        match i with 
                        | GeneticInhibition rep -> 
                            if rep.inhibited = prom then
                                Some rep.inhibitor
                            else
                                None
                        | _ -> None
                        )
                               
                //Transcription (or combined expression)
                match opt.expression with
                | MassAction ->   
        //            let protein = parts |> List.map(fun (_,_,_,p) -> p)
        //            let props = match partsDB.[prom] with GecDB.PROM(p) -> p | _ -> failwith "Should be PROM brick"
        //            let rna = parts |> List.map(fun (_,_,r,_) -> [r])
        //            let product = if opt.translation then  rna else protein //make either RNA or protein
        //            props
        //            |> List.iter(fun p -> 
        //                match p with 
        //                | GecDB.POS(X,kb,ku,ke)                      
        //                | GecDB.NEG(X,kb,ku,ke) ->
        //                        rxn <- rxn.Add([],[X; [prom]],[prom::X],kb)     //binding
        //                        rxn <- rxn.Add([],[prom::X],[X; [prom]],ku)     //unbinding                                                        
        //                        if opt.capacity then
        //                            let bp = sp2str (prom::X)
        //                            string_rxn <- string_rxn.Add(sprintf "%s ->[[capacity] * %s] %s %s" bp (ke.ToString()) bp (product |> List.map sp2str |> String.concat " + "))
        //                        else
        //                            rxn <- rxn.Add([prom::X],[],product,ke)         //transcription or tx/tl                            
        //                | GecDB.CON(kb)         ->  
        //                        if opt.capacity then                                
        //                            string_rxn <- string_rxn.Add(sprintf "%s ->[[capacity] * %s] %s %s" prom (kb.ToString()) prom (product |> List.map sp2str |> String.concat " + "))
        //                        else
        //                            rxn <- rxn.Add([[prom]],[],product,kb)          //basal transcription or tx/tl
        //                ))) 
                    failwith "TODO" 

                    
        
                | Hill t ->
                    let product = if opt.translation then rna else protein //make either RNA or protein
                    let promSpecies = crnBuilder.CreateSpecies (sprintf "PromoterDNA(%s)" partsMap.[PromoterPartId prom].name)
                    let rate_name = sprintf "a_%s" promSpecies

                    let pos_rate_law x a k n = sprintf "(%s*[%s]^%s/([%s]^%s+%s^%s))" a x n x n k n
                    let neg_rate_law x a k n = sprintf "(%s*%s^%s/([%s]^%s+%s^%s))" a k n x n k n
                      
                    let base_rate = crnBuilder.CreateParam (sprintf "BasalExpression(%A)" prom)
                    let pos_rates = 
                        activators
                        |> Array.map(fun complex -> 
                            let activator = complexToName complex
                            let k = crnBuilder.CreateParam (sprintf "Sensitity(%A,%A)" activator prom    )
                            let n = crnBuilder.CreateParam (sprintf "Cooperativity(%A,%A)" activator prom)
                            let a = crnBuilder.CreateParam (sprintf "Activity(%A,%A)" activator prom     )
                            pos_rate_law activator a k n
                            )
                    let neg_rates = 
                        repressors
                        |> Array.map(fun complex -> 
                            let repressor = complexToName complex
                            let k = crnBuilder.CreateParam (sprintf "Sensitity(%A,%A)" repressor prom    )
                            let n = crnBuilder.CreateParam (sprintf "Cooperativity(%A,%A)" repressor prom)
                            let a = crnBuilder.CreateParam (sprintf "Activity(%A,%A)" repressor prom     )
                            neg_rate_law repressor a k n
                            )

                    let condMergeStr (A:string) (B:string) c = 
                        match A.Trim(), B.Trim() with
                        | "", "" -> ""
                        | "", x  | x, "" -> x
                        | x, y -> sprintf "(%s) %s (%s)" x c y 

                    let reg_rate = 
                        match t with                     
                        | Additive       -> Array.append pos_rates neg_rates |> String.concat " + " 
                        | Multiplicative -> Array.append pos_rates neg_rates |> String.concat " * " 
                        | Mixed          -> condMergeStr (pos_rates |> String.concat " + ") (neg_rates |> String.concat " * ") " * "              

                    let total_rate = condMergeStr base_rate reg_rate " + "

                    crnBuilder.AddRate(rate_name, total_rate)

                    let scale = if opt.capacity then "[capacity] * " else ""
                    let transcriptionRxn = sprintf "%s -> [%s[%s]] %s + %s" promSpecies scale rate_name promSpecies product
                    crnBuilder.AddRxn(transcriptionRxn)                    

                )
                

        let dataBundle = 
            let dataMap = 
                kbslice.data
                |> Map.map (fun _ expData -> expData |> Array.map(fun (sample,data) -> sample.id, data) |> Map.ofSeq)

            experiments
            |> Array.map(fun experimentId -> 
                let data = 
                    samplesOrder 
                    |> Array.map(fun sampleId -> 
                        let ts = dataMap.[experimentId].[sampleId]
                        let cols = 
                            ts.observations
                            |> Map.toArray
                            |> Array.map(fun (signalId, vals) ->
                                      
                                { CrnColumn.name = sprintf "%A" signalId
                                  CrnColumn.values = List.ofArray vals
                                }
                                )

                        { CrnTable.times = ts.times |> Array.map (fun x -> 60.0*(Time.getHours x)) |> List.ofArray
                          CrnTable.columns = List.ofArray cols
                        })

                { Dataset.file = sprintf "%A" experimentId
                  Dataset.data = List.ofArray data
                })

        let maxTime = dataBundle |> Array.map(fun d -> d.data |> List.map (fun table -> List.max table.times) |> List.max) |> Array.max

        //Growth model
        match opt.growth with
        | NoGrowth -> 
            crnBuilder.AddRate("growth", "0.0")
        | Logistic -> 
            crnBuilder.AddParam("growRate", 1.0)
            crnBuilder.AddParam("popCap", 1.0)
            crnBuilder.AddRate("growth", "growRate*(1 - [OD] / popCap)")
            crnBuilder.AddRxn("->[[growth]*[OD]] OD")
        | LagLogistic -> 
            crnBuilder.AddParam("growRate", 1.0)
            crnBuilder.AddParam("popCap", 1.0)
            crnBuilder.AddParam("tlag", 1e-3)
            crnBuilder.AddRxn("init grow 1 @ tlag")    
            crnBuilder.AddRate("growth", "[grow]*growRate*(1 - [OD] / popCap)")
            crnBuilder.AddRxn("->[[growth]*[OD]] OD")        
               

        crnBuilder.AddRate("capacity", "1.0")
        crnBuilder.AddParam("backOD", 1e-3)

        let printMap name (x:Map<string,string>) =
            x 
            |> Map.toArray 
            |> Array.sortBy snd
            |> Array.map(fun (a,b) -> 
                let spaces = String.replicate (20-b.Length) "."
                sprintf "// %s%s%s" b spaces a) 
            |> String.concat "\n"
            |> sprintf "\n\n// ***************%s******************\n%s" name

        let plot_directive =         
            if opt.backgroundAbsorbance then
                crnBuilder.AddRate("ObsOD","[OD]+backOD")
            else
                crnBuilder.AddRate("ObsOD","[OD]")
            fluoros
            |> Set.toArray
            |> Array.map(fun fl ->     
                let ObsName = sprintf "Obs%s" fl
                let bg = crnBuilder.CreateParam (sprintf "%s background fluorescence" fl)
                if opt.backgroundFluorescence then             
                    crnBuilder.AddRate(ObsName,sprintf "[OD]*[%s]+%s" fl bg)
                else
                    crnBuilder.AddRate(ObsName,sprintf "[OD]*[%s]" fl)
                ObsName
                )
            |> Array.append [|"ObsOD"|]
            |> String.concat ";"
            |> sprintf "directive simulator deterministic\ndirective simulation {initial=0.0; final=%f; points=200; plots=[%s]}\n" maxTime
    
        //Initial conditions     
        if opt.initials then 
            let od0 = crnBuilder.CreateParam "OD_0"
            crnBuilder.AddRxn(sprintf "init OD %s" od0)

            crnBuilder.species
            |> Map.toArray
            |> Array.iter(fun (_,s) ->                
                let init = crnBuilder.CreateParam (sprintf "%s_0" s)          
                crnBuilder.AddRxn(sprintf "init %s %s" s init)
                )
            
        let rates_directive = 
            crnBuilder.rates
            |> Map.toArray
            |> Array.map(fun (n,v) -> sprintf "\t%s = %s" n v)
            |> String.concat ";\n"
            |> sprintf "directive rates [\n%s\n]\n"

        let parameters_directive = 
            crnBuilder.parameters 
            |> Map.toArray 
            |> Array.map(fun (n,v) -> sprintf "\t%s = %f, {distribution=Uniform(1E-05, 1E5); interval=Log; variation=Random}" n v)
            |> String.concat ";\n"
            |> sprintf "directive parameters [\n%s\n]\n"

        let sweeps_directive = 
            let vars = condReagents |> Array.map (fun x -> reagentNames.[x]) |> String.concat "," |> sprintf "(%s)"
            samplesOrder
            |> Array.map(fun sampleId ->     
                condReagents
                |> Array.map(fun reagentId -> 
                    let key = (sampleId, reagentId)
                    let conc = if conditionsMap.ContainsKey key then conditionsMap.[key] else 0.0                    
                    sprintf "%g" conc
                    )
                |> String.concat ","
                |> sprintf "(%s)"     
                )
            |> String.concat ";"
            |> sprintf "directive sweeps [autogenerated = [%s = [%s]]]" vars

    

        let speciesDefs = printMap "SPECIES" crnBuilder.species
        let paramDefs   = printMap "PARAMETERS" crnBuilder.paramMap
        let model = 
            crnBuilder.rxn 
            |> Set.toArray 
            |> String.concat " |\n"    
            |> sprintf "%s\n%s\n" rates_directive 
            |> sprintf "%s\n%s\n" parameters_directive      
            |> sprintf "%s\n%s\n" sweeps_directive
            |> sprintf "%s\n%s\n" plot_directive      
            |> sprintf "%s\n%s\n" speciesDefs
            |> sprintf "%s\n%s\n" paramDefs




      
        return (model, dataBundle)
    }




(*

    if opt.QSSA then 
        //Note that with QSSA the forward and backwards rates area replaced by a single parameters (kept to be the forward rate)
        //e.g. A + B<->{k,k'} AB becomes AB = k*A*B (where k = k/k')

        let rxn', rate' = 
            [ for (catalysts, reactants, products, rate) in rxn do  //TODO: Search more efficiently
                let isRev = Set.exists (fun (catalysts', reactants', products', _) -> catalysts = catalysts' && reactants = products'  && products = reactants') rxn
                let RewriteLaw (R:string list list) (P:string list list) = 
                    let complex = sp2str P.Head
                    let components = R |> List.map sp2str |> List.map (sprintf "[%s]")
                    let cats = catalysts |> List.map sp2str |> List.map (sprintf "[%s]")
                    complex, rate.ToString()::components@cats |> String.concat "*" //rate definition                    
                
                let rd = catalysts|> Set.ofSeq, reactants|> Set.ofSeq, products|> Set.ofSeq
                if reactants.Length = 1 && products.Length <> 1 && isRev then                     
                    yield [rd], [RewriteLaw products reactants]
                elif reactants.Length <> 1 && products.Length = 1 && isRev then
                    yield [rd], [RewriteLaw reactants products]                
            ]
            |> List.unzip          
        
        let rrxn = 
            let rxn' = rxn' |> List.concat |> Set.ofSeq
            rxn 
            |> Set.filter (fun (catalysts, reactants, products, _ ) -> 
                let rd = (catalysts |> Set.ofSeq, reactants |> Set.ofSeq, products |> Set.ofSeq) 
                let rd' = (catalysts |> Set.ofSeq, products |> Set.ofSeq, reactants |> Set.ofSeq)                 
                not (rxn'.Contains rd) && not (rxn'.Contains rd')) 
            |> Set.ofSeq
        rxn <- rrxn
        rates <- rate' |> List.concat |> List.fold(fun acc x -> acc.Add x) rates

        
       
    //Initial conditions     
    if opt.initials then 
        species
        |> Set.filter (sp2str >> rates.ContainsKey >> not)
        |> Set.iter(fun s -> 
            let s' = sp2str s
            let init = sprintf "%s_0" s'            
            string_rxn <- string_rxn.Add(sprintf "init %s %s" s' init)
            
            if s.Head = "OD" then 
                parameters <- parameters.Add(init, 0.002)
            else
                parameters <- parameters.Add(init, 0.0)
            )
           

 
*)