namespace BCKG.Domain

open System
#if FABLE_COMPILER
open Thoth.Json
#else
open Thoth.Json.Net
#endif

type Barcode = Barcode of string
    with
    override this.ToString() = match this with Barcode x -> x
    static member toString (Barcode x) = x
    

type Tag = Tag of string 
    with
    override this.ToString() = match this with Tag x -> x
    static member toString (Tag x) = x


type ExperimentId = ExperimentId of System.Guid
    with
    member this.guid = let (ExperimentId guid) = this in guid
    override this.ToString() = match this with ExperimentId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ExperimentId
    static member toString (ExperimentId x) = x.ToString()
    static member fromString (x:string) = 
        try 
            Some (System.Guid.Parse x |> ExperimentId)
        with 
            | _  -> None            

type DNAId = DNAId of System.Guid
    with
    override this.ToString() = match this with DNAId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> DNAId

type RNAId = RNAId of System.Guid
    with
    override this.ToString() = match this with RNAId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> RNAId

type ChemicalId = ChemicalId of System.Guid
    with
    override this.ToString() = match this with ChemicalId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ChemicalId

type ProteinId = ProteinId of System.Guid
    with
    override this.ToString() = match this with ProteinId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ProteinId

type GenericEntityId = GenericEntityId of System.Guid
    with
    override this.ToString() = match this with GenericEntityId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> GenericEntityId

type ReagentId = 
    | DNAReagentId of DNAId
    | RNAReagentId of RNAId
    | ProteinReagentId of ProteinId
    | ChemicalReagentId of ChemicalId
    | GenericEntityReagentId of GenericEntityId
    with
    override this.ToString() = 
        match this with 
        | DNAReagentId x -> x.ToString()
        | RNAReagentId x -> x.ToString()
        | ProteinReagentId x -> x.ToString()
        | ChemicalReagentId x -> x.ToString()
        | GenericEntityReagentId x -> x.ToString()
    static member toString (rid:ReagentId) = rid.ToString()
    
    static member GetType (rid:ReagentId) =
        match rid with 
        | DNAReagentId _ -> "DNA"
        | RNAReagentId _ -> "RNA"
        | ProteinReagentId _ -> "Protein"
        | ChemicalReagentId _ -> "Chemical"
        | GenericEntityReagentId _ -> "GenericEntity"
    static member FromType (guid:System.Guid) (str:string) = 
        match str with 
        | "DNA" -> guid |> DNAId |> DNAReagentId 
        | "RNA" -> guid |> RNAId |> RNAReagentId
        | "Chemical" -> guid |> ChemicalId |> ChemicalReagentId
        | "Protein" -> guid |> ProteinId |> ProteinReagentId
        | "GenericEntity" -> guid |> GenericEntityId |> GenericEntityReagentId
        | _ -> failwithf "Unknown ReagentId Type: %s" (str)
    
    member this.guid = 
        match this with 
        | DNAReagentId (DNAId guid) -> guid 
        | RNAReagentId (RNAId guid) -> guid
        | ChemicalReagentId (ChemicalId guid) -> guid
        | ProteinReagentId (ProteinId guid) -> guid
        | GenericEntityReagentId (GenericEntityId guid) -> guid
        

type FileId = FileId of System.Guid
    with
    override this.ToString() = match this with FileId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> FileId
    static member toString (FileId x) = x.ToString()
    static member fromString (x:string) = 
        try 
            Some (System.Guid.Parse x |> FileId)
        with 
            | _  -> None            
    static member addFile = "++file"
    static member addFiles = "++files"
    
type CDSId = CDSId of System.Guid
    with
    override this.ToString() = match this with CDSId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> CDSId
    

type PromoterId = PromoterId of System.Guid      
    with
    override this.ToString() = match this with PromoterId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> PromoterId

type TerminatorId = TerminatorId of System.Guid    
    with
    override this.ToString() = match this with TerminatorId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> TerminatorId

type UserDefinedId = UserDefinedId of System.Guid
    with
    override this.ToString() = match this with UserDefinedId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> UserDefinedId

type ScarId = ScarId of System.Guid           
    with
    override this.ToString() = match this with ScarId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ScarId

type RBSId = RBSId of System.Guid           
    with
    override this.ToString() = match this with RBSId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> RBSId

type BackboneId = BackboneId of System.Guid       
    with
    override this.ToString() = match this with BackboneId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> BackboneId

type OriId = OriId of System.Guid           
    with
    override this.ToString() = match this with OriId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> OriId

type LinkerId = LinkerId of System.Guid        
    with
    override this.ToString() = match this with LinkerId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> LinkerId

type RestrictionSiteId = RestrictionSiteId of System.Guid
    with
    override this.ToString() = match this with RestrictionSiteId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> RestrictionSiteId

type ObservationId = ObservationId of System.Guid
    with
    override this.ToString() = match this with ObservationId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ObservationId


type PartId = 
    | CDSPartId              of CDSId
    | PromoterPartId         of PromoterId
    | TerminatorPartId       of TerminatorId
    | UserDefinedPartId      of UserDefinedId
    | RBSPartId              of RBSId
    | ScarPartId             of ScarId
    | BackbonePartId         of BackboneId
    | OriPartId              of OriId
    | LinkerPartId           of LinkerId
    | RestrictionSitePartId  of RestrictionSiteId
    with
    override this.ToString() = 
        match this with 
        | CDSPartId              x -> x.ToString()
        | PromoterPartId         x -> x.ToString()
        | TerminatorPartId       x -> x.ToString()
        | UserDefinedPartId      x -> x.ToString()
        | RBSPartId              x -> x.ToString()
        | ScarPartId             x -> x.ToString()
        | BackbonePartId         x -> x.ToString()
        | OriPartId              x -> x.ToString()
        | LinkerPartId           x -> x.ToString()
        | RestrictionSitePartId  x -> x.ToString()
    member this.guid = (this.ToString() |> System.Guid)
    static member GetType (pid:PartId) = 
        match pid with 
        | CDSPartId              x -> "CDS"
        | PromoterPartId         x -> "Promoter"
        | TerminatorPartId       x -> "Terminator"
        | UserDefinedPartId      x -> "UserDefined"
        | RBSPartId              x -> "RBS"
        | ScarPartId             x -> "Scar"
        | BackbonePartId         x -> "Backbone"
        | OriPartId              x -> "Ori"
        | LinkerPartId           x -> "Linker"
        | RestrictionSitePartId  x -> "RestrictionSite"
    static member FromType (guid:System.Guid) (str:string) = 
        match str with 
        | "CDS"             -> guid |> CDSId             |> CDSPartId             
        | "Promoter"        -> guid |> PromoterId        |> PromoterPartId       
        | "Terminator"      -> guid |> TerminatorId      |> TerminatorPartId     
        | "UserDefined"     -> guid |> UserDefinedId     |> UserDefinedPartId    
        | "RBS"             -> guid |> RBSId             |> RBSPartId            
        | "Scar"            -> guid |> ScarId            |> ScarPartId           
        | "Backbone"        -> guid |> BackboneId        |> BackbonePartId       
        | "Ori"             -> guid |> OriId             |> OriPartId            
        | "Linker"          -> guid |> LinkerId          |> LinkerPartId         
        | "RestrictionSite" -> guid |> RestrictionSiteId |> RestrictionSitePartId
        | _ -> failwithf "Unknown PartId type: %s" str

    static member toString (pid:PartId) = pid.ToString()


type ExperimentOperationId = ExperimentOperationId of System.Guid
    with
    override this.ToString() = match this with ExperimentOperationId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ExperimentOperationId
    static member toString (ExperimentOperationId x) = x.ToString()
    static member fromString (x:string) =
        try 
            Some (System.Guid.Parse x |> ExperimentOperationId)
        with 
            | _  -> None            

type ReplicateId = ReplicateId of System.Guid
    with
    override this.ToString() = match this with ReplicateId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> ReplicateId
    static member toString (ReplicateId x) = x.ToString()
    static member fromString (x:string) = 
        try 
            Some (System.Guid.Parse x |> ReplicateId)
        with
            | _ -> None

type SampleId = SampleId of System.Guid
    with
    member this.guid = let (SampleId guid) = this in guid
    override this.ToString() = match this with SampleId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> SampleId
    static member toString (SampleId x) = x.ToString()
    static member fromString (x:string) =
        try 
            Some (System.Guid.Parse x |> SampleId)
        with 
            | _  -> None            

type SignalId = SignalId of System.Guid
    with
    member this.guid = let (SignalId guid) = this in guid
    override this.ToString() = match this with SignalId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> SignalId
    static member toString (SignalId x) = x.ToString()
    static member fromString (x:string) =
        try 
            Some (System.Guid.Parse x |> SignalId)
        with 
            | _  -> None            

type InteractionId = InteractionId of System.Guid
    with 
    override this.ToString() = match this with InteractionId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> InteractionId
    static member toString (InteractionId x) = x.ToString()
    static member fromString (x:string) = 
        try 
            Some (System.Guid.Parse x |> InteractionId)
        with 
            | _  -> None 

type CellId = CellId of System.Guid
    with
    member this.guid = let (CellId guid) = this in guid
    override this.ToString() = match this with CellId x -> x.ToString()
    static member Create() = System.Guid.NewGuid() |> CellId
    static member toString (CellId x) = x.ToString()
    static member fromString (x:string) = 
        try 
            Some (System.Guid.Parse x |> CellId)
        with 
            | _  -> None 

//type TagId = TagId of System.Guid
//    with
//    override this.ToString() = match this with TagId x -> x.ToString()
//    static member Create() = System.Guid.NewGuid() |> TagId
//    static member toString (TagId x) = x.ToString()
//    static member fromString (x:string) =
//        let result = ref (System.Guid.NewGuid())
//        let flag = System.Guid.TryParse(x, result)
//        if flag then Some (!result |> TagId) else None


//type Tag = 
//    | Project of string
//    | Keyword of string


   
type Concentration = 
    | NM of float //nM
    | UM of float //uM
    | MM of float //mM    
    | M  of float //M
    | X  of float //X
    | NGUL of float //ng/uL  //TODO: implement as compound of weight and volume?
    | NGML of float //ng/mL  //TODO: implement as compound of weight and volume?
    | UmL of float  //U/mL   //TODO: implement as compound?
    | ULML of float //uL/mL
    | UGML of float //ug/mL
    | PERC of float //

    static member getValue (conc:Concentration) = 
        match conc with 
        | NM x  
        | UM x
        | MM x
        | M  x
        | X  x
        | NGUL x
        | NGML x
        | UmL x 
        | ULML x -> x
        | UGML x -> x
        | PERC x -> x
    static member getUnit (conc:Concentration) = 
        match conc with 
        | NM   _ -> "nM"
        | UM   _ -> "uM"
        | MM   _ -> "mM"
        | M    _ -> "M"
        | X    _ -> "X"
        | NGUL _ -> "ng/uL"
        | NGML _ -> "ng/mL"
        | UGML _ -> "ug/mL"
        | UmL  _ -> "U/mL"
        | ULML _ -> "uL/mL"
        | PERC _ -> "%"
    static member Create (x:float) (unit:string) = 
        match unit with 
        | "nM"    -> NM x
        | "uM"    -> UM x
        | "mM"    -> MM x
        | "M"     -> M x
        | "X"     -> X x
        | "ng/uL" -> NGUL x
        | "ng/mL" -> NGML x
        | "U/mL"  -> UmL x
        | "uL/mL" -> ULML x
        | "ug/mL" -> UGML x
        | "%"     -> PERC x
        | _ -> failwithf "Unknown concentraiton unit '%s'" unit
    
    static member getUM (c:Concentration) = 
        match c with         
        | NM x -> x/1e3            
        | UM x -> x                    
        | MM x -> x*1e3            
        | M x ->  x*1e6
        | _ -> failwith "Not supported"            

    static member toString (c:Concentration) =                
        match c with 
        | NM x -> 
            if x > 100.0 then 
                Concentration.toString (UM (x/1000.0))
            else
                sprintf "%gnM" x

        | UM x -> 
            if x > 100.0 then 
                Concentration.toString (MM (x/1000.0))
            elif x < 0.1 then 
                Concentration.toString (NM (x*1000.0))
            else
                sprintf "%guM" x
        
        | MM x -> 
            if x > 100.0 then 
                Concentration.toString (M (x/1000.0))
            elif x < 0.1 then 
                Concentration.toString (UM (x*1000.0))
            else
                sprintf "%gmM" x

        | M x ->     
            if x < 0.1 then 
                Concentration.toString (MM (x*1000.0))
            else
                sprintf  "%gM" x
        
        | X x -> sprintf  "%gX" x

        | NGUL x -> sprintf  "%gng/uL" x

        | NGML x -> sprintf  "%gng/mL" x //TODO: convert between ng/mL and ng/uL for printing?

        | UmL x -> sprintf  "%gU/mL" x
        
        | ULML x -> sprintf  "%guL/mL" x 
        
        | UGML x -> sprintf  "%gug/mL" x 

        | PERC x -> sprintf  "%g%s" x "%" 

    static member zero = UM 0.0

    static member Parse (x:string) = 
       try
           let regexMatch = System.Text.RegularExpressions.Regex.Match(x, "([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\w*/?\w+|%)")
           let units = regexMatch.Groups.[3].Value
           let value = float regexMatch.Groups.[1].Value
           Concentration.Create value units
       with
       | _ ->  UM 1.0

    static member TryParse (x:string) = 
        try             
            Concentration.Parse x |> Some            
        with
        | _ -> None        


    static member op_Division (c:Concentration, x:float) = 
        if x = 0.0 then failwith "Concentration division by 0"
        else
            Concentration.Create ((Concentration.getValue c)/x) (Concentration.getUnit c)
    
    static member decode : Decoder<Concentration> =
        Decode.object(fun get ->
            let concVal = get.Required.Field "value" Decode.float
            let concUnit = get.Required.Field "unit" Decode.string
            Concentration.Create concVal concUnit
            )

    static member encode (conc:Concentration) = 
        Encode.object [
            "value", Encode.float (Concentration.getValue conc)
            "unit", Encode.string (Concentration.getUnit conc)
        ]

//type Volume = 
//    | UL of float //uL
//    | ML of float //mL    
    
type Time = 
    | Hours of float
    | Min of float
    | Sec of float
    static member getHours (t:Time) = 
        match t with 
        | Hours x -> x
        | Min x -> x/60.0
        | Sec x -> x/(60.0*60.0)
    static member getValue (t:Time) = 
        match t with 
        | Hours x
        | Min x 
        | Sec x -> x
    static member getUnit (conc:Time) = 
        match conc with 
        | Hours _ -> "h"
        | Min   _ -> "min"
        | Sec   _ -> "sec"        
    static member Create (x:float) (unit:string) = 
        match unit with 
        | "h"    -> Hours x
        | "min"  -> Min x
        | "sec"  -> Sec x        
        | _ -> failwithf "Unknown time unit %s" unit
    static member toString (t:Time) =
        sprintf "%f%s" (Time.getValue t) (Time.getUnit t)

    static member decode: Decoder<Time> =
        Decode.object(fun get ->
            let timeVal = get.Required.Field "value" (Decode.float)
            let timeUnit = get.Required.Field "units" (Decode.string)
            Time.Create timeVal timeUnit)
               
    static member encode (time: Time) = 
        let fields = 
            [
                "value", Encode.float (Time.getValue time)
                "units", Encode.string (Time.getUnit time)
            ]
        Encode.object fields


        

type FileType = 
    | AnthaBundleSource
    | AnthaBundleFinal
    | AnthaPlateLayout
    | AnthaInputPlate
    | CharacterizationData
    | SequencingData
    | CrnModel
    | MiscFile
    with 
    static member toString (t:FileType) = 
        match t with 
        | AnthaBundleSource    -> "AnthaBundleSource"
        | AnthaBundleFinal     -> "AnthaBundleFinal"  
        | AnthaPlateLayout     -> "AnthaPlateLayout"  
        | AnthaInputPlate      -> "AnthaInputPlate"
        | CharacterizationData -> "CharacterizationData"              
        | SequencingData       -> "SequencingData"
        | CrnModel             -> "CrnModel"
        | MiscFile             -> "MiscFile"

    static member fromString (s:string) = 
        match s with
        | "AnthaBundleSource"   -> AnthaBundleSource
        | "AnthaBundleFinal"    -> AnthaBundleFinal  
        | "AnthaPlateLayout"    -> AnthaPlateLayout  
        | "AnthaInputPlate"     -> AnthaInputPlate
        | "CharacterizationData"-> CharacterizationData              
        | "SequencingData"      -> SequencingData
        | "CrnModel"            -> CrnModel
        | "MiscFile"            -> MiscFile         
        | _ -> failwithf "Unknown file type %s" s


type FileRef = 
    { fileName : string
      fileId : FileId
      Type : FileType
    } with
    static member Create(fileName, fileType) = 
        { fileName = fileName
          fileId = FileId.Create()
          Type = fileType
        }
    static member empty = FileRef.Create("", MiscFile)
    static member addFileId = "++fileId"
    static member removeFileId = "--fileId"
    static member targetFileId = "fileId"
    static member removeType = "--type"
    static member addType = "++type"
    static member removeFileName = "--fileName"
    static member addFileName = "++fileName"

type Position = {
    row : int
    col : int
    } with
    static member toString (p:Position) = 
        sprintf "%c%i" (char (p.row+65)) (p.col + 1)
    
    static member fromString (s:string) = 
        //TODO: assuming single character. Need to generalize to 384-well plates
        { row = (s.[0] |> char |> int) - 65
          col = (int s.[1..]) - 1
        }

    static member decode: Decoder<Position> =
        Decode.object(fun get ->
            {   row = get.Required.Field "row" Decode.int
                col = get.Required.Field "col" Decode.int
            })
    
    static member encode (position: Position) = 
        let requiredFields = 
            [ "row", Encode.int (position.row)
              "col", Encode.int (position.col)
            ]
        Encode.object requiredFields

type ExperimentScale = 
    | ShakeFlask of float
    | DeepPlate24 of float
    | LowPlate24 of float
    | Plate96 of float
    | Bioreactor of float
    static member getVolume (ts:ExperimentScale) = 
        match ts with 
        | ShakeFlask(v)   -> v
        | DeepPlate24(v)  -> v
        | LowPlate24(v)   -> v
        | Plate96(v)      -> v
        | Bioreactor(v)   -> v
    static member toString (ts:ExperimentScale) = 
        match ts with 
        | ShakeFlask(v)    -> "Shake Flask " + v.ToString() + "mL"
        | DeepPlate24(v)   -> "24-Deep Well " + v.ToString() + "mL"
        | LowPlate24(v)   -> "24-Low Well " + v.ToString() + "mL"
        | Plate96(v)       -> "96 Well " + v.ToString() + "mL"
        | Bioreactor(v)    -> "Bioreactor " + v.ToString() + "mL"

    static member fromString (s:string) = 
        let getVol (str) =
            let regexMatch = System.Text.RegularExpressions.Regex.Match(str,"(\d*\.?\d+)mL")
            Double.Parse(regexMatch.Groups.[1].Value)
        match s with 
        | str when str.StartsWith("Shake Flask") -> ShakeFlask(getVol(s.Substring(12)))
        | str when str.StartsWith("24-Deep Well") -> DeepPlate24(getVol(s.Substring(13)))
        | str when str.StartsWith("24-Low Well") -> LowPlate24(getVol(s.Substring(12)))
        | str when str.StartsWith("96 Well") -> Plate96(getVol(s.Substring(8)))
        | str when str.StartsWith("Bioreactor") -> Bioreactor(getVol(s.Substring(11)))
        | _ -> failwithf "Unknown Scale: %s" s

type ExperimentType = 
    | BuildExperiment //of BuildExperimentData
    | TestExperiment //of TestExperimentData
    static member toString (e:ExperimentType)  = 
        match e with 
        | BuildExperiment -> "TypeIIs assembly"
        | TestExperiment -> "Characterization"
    static member fromString (s:string) = 
        let s' = s.Split ':'
        match s'.[0] with 
        | "TypeIIs assembly" -> BuildExperiment
        | "Characterization" -> TestExperiment
        | _ -> failwithf "Unknown experiment type %s" s

type ExperimentOperationType = 
    //shared events
    | AnthaExecuted
    | AnthaBundleUploaded
    | AnthaLayoutUploaded    
    | ExperimentFinished
    | ExperimentStarted    

    //events for test experiments
    | BacterialStocksInnoculated
    | OvernightStocksDiluted        
    | PlateReaderStarted
    | ResultsProcessed

    //events for build experiments
    | InputPlatePrepared
    | PlateIncubated
    | ColoniesPicked

    // misc events
    | Induction
    | Transfection
    with 
    static member toString (e:ExperimentOperationType) = 
        match e with 
        | AnthaExecuted              -> "AnthaExecuted"
        | AnthaBundleUploaded        -> "AnthaBundleUploaded"
        | AnthaLayoutUploaded        -> "AnthaLayoutUploaded"
        | BacterialStocksInnoculated -> "BacterialStocksInnoculated"
        | OvernightStocksDiluted     -> "OvernightStocksDiluted"   
        | PlateReaderStarted         -> "PlateReaderStarted"   
        | InputPlatePrepared         -> "InputPlatePrepared"
        | PlateIncubated             -> "PlateIncubated"
        | ColoniesPicked             -> "ColoniesPicked"
        | ExperimentStarted          -> "ExperimentStarted"
        | ExperimentFinished         -> "ExperimentFinished"
        | ResultsProcessed           -> "ResultsProcessed"
        | Induction                  -> "Induction"
        | Transfection               -> "Transfection"
    static member fromString (s:string) = 
        match s with                 
        | "AnthaExecuted" -> AnthaExecuted            
        | "AnthaBundleUploaded" -> AnthaBundleUploaded      
        | "AnthaLayoutUploaded" -> AnthaLayoutUploaded
        | "BacterialStocksInnoculated" -> BacterialStocksInnoculated
        | "OvernightStocksDiluted"    -> OvernightStocksDiluted   
        | "PlateReaderStarted"    -> PlateReaderStarted       
        | "InputPlatePrepared" -> InputPlatePrepared       
        | "PlateIncubated" -> PlateIncubated           
        | "ColoniesPicked" -> ColoniesPicked        
        | "ExperimentStarted" -> ExperimentStarted
        | "ExperimentFinished" -> ExperimentFinished        
        | "ResultsProcessed" -> ResultsProcessed
        | "Induction" -> Induction
        | "Transfection" -> Transfection
        | _ -> failwithf "Unknown experiment event type %s" s
    static member toDescription (e:ExperimentOperationType) = 
        match e with 
        | AnthaExecuted              -> "Download and execute Antha"
        | AnthaBundleUploaded        -> "Upload Antha output bundle"
        | AnthaLayoutUploaded        -> "Upload Antha layout file"
        | BacterialStocksInnoculated -> "Innoculate bacterial stocks"
        | OvernightStocksDiluted     -> "Dilute overnight stocks"   
        | PlateReaderStarted         -> "Start plate reader run"   
        | InputPlatePrepared         -> "Prepare input plate with reagents"
        | PlateIncubated             -> "Incubate output plate"
        | ColoniesPicked             -> "Pick colonies"
        | ExperimentStarted          -> "Start experiment"
        | ExperimentFinished         -> "Finalize experiment"
        | ResultsProcessed           -> "Process experimental data"
        | Transfection               -> "Transfect the genome"
        | Induction                  -> "Induction step"

    static member TestProtocolEventsOrder = 
        [ ExperimentStarted
          BacterialStocksInnoculated
          OvernightStocksDiluted
          AnthaExecuted
          AnthaBundleUploaded
          AnthaLayoutUploaded
          PlateReaderStarted
          ExperimentFinished
          ResultsProcessed
        ]
    static member BuildProtocolEventsOrder = 
        [ ExperimentStarted
          InputPlatePrepared          
          AnthaExecuted
          AnthaBundleUploaded
          AnthaLayoutUploaded
          PlateIncubated
          ColoniesPicked
          ExperimentFinished
        ]

type ExperimentOperation = 
    { id : ExperimentOperationId
      timestamp : System.DateTime
      Type : ExperimentOperationType
    } with 
    static member Create (t:ExperimentOperationType) = 
        { id = ExperimentOperationId.Create()
          Type = t
          timestamp = System.DateTime.UtcNow
        }

    static member decode:Decoder<ExperimentOperation> = 
        Decode.object(fun get -> 
            {
                id = get.Required.Field "id" Decode.string |> System.Guid |> ExperimentOperationId
                timestamp = 
                    let ts_string = get.Required.Field "timestamp" Decode.string
                    System.DateTime.ParseExact(ts_string, "o", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None)
                Type = get.Required.Field "type" Decode.string |> ExperimentOperationType.fromString
            })
    
    static member encode (eo:ExperimentOperation) = 
        Encode.object[
            "id", eo.id.ToString() |> Encode.string
            "timestamp", eo.timestamp.ToUniversalTime().ToString("o") |> Encode.string
            "type", eo.Type |> ExperimentOperationType.toString |> Encode.string
        ]

    static member empty = ExperimentOperation.Create ExperimentFinished
    static member addExperimentOperationId = "++experimentOperationId"
    static member removeExperimentOperationId = "--experimentOperationId"
    static member targetExperimentOperationId = "experimentOperationId"

    static member addTimestamp = "++timestamp"
    static member removeTimestamp = "--timestamp"
    
    static member addType = "++type"
    static member removeType = "--type"

type PositionVariety =
    | NotLoaded
    | Hardcoded
    | FromBarcodeReader

type WellPositions = {
    positionVariety : PositionVariety
    plasmidStocks1Positions : Map<Position, Barcode>
    bacterialStocks1Positions : Map<Position, Barcode>
    demoStoragePositions : Map<Position, Barcode>
    targetPlatePositions : Map<Position, Barcode>
}

type PartProperties = {
    sequence: string
    name: string
    deprecated:bool
    } 
    with
    static member empty = {sequence = "";name = ""; deprecated = false}

type CDSPart             = {id:CDSId; properties:PartProperties}
type PromoterPart        = {id:PromoterId; properties:PartProperties}
type TerminatorPart      = {id:TerminatorId;properties:PartProperties}
type UserDefinedPart     = {id:UserDefinedId;properties:PartProperties}
type ScarPart            = {id:ScarId;properties:PartProperties}
type RBSPart             = {id:RBSId;properties:PartProperties}
type BackbonePart        = {id:BackboneId;properties:PartProperties}
type OriPart             = {id:OriId;properties:PartProperties}
type LinkerPart          = {id:LinkerId;properties:PartProperties}
type RestrictionSitePart = {id:RestrictionSiteId;properties:PartProperties}

type Part = 
    | CDS of CDSPart
    | Promoter  of PromoterPart
    | Terminator  of TerminatorPart
    | UserDefined  of UserDefinedPart
    | Scar of ScarPart
    | RBS of RBSPart
    | Backbone of BackbonePart
    | Ori of OriPart
    | Linker of LinkerPart
    | RestrictionSite of RestrictionSitePart
    
    with
    static member empty = UserDefined{id = UserDefinedId.Create(); properties = PartProperties.empty}

    static member SetProperties (p:Part) (props':PartProperties)= 
        match p with 
        | CDS             prop -> CDS             {prop with properties = props'}
        | Promoter        prop -> Promoter        {prop with properties = props'}
        | Terminator      prop -> Terminator      {prop with properties = props'}
        | UserDefined     prop -> UserDefined     {prop with properties = props'}
        | Scar            prop -> Scar            {prop with properties = props'}
        | RBS             prop -> RBS             {prop with properties = props'}
        | Backbone        prop -> Backbone        {prop with properties = props'}
        | Ori             prop -> Ori             {prop with properties = props'}
        | Linker          prop -> Linker          {prop with properties = props'}
        | RestrictionSite prop -> RestrictionSite {prop with properties = props'}

    static member SetSequence (p:Part) (sequence:string) = 
        let props = Part.GetProperties p
        Part.SetProperties p {props with sequence = sequence}

    static member SetDeprecated (p:Part) (deprecated:bool) =
        let props = Part.GetProperties p
        Part.SetProperties p {props with deprecated = deprecated}            

    static member GetProperties (p:Part) = 
        match p with 
        | CDS             prop -> prop.properties
        | Promoter        prop -> prop.properties
        | Terminator      prop -> prop.properties
        | UserDefined     prop -> prop.properties
        | Scar            prop -> prop.properties
        | RBS             prop -> prop.properties
        | Backbone        prop -> prop.properties
        | Ori             prop -> prop.properties
        | Linker          prop -> prop.properties
        | RestrictionSite prop -> prop.properties

    static member ToGenBank (p : Part) =
        match p with
        | CDS _ -> "CDS"
        | Promoter _ -> "promoter"
        | Terminator _ -> "terminator"
        | Ori _ -> "rep_origin"
        | RBS _ -> "RBS"
        | Backbone _ -> "source"
        | Linker _ -> "transit_peptide"
        | Scar _ -> "misc_signal"
        | RestrictionSite _ -> "misc_binding"
        | UserDefined _ -> "user-defined"
    
    static member Color(p : Part) =
        match p with
        | CDS _ -> "#ff797d"
        | Promoter _ -> "#346ee0"
        | Terminator _ -> "#9d1b1c"
        | Ori _ -> "#999999"
        | RBS _ -> "cyan"
        | Backbone _ -> "#c0c0c0"
        | Linker _ -> "#cccccc"
        | Scar _ -> "#008040"
        | RestrictionSite _ -> "#ffff00"
        | UserDefined _ -> "#aaaaaa"
    
    static member ToVisbol(p : Part) =
        match p with
        | CDS _ -> "cds"
        | Promoter _ -> "promoter"
        | RBS _ -> "res"
        | Terminator _ -> "terminator"
        | Scar _ -> "assembly-scar"
        | RestrictionSite _ -> "restriction-site"
        | Backbone _ -> "engineered-region"
        | Ori _ -> "origin-of-replication"
        | Linker _ -> "protein-domain"
        | UserDefined _ -> "user-defined"
    
    static member GetType (p:Part) = 
        match p with 
        | CDS             _ -> "CDS"
        | Promoter        _ -> "Promoter"
        | Terminator      _ -> "Terminator"
        | UserDefined     _ -> "UserDefined"
        | Scar            _ -> "Scar"
        | RBS             _ -> "RBS"
        | Backbone        _ -> "Backbone"
        | Ori             _ -> "Ori"
        | Linker          _ -> "Linker"
        | RestrictionSite _ -> "RestrictionSite"

    static member FromStringType (guid:System.Guid) (props:PartProperties) (partType:string) = 
        match partType with
        | "CDS" -> CDS({id = guid |> CDSId; properties = props})
        | "Promoter" -> Promoter({id = guid |> PromoterId; properties = props})
        | "Terminator" -> Terminator({id = guid |> TerminatorId; properties = props})
        | "UserDefined" -> UserDefined({id = guid |> UserDefinedId; properties = props})
        | "Scar" -> Scar({id = guid |> ScarId; properties = props})
        | "RBS" -> RBS({id = guid |> RBSId; properties = props})
        | "Backbone" -> Backbone({id = guid |> BackboneId; properties = props})
        | "Ori" -> Ori({id = guid |> OriId; properties = props})
        | "Linker" -> Linker({id = guid |> LinkerId; properties = props})
        | "RestrictionSite" -> RestrictionSite({id = guid |> RestrictionSiteId; properties = props})
        | _ -> failwithf "[ERROR] %s not a recognized Part type." (partType)
    
    static member FromGenBankStringType (guid:System.Guid)  (props:PartProperties) (partType:string) = 
        match partType with
        | "CDS" -> CDS({id = guid |> CDSId; properties = props})
        | "promoter" -> Promoter({id = guid |> PromoterId; properties = props})
        | "terminator" -> Terminator({id = guid |> TerminatorId; properties = props})
        | "user-defined" -> UserDefined({id = guid |> UserDefinedId; properties = props})
        | "misc_signal" -> Scar({id = guid |> ScarId; properties = props})
        | "RBS" -> RBS({id = guid |> RBSId; properties = props})
        | "source" -> Backbone({id = guid |> BackboneId; properties = props})
        | "rep_origin" -> Ori({id = guid |> OriId; properties = props})
        | "transit_peptide" -> Linker({id = guid |> LinkerId; properties = props})
        | "misc_binding" -> RestrictionSite({id = guid |> RestrictionSiteId; properties = props})
        | _ -> failwithf "[ERROR] %s not a recognized GenBank Part type." (partType)
    
    static member Available = [
        "CDS"
        "Promoter"
        "Terminator"
        "UserDefined"
        "Scar"
        "RBS"
        "Backbone"
        "Ori"
        "Linker"
        "RestrictionSite"    
    ]

    static member decode: Decoder<Part> =
        Decode.object(fun get ->
            let partProperties = {
                name = get.Required.Field "name" Decode.string
                sequence = get.Required.Field "sequence" Decode.string
                deprecated = get.Optional.Field "deprecated" Decode.bool
                             |> Option.defaultValue false
            }
            let guid = get.Required.Field "id" Decode.guid
            let partType = get.Required.Field "type" Decode.string
            Part.FromStringType guid partProperties partType)

    static member encode (part:Part) = 
        let requiredFields = 
            [ "id", Encode.string (part.id.ToString())
              "name", Encode.string (part.name)
              "type", Encode.string (part.getType)
              "sequence", Encode.string (part.sequence)
              "deprecated", Encode.bool (part.deprecated)
            ]
        Encode.object requiredFields

    member this.getProperties = Part.GetProperties this
    member this.id = 
        match this with 
        | CDS             x -> x.id |> CDSPartId
        | Promoter        x -> x.id |> PromoterPartId
        | Terminator      x -> x.id |> TerminatorPartId
        | UserDefined     x -> x.id |> UserDefinedPartId
        | Scar            x -> x.id |> ScarPartId
        | RBS             x -> x.id |> RBSPartId
        | Backbone        x -> x.id |> BackbonePartId
        | Ori             x -> x.id |> OriPartId
        | Linker          x -> x.id |> LinkerPartId
        | RestrictionSite x -> x.id |> RestrictionSitePartId
        
    member this.name = this.getProperties.name
    member this.sequence = this.getProperties.sequence
    member this.deprecated = this.getProperties.deprecated
    member this.getType = Part.GetType this
    member this.guid = this.id.ToString() |> System.Guid
    

    static member addName = "++name"
    static member removeName = "--name"
    static member addSequence = "++sequence"
    static member removeSequence = "--sequence"
    static member addType = "++type"
    static member removeType = "--type"
    static member addTag = "++tag"
    static member removeTag = "--tag"
    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"

type DNAType =
    | SourceLinearDNA
    | SourcePlasmidDNA
    | AssembledPlasmidDNA
    | GenericPlasmidDNA
    with 
    static member ToString (dt:DNAType) = 
        match dt with 
        | SourceLinearDNA -> "Linear DNA (source)"
        | SourcePlasmidDNA -> "Plasmid DNA (source)"
        | AssembledPlasmidDNA -> "Plasmid DNA (assembled)"
        | GenericPlasmidDNA -> "Plasmid DNA (generic)"
    static member fromString (str:string) = 
        match str with 
        | "Linear DNA (source)" -> SourceLinearDNA
        | "Plasmid DNA (source)" -> SourcePlasmidDNA
        | "Plasmid DNA (assembled)" -> AssembledPlasmidDNA
        | "Plasmid DNA (generic)" -> GenericPlasmidDNA
        | _ -> failwithf "Unknown type of DNA: %s" str

    static member Available = [SourceLinearDNA; SourcePlasmidDNA; AssembledPlasmidDNA]

type RNAType =
    | MessengerRNA
    | TransferRNA
    | GuideRNA
    | SmallRNA
    with
    static member ToString (rt:RNAType) = 
        match rt with 
        | MessengerRNA -> "Messenger RNA"
        | TransferRNA-> "Transfer RNA"
        | GuideRNA -> "Guide RNA"
        | SmallRNA -> "Small RNA"
    static member fromString (str:string) = 
        match str with 
        | "Messenger RNA" -> MessengerRNA
        | "Transfer RNA" -> TransferRNA
        | "Guide RNA" -> GuideRNA
        | "Small RNA" -> SmallRNA
        | _ -> failwithf "Unknown type of RNA: %s" str

type ChemicalType = 
    | Media
    | Antibiotic
    | SmallMolecule
    | Other
    with 
    static member ToString (ct:ChemicalType) = 
        match ct with 
        | Media -> "Media"
        | Antibiotic -> "Antibiotic"
        | SmallMolecule -> "Small Molecule"
        | Other -> "Other"
    static member fromString (str:string) = 
        match str with 
        | "Media" -> Media
        | "Antibiotic" -> Antibiotic
        | "Small Molecule" -> SmallMolecule
        | "Other" -> Other
        | _ -> failwithf "Unknown chemical type: %s" str

type ReagentProperties = {
    name : string 
    notes: string
    barcode : Barcode option
    deprecated:bool    
    } with 
    static member empty = 
        { name = ""
          barcode = None
          notes = ""
          deprecated = false
    }

type DNAReagent = {
    id:DNAId
    properties : ReagentProperties
    Type : DNAType
    sequence: string
    concentration: Concentration option
} with
    //Can the devices be used in a BUILD protocol
    static member IsBuildable (reagent:DNAReagent) = 
     reagent.Type = AssembledPlasmidDNA && //we can only build Assembled devices (TODO: only plasmid?)
     reagent.properties.barcode = None //We should not build devices that we already have

    static member empty = 
         { 
           id = DNAId.Create()
           properties = ReagentProperties.empty
           Type = DNAType.SourceLinearDNA
           sequence = ""
           concentration = None
        }

type RNAReagent = {
    id: RNAId
    properties : ReagentProperties
    sequence:string
    Type : RNAType
} 

type ChemicalReagent = {
    id:ChemicalId
    properties : ReagentProperties
    Type:ChemicalType
}

type ProteinReagent = {
    id:ProteinId
    properties : ReagentProperties
    isReporter:bool
}

type GenericEntityReagent = {
    id: GenericEntityId
    properties:ReagentProperties 
}

type PartInstance = 
    { id : PartId   //guid of the part
      position: int //position in sequence
      orientation: bool //true: 5'->3', false: reversed
    }
    static member ContainsBackbone (partsDB:Map<PartId, Part>) (parts: seq<PartInstance>) = 
        parts
        |> Seq.exists (fun part ->
                match partsDB.[part.id] with
                | Backbone _ -> true
                | _ -> false)        

                       
//TODO: "DerivedFrom" relationship for virtual and physical reagents (e.g. DNA)
//TODO: "BuiltFrom" relationship for assembled DNA constructs built from source DNA constructs
//TODO: Split Reagents into DNA and non-DNA ones. Store both in a common noSQL table?
//TODO: Types of tags (e.g. Project, source, etc)


type Reagent = 
    | Chemical of ChemicalReagent
    | DNA of DNAReagent
    | RNA of RNAReagent
    | Protein of ProteinReagent
    | GenericEntity of GenericEntityReagent
    with
    static member empty = GenericEntity({id = GenericEntityId.Create();properties=ReagentProperties.empty})
    
    static member addName = "++name"
    static member removeName = "--name"
    static member addBarcode = "++barcode"
    static member removeBarcode = "--barcode"
    static member addType = "++type"
    static member removeType = "--type"
    static member addContext = "++context"
    static member removeContext = "--context"
    static member addSequence = "++sequence"
    static member removeSequence = "--sequence"
    static member addConcentration = "++concentration"
    static member removeConcentration = "--concentration"
    static member addNotes = "++notes"
    static member removeNotes = "--notes"
    static member addTag = "++tag"
    static member removeTag = "--tag"

    static member addDNAType = "++dnatype"
    static member removeDNAType = "--dnatype"
    static member addRNAType = "++rnatype"
    static member removeRNAType = "--rnatype"
    static member addChemicalType = "++chemicaltype"
    static member removeChemicalType = "--chemicaltype"
    
    static member addIsReporter = "++isReporter"
    static member removeIsReporter = "--isReporter"

    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"
    
    static member GetType (r:Reagent) = 
        match r with 
        | Chemical _ -> "Chemical"
        | DNA _ -> "DNA"
        | RNA _ -> "RNA"
        | Protein _ -> "Protein"
        | GenericEntity _ -> "Generic Entity"
    
    member this.getType = Reagent.GetType this

    
    static member WithBarcode (barcode:Barcode option) (reagent:Reagent) = 
        let props = Reagent.GetProperties reagent
        Reagent.UpdateProperties reagent {props with barcode = barcode}
    member this.SetBarcode (barcode:Barcode option) = 
        this |> Reagent.WithBarcode barcode

    static member GetProperties (r:Reagent) = 
        match r with 
        | Chemical chem -> chem.properties
        | DNA dna -> dna.properties
        | RNA rna -> rna.properties
        | Protein prot -> prot.properties
        | GenericEntity ge -> ge.properties
    static member UpdateProperties (r:Reagent) (props:ReagentProperties) = 
        match r with 
        | DNA (dna) -> DNA({dna with properties = props})
        | RNA (rna) -> RNA({rna with properties = props})
        | Protein (prot) -> Protein({prot with properties = props})
        | Chemical (chem) -> Chemical({chem with properties = props})
        | GenericEntity ge -> GenericEntity ({ge with properties = props})
    
    static member UpdatePropertiesByType (r:Reagent) (props:ReagentProperties) = 
        match r with 
        | DNA (dna) -> DNA({dna with properties = props})
        | RNA (rna) -> RNA({rna with properties = props})
        | Protein (prot) -> Protein({prot with properties = props})
        | Chemical (chem) -> Chemical({chem with properties = props})
        | GenericEntity ge -> GenericEntity ({ge with properties = props})
    
    static member Available = ["DNA"; "RNA"; "Chemical"; "Generic Entity"]
    member this.getProperties = Reagent.GetProperties this
    member this.deprecated = this.getProperties.deprecated
    member this.name = this.getProperties.name
    member this.barcode = this.getProperties.barcode
    member this.notes = this.getProperties.notes
    member this.id = 
        match this with 
        | DNA x -> x.id |> DNAReagentId
        | RNA x -> x.id |> RNAReagentId
        | Chemical x -> x.id |> ChemicalReagentId
        | Protein x -> x.id |> ProteinReagentId
        | GenericEntity x -> x.id |> GenericEntityReagentId
    
    static member decode: Decoder<Reagent> =
        Decode.object(fun get ->
            let reagentProperties : ReagentProperties = {
                name = get.Required.Field "name" Decode.string
                notes = get.Optional.Field "notes" Decode.string
                        |> Option.defaultValue ""
                barcode =
                    let bvalue = get.Optional.Field "barcode" Decode.string
                                 |> Option.defaultValue ""
                    match bvalue with
                    | "" -> None
                    | _ -> bvalue |> Barcode |> Some
                deprecated = get.Optional.Field "deprecated" Decode.bool
                             |> Option.defaultValue false
            }
            let guid = get.Required.Field "id" Decode.guid
            let reagentType = get.Optional.Field "type" Decode.string
                                |> Option.defaultValue "Generic"
            match reagentType with
            | "SourceLinearDNA" | "SourcePlasmidDNA" | "AssembledPlasmidDNA" ->
                let dnaType = 
                    match reagentType with
                    | "SourceLinearDNA" -> SourceLinearDNA
                    | "SourcePlasmidDNA" -> SourcePlasmidDNA
                    | "AssembledPlasmidDNA" -> AssembledPlasmidDNA
                    | _ -> failwithf "%s not a recognized type of DNA" (reagentType)
                let conc = get.Optional.Field "concentration" Concentration.decode
                let sequence = get.Required.Field "sequence" Decode.string
                {
                    id = guid |> DNAId
                    properties = reagentProperties
                    sequence = sequence
                    concentration = conc
                    Type = dnaType
                }
                |> DNA
            | "MessengerRNA" | "TransferRNA" | "GuideRNA" | "SmallRNA" ->
                let rnaType = 
                    match reagentType with
                    | "MessengerRNA" -> MessengerRNA
                    | "TransferRNA" -> TransferRNA
                    | "GuideRNA" -> GuideRNA
                    | "SmallRNA" -> SmallRNA
                    | _ -> failwithf "%s not a recognized type of RNA" (reagentType)
                let sequence = get.Required.Field "sequence" Decode.string
                {
                    id = guid |> RNAId
                    properties = reagentProperties
                    sequence = sequence
                    Type = rnaType
                }
                |> RNA
            | "Media" | "Antibiotic" | "SmallMolecule"  | "Other" ->
                let chemicalType = 
                    match reagentType with
                    | "Media" -> Media
                    | "Antibiotic" -> Antibiotic
                    | "SmallMolecule" -> SmallMolecule
                    | "Other" -> Other
                    | _ -> failwithf "%s not a recognized type of Chemical" (reagentType)                   
                {
                    id = guid |> ChemicalId
                    properties = reagentProperties
                    Type = chemicalType
                }
                |> Chemical
            | "Generic" ->
                match get.Optional.Field "isReporter" Decode.bool with
                | Some(isReporter) ->
                    {id = guid |> ProteinId; properties = reagentProperties; isReporter = isReporter}
                    |> Protein
                | None -> {id = guid |> GenericEntityId; properties = reagentProperties} |> GenericEntity
            | _ -> failwithf "Unknown type of reagent %s encountered." (reagentType)
            )
    
    static member encode (reagent:Reagent) = 
        let reagentProperties = [
            "id", Encode.string (reagent.id.ToString())
            "name", Encode.string (reagent.name)
            "notes", Encode.string (reagent.notes)
            "deprecated", Encode.bool (reagent.deprecated)
            match reagent.barcode with | Some(b) -> "barcode", Encode.string (b.ToString()) | None -> ()
        ]
        match reagent with
        | DNA dna -> 
            let dnatype = 
                match dna.Type with
                | SourceLinearDNA -> "SourceLinearDNA"
                | SourcePlasmidDNA -> "SourcePlasmidDNA"
                | AssembledPlasmidDNA -> "AssembledPlasmidDNA"
                | GenericPlasmidDNA -> "GenericPlasmidDNA"
            let dnaProperties = [
                "type", Encode.string dnatype
                "sequence", Encode.string dna.sequence
            ]
            Encode.object (reagentProperties@dnaProperties)
        | RNA rna -> 
            let rnatype = 
                match rna.Type with
                | MessengerRNA -> "MessengerRNA"
                | TransferRNA -> "TransferRNA"
                | SmallRNA -> "SmallRNA"
                | GuideRNA -> "GuideRNA"
            let rnaProperties = [
                "type", Encode.string rnatype
                "sequence", Encode.string rna.sequence
            ]
            Encode.object (reagentProperties@rnaProperties)
        | Chemical chem -> 
            let chemicalType = 
                match chem.Type with 
                | Media -> "Media"
                | Antibiotic -> "Antibiotic"
                | SmallMolecule -> "SmallMolecule"
                | Other -> "Other"
            Encode.object (("type",Encode.string chemicalType)::reagentProperties)
        | Protein prot -> 
            Encode.object (("isReporter", Encode.bool prot.isReporter)::reagentProperties)
        | GenericEntity ge -> Encode.object reagentProperties

    
    static member AnthaName (r:Reagent) = 
        sprintf "%s %s" r.getProperties.name (r.id.ToString())
        
        
    member this.anthaName = Reagent.AnthaName this

    static member MkDnaDeviceName (partsDB:Map<PartId,Part>) (parts: PartInstance list) =
        parts
        |> Seq.map (fun p -> p.id)
        |> Seq.filter (fun pid ->
                match partsDB.[pid] with
                | CDS _ | Promoter _ | RBS _ | Linker _ | Backbone _ -> true
                | _ -> false)
        |> Seq.map(fun pid -> partsDB.[pid].getProperties.name)
        |> String.concat "_"

    //returns a parts list (part name, index)
    static member AnnotateSequence (parts:Map<PartId,Part>) (sequence:string) =         
        let ALLOW_OVERLAP = false
        let rec FindIDs start (s : string) (s' : string) : int list =
            if s.Contains s' then
                let i = s.IndexOf s'
                let start' = if ALLOW_OVERLAP then i+1 else i + s'.Length + 1
                let cont = 
                    if start' > s.Length then List.empty                    
                    else
                        FindIDs (start + start') s.[start'..] s'             
                (i+start) :: cont
            else List.empty
                
        let mutable S = sequence        
        parts
        |> Map.toList        
        |> List.map snd
        |> List.sortByDescending (fun part -> part.getProperties.sequence.Length)
        |> List.map (fun part ->                        
            let blank = String.replicate part.getProperties.sequence.Length "*"                                                       
            let RunScan isForward s = 
                let instances =                     
                    FindIDs 0 S s                    
                    |> List.map (fun i ->                         
                        { PartInstance.id = part.id
                          PartInstance.position = i
                          PartInstance.orientation = isForward
                        })                                                         
                if not ALLOW_OVERLAP then
                    S <- S.Replace(s, blank)                    
                instances
                
            let revComplement (s:string) = 
                s.ToUpper() 
                |> Seq.map(fun c -> 
                    match c with 
                    | 'A' -> 'T'
                    | 'G' -> 'C'
                    | 'C' -> 'G'
                    | 'T' -> 'A'
                    | _ -> failwithf "Unexpected DNA base %c" c
                    )
                |> Seq.rev
                |> Array.ofSeq
                |> System.String

            List.append 
                (RunScan true part.getProperties.sequence) //forward scan
                (RunScan false (revComplement part.getProperties.sequence)) //reverse scan                            
            )
        |> List.concat
        |> List.sortBy(fun p -> p.position)
    
    static member SequenceFromGenBank (content : string) =
        let origin_id = content.IndexOf("ORIGIN")

        let sequence =
            content.[origin_id..]
                .Split([| "\r"; "\n" |],
                       System.StringSplitOptions.RemoveEmptyEntries)            
            |> Array.filter (fun l ->                 
                l.Trim().Length > 0 &&      //ignore empty lines
                l.Trim().[0..1] <> "//" &&  //ignore comments
                l.Length > 10  &&           //ignore lines that are too short to contain sequence
                l.[10..].Trim().Length>0    //ignore empty sequence lines
                )            
            |> Array.map (fun l -> 
                
                l.[10..].Replace(" ", ""))
            |> String.concat ""
            |> fun S -> S.ToUpper().Trim()
        sequence

    static member SequenceToGenBank (parts: Map<PartId,Part>) (annotation:PartInstance list) (name:string) (sequence : string) =                                
        //export settings
        let num_bases = 10 //bases per group
        let num_groups = 6 //groups per row
        //GenBank generation        

        let SplitBy (n : int) (L : 'a []) =
            let mx = System.Math.Floor((float L.Length) / (float n)) |> int
            [| 0..mx |]
            |> Array.map (fun i ->
                   if i * n + n - 1 < L.Length then L.[i * n..i * n + n - 1]
                   else L.[i * n..])

        let origin =
            sequence.ToCharArray()
            |> SplitBy num_bases
            |> Array.map (fun ca -> System.String(ca))
            |> SplitBy num_groups
            |> Array.mapi (fun i s ->
                   let S = s |> String.concat " "
                   //Extra varible to avoid breaking Fantomas https://github.com/fsprojects/fantomas/issues/365
                   let asString =
                       (sprintf "%i" (i * num_bases * num_groups + 1))
                   let num = asString.PadLeft(9)
                   sprintf "%s %s" num S)
            |> String.concat "\n"
            |> sprintf "ORIGIN\n%s"

        let features =
            let MkFeature orientation (k : string) (s, e) =
                if orientation then 
                    sprintf "     %s%i..%i" (k.PadRight(16)) s e
                else
                    sprintf "     %scomplement(%i..%i)" (k.PadRight(16)) s e

            let MkProp(k : string) = sprintf "%s/%s" (String.replicate 21 " ") k

            annotation
            |> List.mapi
                   (fun i part ->
                       [ MkFeature part.orientation (parts.[part.id] |> Part.ToGenBank)
                             (part.position + 1, part.position + parts.[part.id].getProperties.sequence.Length)
                         MkProp(sprintf "label=\"%s\"" parts.[part.id].getProperties.name)

                         MkProp
                             (sprintf "ApEinfo_fwdcolor=\"%s\""
                                  (parts.[part.id] |> Part.Color)) 
                        ])

            |> List.concat
            |> String.concat "\n"
            |> sprintf "FEATURES             Location/Qualifiers\n%s"

        let locus =
            ("LOCUS").PadRight(12) + name.PadRight(13)
            + (sprintf "%i bp" sequence.Length).PadRight(11) + ("DNA").PadRight(16)
            + ("UNA").PadRight(10) + (System.DateTime.Now.ToShortDateString())
        sprintf "%s\n%s\n%s" locus features origin

type InteractionNodeType = 
    | Template
    | Product
    | Activator
    | Activated
    | Inhibitor
    | Inhibited
    | Reactant
    | Regulator
    | Regulated
    | Enzyme
    with
    static member toString(iet:InteractionNodeType) = 
        match iet with 
        | Template     -> "Template"      
        | Product      -> "Product"  
        | Activator    -> "Activator"
        | Activated    -> "Activated"
        | Inhibitor    -> "Inhibitor"
        | Inhibited    -> "Inhibited"
        | Reactant     -> "Reactant" 
        | Enzyme       -> "Enzyme"
        | Regulator    -> "Regulator"
        | Regulated    -> "Regulated"
    static member fromString(str:string) =  
        match str with 
        | "Template"   -> Template      
        | "Product"    -> Product  
        | "Activator"  -> Activator
        | "Activated"  -> Activated
        | "Inhibitor"  -> Inhibitor
        | "Inhibited"  -> Inhibited
        | "Reactant"   -> Reactant 
        | "Enzyme"     -> Enzyme   
        | "Regulated"  -> Regulated
        | "Regulator"  -> Regulator
        | _ -> failwithf "%s is an unknown type of interaction node. InteractionEntityNodes can only either be Source or Target" str

type InteractionProperties = 
    {
        id:InteractionId;
        notes: string;
        deprecated:bool
    } with 
    static member empty = {
        id = InteractionId.Create()
        notes = ""
        deprecated = false
    }
    
type CodesForInteraction = {
    properties:InteractionProperties
    cds:CDSId
    protein:ProteinId
    } with
    static member addCDS = "++cds"
    static member removeCDS = "--cds"
    static member addProtein = "++protein"
    static member removeProtein = "--protein"

type GeneticActivationInteraction = {
    properties:InteractionProperties
    activator:ReagentId list
    activated:PromoterId
    } with 
    static member addActivator = "++activator"
    static member removeActivator = "--activator"
    static member addActivated = "++activated"
    static member removeActivated = "--activated"
    
type GeneticInhibitionInteraction = {
    properties:InteractionProperties
    inhibitor:ReagentId list
    inhibited:PromoterId
    } with 
    static member addInhibitor = "++inhibitor"
    static member removeInhibitor = "--inhibitor"
    static member addInhibited = "++inhibited"
    static member removeInhibited = "--inhibited"

type GenericInteraction = {
    properties:InteractionProperties
    regulator:ReagentId list
    regulated:ReagentId
    } with 
    static member addRegulator = "++regulator"
    static member removeRegulator = "--regulator"
    static member addRegulated = "++regulated"
    static member removeRegulated = "--regulated"
  
type ReactionInteraction = {
    properties:InteractionProperties
    reactants:ReagentId list list
    enzyme:ReagentId list 
    products:ReagentId list list
    } with 
    static member addReactants = "++reactants"
    static member removeReactants = "--reactants"
    static member addEnzyme = "++enzyme"
    static member removeEnzyme = "--enzyme"
    static member addProducts = "++products"
    static member removeProducts = "--products"

type Interaction = 
    | CodesFor of CodesForInteraction
    | GeneticActivation of GeneticActivationInteraction
    | GeneticInhibition of GeneticInhibitionInteraction
    | Reaction of ReactionInteraction
    | GenericActivation of GenericInteraction
    | GenericInhibition of GenericInteraction
    
    with
    static member empty = Reaction({properties = InteractionProperties.empty;reactants = []; enzyme = []; products = []})
    static member GetProperties (i:Interaction) = 
        match i with 
        | CodesFor c -> c.properties
        | GeneticActivation ga -> ga.properties
        | GeneticInhibition gi -> gi.properties
        | Reaction rxn -> rxn.properties
        | GenericActivation gi -> gi.properties
        | GenericInhibition gi -> gi.properties

    member this.getProperties = Interaction.GetProperties this
    member this.id = this.getProperties.id
    member this.guid = this.id.ToString() |> System.Guid
    member this.notes = this.getProperties.notes
    member this.deprecated = this.getProperties.deprecated
    static member GetType (i:Interaction) = 
        match i with 
        | CodesFor _ -> "CodesFor"
        | GeneticActivation _ -> "Genetic Activation"
        | GeneticInhibition _ -> "Genetic Inhibition"
        | Reaction _ -> "Reaction"
        | GenericActivation _ -> "Generic Activation"
        | GenericInhibition _ -> "Generic Inhibition"


    member this.getType = Interaction.GetType this
    static member addNotes = "++notes"
    static member removeNotes = "--notes"
    static member addType = "++type"
    static member removeType = "--type"
    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"

type CellCompartment =
    | Chromosome
    | Plasmid
    | Cytosol
    with
    static member toString (comp:CellCompartment) = 
        match comp with 
        | Chromosome -> "Chromosome"
        | Plasmid -> "Plasmid"
        | Cytosol -> "Cytosol" 
    static member fromString (str:string) = 
        match str with 
        | "Chromosome" -> Chromosome 
        | "Plasmid" -> Plasmid
        | "Cytosol" -> Cytosol 
        | _ -> failwithf "%s is unrecognized type of Cell Compartment." str

type CellProperties = { 
    id : CellId
    name:string 
    notes: string
    barcode : Barcode option //if None, the construct is virtual
    genotype: string
    deprecated:bool
    } with
    static member empty = 
        { id = CellId.Create()
          name = ""
          notes = ""
          barcode = None
          genotype = ""
          deprecated = false
        }

type CellEntity = {
    cellId:CellId
    compartment:CellCompartment
    entity:ReagentId
    } with 
    static member empty = {cellId = CellId.Create();compartment=CellCompartment.Plasmid;entity = GenericEntityId.Create() |> GenericEntityReagentId}
    
    static member decode:Decoder<CellEntity> =
        Decode.object(fun get ->
            {
                cellId = get.Required.Field "cellId" Decode.string |> System.Guid |> CellId
                compartment = get.Required.Field "compartment" Decode.string |> CellCompartment.fromString
                entity =
                    let entity_guid = (get.Required.Field "entityId" Decode.string) |> System.Guid
                    match (get.Required.Field "entityType" Decode.string) with
                    | "DNA" -> entity_guid |> DNAId |> DNAReagentId
                    | "RNA" -> entity_guid |> RNAId |> RNAReagentId
                    | "Chemical" -> entity_guid |> ChemicalId |> ChemicalReagentId
                    | "Protein" -> entity_guid |> ProteinId |> ProteinReagentId
                    | "GenericEntity" -> entity_guid |> GenericEntityId |> GenericEntityReagentId
                    | _ -> failwithf "%s is an unknown Reagent type." (get.Required.Field "entityType" Decode.string)

            })

    static member encode (cellentity:CellEntity) = 
        Encode.object [
            "cellId", Encode.string (cellentity.cellId.ToString())
            "compartment", Encode.string (CellCompartment.toString cellentity.compartment)
            "entityId", Encode.string (cellentity.entity.ToString())
            "entityType", Encode.string (ReagentId.GetType cellentity.entity)
        ]    

    static member addCompartment = "++compartment"
    static member removeCompartment = "--compartment"
    static member addEntity = "++entity"
    static member removeEntity ="--entity"
    static member targetEntity = "entity"
    
    static member addEntityType = "++entityType"
    static member removeEntityType ="--entityType"
    static member targetEntityType = "entityType"

type ProkaryoteType = 
    | Bacteria
    //| Archaea
    with 
    static member ToString (pt:ProkaryoteType)= 
        match pt with 
        | Bacteria -> "Bacteria"
    static member FromString (str:string) = 
        match str with 
        | "Bacteria" -> Bacteria
        | _ -> failwithf "%s is an unknown Prokaryote Type" (str)

type ProkaryoteCellStrain = {
    properties : CellProperties
    Type: ProkaryoteType
}

type EukaryoteCellStrain = {
    properties : CellProperties
    //Type: EukaryoteType
    }

type Cell = 
    | Prokaryote of ProkaryoteCellStrain
    | Eukaryote of EukaryoteCellStrain   
    
    static member empty = Prokaryote({Type= Bacteria; properties = CellProperties.empty})

    static member emptyWithId (cellId:CellId) = Prokaryote {Type= Bacteria; properties = {CellProperties.empty with id = cellId}}

    static member GetProperties (c:Cell) = 
        match c with 
        | Prokaryote p -> p.properties
        | Eukaryote p -> p.properties

    //Cells that can be used in a TEST protocol
    //TODO: can we characterize linear DNA or cell-free DNA (e.g. in TX/TL)
    static member IsTestable (cell:Cell) =     
        cell.getProperties.barcode.IsSome  //must be a physical part (i.e. it exists in a test tube)
        //(reagent.Type = SourcePlasmidDNA || reagent.Type = AssembledPlasmidDNA)  && //must be a plasmid DNA part?
        ////reagent.context.IsSome &&                                                   //must be cloned inside cells

    member private this.getProperties = Cell.GetProperties this
    
    member this.id = this.getProperties.id
    member this.name = this.getProperties.name
    member this.barcode = this.getProperties.barcode
    member this.genotype = this.getProperties.genotype
    member this.notes = this.getProperties.notes
    member this.deprecated = this.getProperties.deprecated

    static member SetName (cell:Cell) (name:string) = 
        match cell with
        | Prokaryote p -> Prokaryote {p with properties = {p.properties with name=name}}
        | Eukaryote p -> Eukaryote {p with properties = {p.properties with name=name}}

    static member SetDeprecated (cell:Cell) (value:bool) = 
        match cell with
        | Prokaryote p -> Prokaryote {p with properties = {p.properties with deprecated=value}}
        | Eukaryote p -> Eukaryote {p with properties = {p.properties with deprecated=value}}

    static member WithNotes (notes:string) (cell:Cell) = 
        match cell with
        | Prokaryote p -> Prokaryote {p with properties = {p.properties with notes=notes}}
        | Eukaryote p -> Eukaryote {p with properties = {p.properties with notes=notes}}

    static member WithBarcode (barcode:Barcode option) (cell:Cell) = 
        match cell with
        | Prokaryote p -> Prokaryote {p with properties = {p.properties with barcode=barcode}}
        | Eukaryote p -> Eukaryote {p with properties = {p.properties with barcode=barcode}}

    static member decode:Decoder<Cell> =
        Decode.object(fun get ->
            let cell_type = get.Required.Field "type" Decode.string
            let cell_properties:CellProperties = {
                id = (get.Required.Field "id" Decode.string) |> System.Guid |> CellId
                name = get.Required.Field "name" Decode.string
                notes = get.Optional.Field "notes" Decode.string
                         |> Option.defaultValue ""
                barcode = 
                    let b = get.Optional.Field "barcode" Decode.string
                    match b with | Some(bval) -> bval |> Barcode |> Some | None -> None
                genotype = get.Optional.Field "genotype" Decode.string
                            |> Option.defaultValue ""
                deprecated = get.Optional.Field "deprecated" Decode.bool
                              |> Option.defaultValue false
            }
            match cell_type with 
            | "Bacteria" -> {properties = cell_properties; Type = ProkaryoteType.FromString cell_type } |> Prokaryote
            | _ -> failwithf "Cell strain of type %s not implemented yet." (cell_type)
            )

    static member encode (cell:Cell) = 
        let typeString = 
            match cell with 
            | Prokaryote(pcs) -> 
                match pcs.Type with 
                | Bacteria -> "Bacteria"
            | _ -> failwith "Not implemented yet"
        Encode.object [
            "id", Encode.string (cell.id.ToString())
            "name", Encode.string (cell.name)
            "notes", Encode.string (cell.notes)
            match cell.barcode with | Some(b) -> "barcode", Encode.string (b.ToString()) | None -> ()
            "genotype", Encode.string (cell.genotype)
            "deprecated", Encode.bool (cell.deprecated)
            "type", Encode.string typeString
        ]

    static member addTag = "++tag"
    static member removeTag = "--tag"

    static member addName = "++name"
    static member removeName = "--name"
    static member addDerivedFrom = "++derivedFrom"
    static member removedDerivedFrom = "--derivedFrom"
    static member addNotes = "++notes"
    static member removeNotes = "--notes"
    static member addBarcode = "++barcode"
    static member removeBarcode = "--barcode"
    static member addGenotype = "++genotype"
    static member removeGenotype = "--genotype"
    
    static member addType = "++type"
    static member removeType = "--type"
    static member addProkaryoteType = "++prokaryoteType"
    static member removeProkaryoteType = "--prokaryoteType"

    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"

    static member AnthaName (c:Cell) =  sprintf "%s %s" c.getProperties.name (c.getProperties.id.ToString())
    
    member this.anthaName = Cell.AnthaName this
    
    static member GetType (c:Cell) = match c with | Prokaryote _ -> "Prokaryote" | Eukaryote _ -> "Eukaryote"
    
    member this.getType = Cell.GetType this
    member this.guid = this.id.ToString() |> System.Guid

type DerivedFrom = 
    | CellLineage        of mother:CellId * daughter:CellId
    | DNAComponent       of source:DNAId  * target:DNAId           //The components that were used to build a DNA construct
    | DNAAssembly        of experiment:ExperimentId * target:DNAId //the experiment that produced the DNA circuit
    | CellTransformation of experiment:ExperimentId * target:CellId
    with
    static member empty = CellLineage(CellId.Create(), CellId.Create())
    static member GetType (df:DerivedFrom) = 
        match df with 
        | CellLineage        _ -> "CellLineage"       
        | DNAComponent       _ -> "DNAComponent"      
        | DNAAssembly        _ -> "DNAAssembly"       
        | CellTransformation _ -> "CellTransformation"
    static member GetSourceGuid (df:DerivedFrom) = 
        match df with 
        | CellLineage        (CellId guid, _) -> guid 
        | DNAComponent       (DNAId guid, _) -> guid
        | DNAAssembly        (ExperimentId guid, _) -> guid 
        | CellTransformation (ExperimentId guid, _) -> guid
    static member GetTargetGuid (df:DerivedFrom) = 
        match df with 
        | CellLineage        (_ , CellId guid) -> guid 
        | DNAComponent       (_ , DNAId guid) -> guid
        | DNAAssembly        (_ , DNAId guid) -> guid 
        | CellTransformation (_ , CellId guid) -> guid
    static member addSource = "++source"
    static member removeSource = "--source"
    static member addTarget = "++target"
    static member removeTarget = "--target"
    static member addType = "++type"
    static member removeType = "--type"
    static member fromType (source:System.Guid) (target:System.Guid) (str:string) = 
        match str with
        | "CellLineage"          -> CellLineage        (CellId source, CellId target)
        | "DNAComponent"         -> DNAComponent       (DNAId source, DNAId target)
        | "DNAAssembly"          -> DNAAssembly        (ExperimentId source, DNAId target)
        | "CellTransformation"   -> CellTransformation (ExperimentId source, CellId target)
        | _                      -> failwithf "Unknown derived rel. type %A" str 

//TODO: define these in the BCKG?
type PlateReaderFilter = 
    | PlateFilter_430_10
    | PlateFilter_480_10
    | PlateFilter_500_10
    | PlateFilter_530
    | PlateFilter_550_10
    | PlateFilter_610_20
    | PlateFilter_485_12
    | PlateFilter_520    
    static member toString (f:PlateReaderFilter) = 
        match f with 
        | PlateFilter_430_10 -> "430-10"
        | PlateFilter_480_10 ->"480-10"
        | PlateFilter_500_10 -> "500-10"
        | PlateFilter_530 -> "530"
        | PlateFilter_550_10 -> "550-10"
        | PlateFilter_610_20 -> "610-20"
        | PlateFilter_485_12 -> "485-12"
        | PlateFilter_520    -> "520"        
    static member fromString (s:string) = 
        match s with 
        | "430-10"-> PlateFilter_430_10
        | "480-10"-> PlateFilter_480_10
        | "500-10"-> PlateFilter_500_10
        | "530"-> PlateFilter_530   
        | "550-10"-> PlateFilter_550_10
        | "610-20"-> PlateFilter_610_20
        | "485-12" -> PlateFilter_485_12
        | "520"  -> PlateFilter_520
        | _ -> failwithf "Unknown plate reader filter %s" s

    static member decode:Decoder<PlateReaderFilter> = 
        Decode.object(fun get -> 
            let midpoint = get.Required.Field "midpoint" Decode.float
            let width_opt = get.Optional.Field "width" Decode.float
            let filter_str =
                match width_opt with
                | Some(width) -> sprintf "%s-%s" (midpoint.ToString()) (width.ToString())
                | None -> sprintf "%s" (midpoint.ToString())
            PlateReaderFilter.fromString filter_str)

    static member encode (prf:PlateReaderFilter) = 
        let prf_str = (PlateReaderFilter.toString prf).Split('-')
        let midpoint,width = 
            match prf_str.Length with
            | 1 -> Double.Parse(prf_str.[0]), None
            | 2 -> Double.Parse(prf_str.[0]), Some(Double.Parse(prf_str.[1]))
            | _ -> failwith "Malformed request?"
        Encode.object [
            "midpoint", Encode.float midpoint
            match width with | Some(w) -> "width", Encode.float w | None -> ()
        ]

type PlateReaderFluorescenceSettings = 
    { emissionFilter: PlateReaderFilter
      excitationFilter: PlateReaderFilter
      gain : float
    }
    static member Create(excitation, emission, gain) = 
        { emissionFilter = emission
          excitationFilter = excitation
          gain = gain
        }

    static member decode:Decoder<PlateReaderFluorescenceSettings> =     
        Decode.object(fun get -> 
            {
                emissionFilter = get.Required.Field "emissionFilter" PlateReaderFilter.decode
                excitationFilter = get.Required.Field "excitationFilter" PlateReaderFilter.decode
                gain = get.Optional.Field "gain" Decode.float
                        |> Option.defaultValue -1.0
            })

    static member encode (base_properties: (string*JsonValue) list) (prfs:PlateReaderFluorescenceSettings) = 
        let prfs_properties = [
            "type", "PlateReaderFluorescence" |> Encode.string
            "emissionFilter", prfs.emissionFilter |> PlateReaderFilter.encode
            "excitationFilter", prfs.excitationFilter |> PlateReaderFilter.encode
            match prfs.gain with | -1.0 -> () | _ -> "gain", prfs.gain |> Encode.float
        ]
        Encode.object (base_properties@prfs_properties)

type PlateReaderAbsorbanceSettings = 
    { wavelength : float //nm
      gain : float
      correction: float  //path correction
    }
    static member Create (wavelength, gain, correction) = 
        { wavelength = wavelength
          gain = gain
          correction = correction
        }

    static member decode:Decoder<PlateReaderAbsorbanceSettings> = 
        Decode.object(fun get -> 
            {
                wavelength = get.Required.Field "wavelength" Decode.float
                correction = get.Required.Field "correction" Decode.float
                gain = get.Optional.Field "gain" Decode.float
                        |> Option.defaultValue -1.0
            })
    
    static member encode (base_properties: (string*JsonValue) list) (pras:PlateReaderAbsorbanceSettings) = 
        let pras_properties = [
            "type", "PlateReaderAbsorbance" |> Encode.string
            "wavelength", pras.wavelength |> Encode.float
            "correction", pras.correction |> Encode.float
            match pras.gain with | -1.0 -> () | _ -> "gain", pras.gain |> Encode.float
        ]
        Encode.object (base_properties@pras_properties)

type SignalSettings = 
    | PlateReaderFluorescence of PlateReaderFluorescenceSettings
    | PlateReaderAbsorbance of PlateReaderAbsorbanceSettings
    | PlateReaderTemperature   
    | PlateReaderLuminescence //TODO: define settings
    | Titre
    | GenericSignal of string
    
    static member toTypeString (t:SignalSettings) = 
        match t with 
        | PlateReaderFluorescence _ -> "PlateReaderFluorescence"
        | PlateReaderAbsorbance _ -> "PlateReaderAbsorbance"
        | PlateReaderTemperature -> "PlateReaderTemperature"
        | PlateReaderLuminescence -> "PlateReaderLuminescence"
        | Titre -> "Titre"
        | GenericSignal x -> sprintf "Generic:%s" x

    static member toString (s:SignalSettings) = 
        match s with 
        | PlateReaderFluorescence ss -> sprintf "PlateReaderFluorescence (%s/%s)" (PlateReaderFilter.toString ss.excitationFilter) (PlateReaderFilter.toString ss.emissionFilter)
        | PlateReaderAbsorbance ss -> sprintf "PlateReaderAbsorbance (%.0fnm)" ss.wavelength
        | PlateReaderTemperature -> "PlateReaderTemperature"
        | PlateReaderLuminescence -> "PlateReaderLuminescence"
        | Titre -> "Titre"
        | GenericSignal x -> sprintf "Generic:%s" x

    static member fromTypeString (s:string) = 
        match s with 
        | "PlateReaderFluorescence" -> (PlateFilter_430_10, PlateFilter_480_10, 1500.0) |> PlateReaderFluorescenceSettings.Create |> PlateReaderFluorescence
        | "PlateReaderAbsorbance" -> (600.0, 658.0, 1.7006) |> PlateReaderAbsorbanceSettings.Create |> PlateReaderAbsorbance
        | "PlateReaderTemperature" -> PlateReaderTemperature
        | "PlateReaderLuminescence" -> PlateReaderLuminescence
        | "Titre" -> Titre
        | x -> 
            let x' = x.Split ':'
            match x'.[0] with 
            | "Generic" -> GenericSignal x'.[1]
            | _ ->             
                failwithf "Unknown signal settings type: %s" s
    
type Signal =  
    { id : SignalId
      settings: SignalSettings
      units : string option
    }
    static member empty = //TODO: meaningful default signal?
        { id = SignalId.Create()
          settings = PlateReaderTemperature
          units = None
        }
    static member toString (s:Signal) = 
        match s.settings with               
        | SignalSettings.PlateReaderFluorescence s ->  
            match s.excitationFilter,s.emissionFilter with
            | PlateFilter_550_10, PlateFilter_610_20 -> "mRFP1"
            | PlateFilter_430_10, PlateFilter_480_10 -> "ECFP"
            | PlateFilter_500_10, PlateFilter_530    -> "EYFP"
            | PlateFilter_485_12, PlateFilter_520    -> "GFP"
            | PlateFilter_485_12, PlateFilter_530    -> "GFP530"
            | _ -> sprintf "F(%s/%s)" (PlateReaderFilter.toString s.excitationFilter) (PlateReaderFilter.toString s.emissionFilter)
        | SignalSettings.PlateReaderAbsorbance s -> 
            if s.wavelength = 600.0 then "OD"
            elif s.wavelength = 700.0 then "OD700"
            else sprintf "OD%.0f" s.wavelength
        | SignalSettings.PlateReaderTemperature -> "Temperature"
        | SignalSettings.PlateReaderLuminescence -> "Luminescence"                
        | SignalSettings.Titre -> "Titre"       
        | SignalSettings.GenericSignal x -> sprintf "%s" x
    
    static member decode:Decoder<Signal> = 
        Decode.object(fun get -> 
            let signalId = get.Required.Field "id" Decode.string |> System.Guid |> SignalId
            let signal_type = get.Required.Field "type" Decode.string
            let settings: SignalSettings = 
                match signal_type with
                | "PlateReaderFluorescence" -> 
                    {
                        emissionFilter = get.Required.Field "emissionFilter" PlateReaderFilter.decode
                        excitationFilter = get.Required.Field "excitationFilter" PlateReaderFilter.decode
                        gain = get.Optional.Field "gain" Decode.float
                                |> Option.defaultValue -1.0
                    } |> PlateReaderFluorescence
                | "PlateReaderAbsorbance" ->  
                    {
                       wavelength = get.Required.Field "wavelength" Decode.float
                       correction = get.Required.Field "correction" Decode.float
                       gain = get.Optional.Field "gain" Decode.float
                                |> Option.defaultValue -1.0
                    } |> PlateReaderAbsorbance
                | "PlateReaderTemperature"-> PlateReaderTemperature
                | "PlateReaderLuminescence" -> PlateReaderLuminescence
                | "Titre" -> Titre
                | "GenericSignal" -> get.Required.Field "name" Decode.string |> GenericSignal
                | _ -> failwithf "%s is not a recognized signal type" signal_type 
            let units = get.Optional.Field "units" Decode.string
            {id = signalId; settings = settings; units = units}) 
    static member encode (signal:Signal) = 
        
        let base_properties = [
            "id", signal.id.ToString() |> Encode.string
            match signal.units with | Some(u) -> "units", u |> Encode.string | None -> () 
        ]
        match signal.settings with 
        | PlateReaderFluorescence(prfs) -> PlateReaderFluorescenceSettings.encode (base_properties) (prfs)
        | PlateReaderAbsorbance(pras) -> PlateReaderAbsorbanceSettings.encode (base_properties) (pras)
        | PlateReaderTemperature ->  ("type","PlateReaderTemperature" |> Encode.string)::base_properties |> Encode.object
        | PlateReaderLuminescence ->  ("type","PlateReaderLuminescence" |> Encode.string)::base_properties |> Encode.object
        | Titre ->  ("type","Titre" |> Encode.string)::base_properties |> Encode.object
        | GenericSignal name ->  
            [
                "type","PlateReaderTemperature" |> Encode.string
                "name", name |> Encode.string
            ]@base_properties 
            |> Encode.object
        
type TimeSeries =
    { times : Time []
      observations : Map<SignalId, float []>
    }
    static member toString (ts:TimeSeries)=
        let keys = ts.observations |> Map.toArray |> Array.map fst
        
        ts.times
        |> Array.mapi(fun i t -> 
            keys
            |> Array.map(fun k -> ts.observations.[k].[i] |> sprintf "%f")
            |> String.concat ","
            |> sprintf "%f,%s" (Time.getHours t) //TODO: Serialize with units?
            )
        |> String.concat "\n"
        |> sprintf "time,%s\n%s" (keys |> Array.map SignalId.toString |> String.concat ",")

    static member fromString (s:string) =         
        let lines = s.Split([|'\n'; 'r'|], StringSplitOptions.RemoveEmptyEntries)
        let headers = lines.[0].Split(',').[1..]
        let data = 
            lines.[1..]
            |> Array.map(fun l -> l.Split(',') |> Array.map (float))

        let observations = 
            headers 
            |> Array.mapi(fun i h -> //TODO: Warnings on ignored signal columns?                
                SignalId.fromString h
                |> Option.map(fun signalId ->                                                 
                        let signalData = data |> Array.map(fun d -> d.[i+1])                
                        (signalId, signalData)
                        )
                )
            |> Array.choose id //select only the columns that parse to correct signal IDs
            |> Map.ofSeq

        { times = data |> Array.map(fun d -> Hours d.[0])
          observations = observations
        }            

type PlateReaderMetaData =
    { virtualWell : Position
      physicalWell : Position option
      physicalPlateName : string option
    }    
    static member Create(well:Position) = 
        { virtualWell = well
          physicalWell = None
          physicalPlateName = None
        }
    static member decode: Decoder<PlateReaderMetaData> =
        Decode.object(fun get ->
            {
                virtualWell =  get.Required.Field "virtualWell" Position.decode
                physicalWell = get.Optional.Field "physicalWell" Position.decode
                physicalPlateName = get.Optional.Field "physicalPlateName" Decode.string
            })
    
    static member encode (metadata: PlateReaderMetaData) = 
        let requiredFields = [ 
            "virtualWell", Position.encode (metadata.virtualWell)
            match metadata.physicalWell with | Some(pw) -> "physicalWell", Position.encode pw | None -> ()
            match metadata.physicalPlateName with | Some(pn) -> "physicalPlateName", Encode.string pn | None -> ()
            ]
        Encode.object requiredFields

type SampleMeta = 
    | PlateReaderMeta of PlateReaderMetaData
    | MissingMeta

    static member toStringType (s:SampleMeta) = 
        match s with 
        | PlateReaderMeta _ -> "PlateReaderMeta"
        | MissingMeta -> "MissingMeta"

    static member fromStringType (s:string) = 
        match s with 
        | "PlateReaderMeta" -> PlateReaderMetaData.Create({row=0; col=0}) |> PlateReaderMeta
        | "MissingMeta" -> MissingMeta
        | _ -> failwithf "Unknown sample metadata type: %s" s

type SampleDevice = 
    {
        cellId: CellId
        sampleId: SampleId
        cellDensity: float option
        cellPreSeeding: float option
    } with
    static member empty = 
        {
        cellId = CellId.Create()
        sampleId = SampleId.Create()
        cellDensity = None
        cellPreSeeding = None
        }

    static member Create (cellId: CellId) (sampleId:SampleId) (cellDensity:float option) (cellPreSeeding:float option): SampleDevice =
        {cellId=cellId; sampleId=sampleId; cellDensity=cellDensity; cellPreSeeding=cellPreSeeding}

    static member decode: Decoder<SampleDevice> =
        Decode.object(fun get ->
            {  cellId=get.Required.Field "cellId" Decode.string |> System.Guid |> CellId
               sampleId=get.Required.Field "sampleId" Decode.string |> System.Guid |> SampleId
               cellDensity=get.Optional.Field "cellDensity" Decode.float
               cellPreSeeding=get.Optional.Field "cellPreSeeding" Decode.float
            }
        )
    
    static member encode (sampleDevice: SampleDevice) = 
        let requiredFields = 
            [ "cellId", Encode.string (sampleDevice.cellId.ToString()) 
              "sampleId", Encode.string (sampleDevice.sampleId.ToString())
              match sampleDevice.cellDensity with | Some(c) -> "cellDensity", Encode.float c | None -> ()
              match sampleDevice.cellPreSeeding with | Some(c) -> "cellPreSeeding", Encode.float c | None -> ()
            ]
        Encode.object requiredFields

type Condition = 
    { reagentId: ReagentId
      sampleId: SampleId
      concentration: Concentration
      time : Time option //optional induction time (delay in minutes after the beginning of the experiment)
    }
    static member addReagentId = "++reagentId"
    static member removeReagentId = "--reagentId"
    static member addReagentEntity = "++reagentType"
    static member removeReagentEntity = "--reagentType"
    static member targetReagentId = "reagentId"
    static member targetReagentEntity = "reagentType"
    
    static member addValue = "++value"
    static member removeValue = "--value"
    
    static member addTime = "++time"
    static member removeTime = "--time"
    static member addTimeUnits = "++timeUnits"
    static member removeTimeUnits = "--timeUnits"


    static member Create() = 
           { reagentId = GenericEntityId.Create() |> GenericEntityReagentId
             sampleId = SampleId.Create()
             concentration = UM 0.0
             time = None
           }
    static member Create(reagentId, sampleId, conc) = 
        { reagentId = reagentId
          sampleId = sampleId
          concentration = conc
          time = None
        }
    static member Create(reagentId, sampleId, conc, time) = 
        { reagentId = reagentId
          sampleId = sampleId
          concentration = conc
          time = time
        }
    static member Create(reagentId, conc, time) = 
        { reagentId = reagentId
          sampleId = SampleId.Create()
          concentration = conc
          time = time
        }

    static member decode: Decoder<Condition> =
        Decode.object(fun get ->
            let reagentGuid = get.Required.Field "reagentId" Decode.string |> System.Guid
            let reagentType = get.Required.Field "reagentType" Decode.string
            {  reagentId = ReagentId.FromType reagentGuid reagentType
               sampleId = get.Required.Field "sampleId" Decode.string |> System.Guid |> SampleId
               concentration = get.Required.Field "concentration" Concentration.decode
               time = get.Optional.Field "time" Time.decode
            })
    
    static member encode (condition: Condition) = 
        let requiredFields = 
            [ 
                "reagentId", Encode.string (condition.reagentId.ToString())
                "reagentType", Encode.string (ReagentId.GetType condition.reagentId)
                "sampleId",  Encode.string (condition.sampleId.ToString())
                "concentration", Concentration.encode condition.concentration
                match condition.time with | Some(t) -> "time", Time.encode t | None -> ()
            ]
        Encode.object requiredFields

type Sample =
    { id : SampleId      
      experimentId: ExperimentId
      meta : SampleMeta
      deprecated : bool
    } with 
    static member empty :Sample = 
        { id = SampleId.Create()
          experimentId = ExperimentId.Create() //TODO: not attached to a parent?
          meta = MissingMeta
          deprecated = false
        }
    static member addExperimentId = "++experimentId"
    static member removeExperimentId = "--experimentId"
    static member addMetaType = "++metaType"
    static member removeMetaType = "--metaType"
    static member addVirtualWellRow = "++virtualWellRow"
    static member removeVirtualWellRow = "--virtualWellRow"
    static member addVirtualWellCol = "++virtualWellCol"
    static member removeVirtualWellCol = "--virtualWellCol"
    static member addPhysicalPlateName = "++physicalPlateName"
    static member removePhysicalPlateName = "--physicalPlateName"
    static member addPhysicalWellRow = "++physicalWellRow"
    static member removePhysicalWellRow = "--physicalWellRow"
    static member addPhysicalWellCol = "++physicalWellCol"
    static member removePhysicalWellCol = "--physicalWellCol"
    static member addCellId = "++cellId"
    static member removeCellId = "--cellId"
    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"
    static member addTag = "++tag"
    static member removeTag = "--tag"
    static member addReplicate = "++replicate"
    static member removeReplicate = "--replicate"

    member this.guid = this.id.ToString() |> System.Guid

    static member decode: Decoder<Sample> =
        Decode.object(fun get ->
            let metadata =
                match get.Optional.Field "meta" PlateReaderMetaData.decode with
                | None -> MissingMeta
                | Some x -> x |> PlateReaderMeta
            {
                id =  get.Required.Field "id" Decode.string |> System.Guid |> SampleId
                experimentId = get.Required.Field "experimentId"  Decode.string |> System.Guid |> ExperimentId
                meta =  metadata
                deprecated = get.Required.Field "deprecated" Decode.bool
            })
    
    static member encode (sample: Sample) = 
        let requiredFields = 
            [ "id", Encode.string (sample.id.ToString())
              "experimentId", Encode.string (sample.experimentId.ToString())
              match sample.meta with | PlateReaderMeta(prmd) -> "meta", PlateReaderMetaData.encode prmd | MissingMeta -> ()  
              "deprecated", Encode.bool (sample.deprecated)
            ]
        Encode.object requiredFields

type Observation = 
    { id        : ObservationId
      sampleId  : SampleId
      signalId  : SignalId
      value     : float
      timestamp : System.DateTime option
      replicate : ReplicateId option
    }
    static member empty:Observation = 
        { id        = ObservationId.Create()
          sampleId  = SampleId.Create()
          signalId  = SignalId.Create()
          value     = 0.0
          timestamp = None
          replicate = None
        }
    static member addSampleId  = "++sampleId"
    static member addSignalId  = "++signalId"
    static member addValue     = "++value"
    static member addTimestamp = "++timestamp"
    static member addReplicate = "++replicate"

    static member decode: Decoder<Observation> =
        Decode.object(fun get ->
            let timestamp = 
                match  get.Optional.Field "timestamp" Decode.datetime with
                | None -> None
                | Some x -> x |> Some
            let replicate =
                match get.Optional.Field "replicate" Decode.string with
                | None -> None
                | Some x -> x |> System.Guid |> ReplicateId |> Some
            { id = get.Required.Field "id" Decode.string |> System.Guid  |> ObservationId
              sampleId = get.Required.Field "sampleId" Decode.string |> System.Guid  |> SampleId
              signalId = get.Required.Field "signalId" Decode.string |> System.Guid  |> SignalId
              value = get.Required.Field "value" Decode.float
              timestamp = timestamp
              replicate = replicate            })

    static member encode (obs:Observation) =
        let requiredFields = 
            [ "id", Encode.string (obs.id.ToString())
              "sampleId", Encode.string (obs.sampleId.ToString())
              "signalId", Encode.string (obs.signalId.ToString())
              "value", Encode.float (obs.value)
              match obs.replicate with | Some(a) -> "replicate", Encode.string (a.ToString()) | None -> ()
              match obs.timestamp with | Some(b) -> "timestamp", Encode.datetime b | None -> ()
            ]
        Encode.object requiredFields

//NOTE: samples associated with the experiment are handled separately
type Experiment = 
    {id : ExperimentId
     name : string
     notes : string    
     Type : ExperimentType
     //scale: ExperimentScale #TODO: Add this 
     deprecated : bool
    }
    static member empty = 
        {id = ExperimentId.Create()
         name = ""
         notes = ""                  
         Type = TestExperiment
         deprecated = false
        }

    static member decode:Decoder<Experiment> = 
        Decode.object(fun get -> 
            let expt_id = get.Required.Field "id" Decode.string |> System.Guid |> ExperimentId
            let name = get.Required.Field "name" Decode.string
            let notes = get.Optional.Field "notes" Decode.string 
                         |> Option.defaultValue ""
            let deprecated = get.Optional.Field "deprecated" Decode.bool
                              |> Option.defaultValue false
            let type_str = get.Required.Field "type" Decode.string
            let expt_type = 
                match type_str with
                | "BuildExperiment" -> ExperimentType.BuildExperiment
                | "TestExperiment" -> ExperimentType.TestExperiment
                | _ -> failwithf "%s is not a recognized type of Experiment" (type_str)
            {id = expt_id; Type = expt_type; deprecated = deprecated; name = name; notes = notes})
    static member encode (expt:Experiment) = 
        let base_properties = [
            "id", expt.id.ToString() |> Encode.string
            "name", expt.name |> Encode.string
            "notes", expt.notes |> Encode.string
            "deprecated", expt.deprecated |> Encode.bool
        ]
        match expt.Type with 
        | BuildExperiment -> ("type", "BuildExperiment" |> Encode.string)::base_properties |> Encode.object
        | TestExperiment -> ("type", "TestExperiment" |> Encode.string)::base_properties |> Encode.object
    
    static member addName = "++name"
    static member removeName = "--name"
    static member addType = "++type"
    static member removeType = "--type"
    static member addNotes = "++notes"
    static member removeNotes = "--notes"
    static member addTag = "++tag"
    static member removeTag = "--tag"
    static member addDeprecate = "++deprecate"
    static member removeDeprecate = "--deprecate"
    
    member this.guid = this.id.ToString() |> System.Guid


    //TODO: check for event consistency wrt order?
    static member getProgress (e:Experiment) (operations:ExperimentOperation[]) =                
        let eventsOrder = 
            match e.Type with 
            | BuildExperiment -> ExperimentOperationType.BuildProtocolEventsOrder
            | TestExperiment -> ExperimentOperationType.TestProtocolEventsOrder
            |> Array.ofSeq
        
        let m = eventsOrder.Length            
            
        if Seq.isEmpty operations then 
            m, 0, None, Some ExperimentStarted
        else
            let lastEvent = (operations |> Array.sortBy (fun ev -> ev.timestamp) |> Array.last).Type            
            let n = eventsOrder |> Array.findIndex ( (=) lastEvent)                
            let nextEvent = 
                if n+1< eventsOrder.Length then 
                    Some eventsOrder.[n+1]
                else
                    None
            m, (n+1), Some lastEvent, nextEvent


//Minimal index used for tools like MSGC            
//TODO: Deprecate and replace with MemoryInstance or server calls (Task #10672)
type KnowledgeGraph = {
    partsMap       : Map<PartId, Part> //a collection of known DNA parts (e.g. biobricks). Stored as a map from part names to parts //TODO: index from ids?
    experimentsMap : Map<ExperimentId, Experiment>
    reagentsMap    : Map<ReagentId, Reagent>       
    samplesMap     : Map<SampleId, Sample>    
    signalsMap     : Map<SignalId, Signal>
    cellsMap       : Map<CellId,Cell> 


    //Depricated structures (previously stored in separate entities)
    tags                 : Map<System.Guid, Tag[]>
    experimentOperations : Map<ExperimentId, ExperimentOperationId[]>
    operationsMap        : Map<ExperimentOperationId, ExperimentOperation>
    experimentSignals    : Map<ExperimentId, SignalId[]>    
    sampleConditions     : Map<SampleId, Condition[]>
    sampleCells          : Map<SampleId, (CellId * (float option * float option))[]> //cell id to cell density (if available)  
    //experimentSamples    : Map<ExperimentId, Set<SampleId>>
    sampleFiles          : Map<SampleId, FileRef[]>
    cellEntities         : Map<CellId, CellEntity[]>
    observations         : Map<SampleId*SignalId, Observation[]>
    }  with
    static member empty = 
        { partsMap       = Map.empty
          experimentsMap = Map.empty
          reagentsMap    = Map.empty                          
          signalsMap     = Map.empty
          samplesMap     = Map.empty
          cellsMap       = Map.empty

          tags                 = Map.empty
          experimentOperations = Map.empty          
          operationsMap        = Map.empty          
          sampleConditions     = Map.empty
          sampleCells          = Map.empty
          experimentSignals    = Map.empty
          //experimentSamples    = Map.empty
          sampleFiles          = Map.empty
          cellEntities         = Map.empty
          observations         = Map.empty
        }
        
    //Safe getters
    member this.GetTags (guid:System.Guid) = if this.tags.ContainsKey guid then this.tags.[guid] else Array.empty
    member this.GetExperimentOperations (experimentId:ExperimentId) = if this.experimentOperations.ContainsKey experimentId then this.experimentOperations.[experimentId] else Array.empty
    member this.GetSampleCells (sampleId:SampleId) = if this.sampleCells.ContainsKey sampleId then this.sampleCells.[sampleId] else Array.empty
    member this.GetSampleConditions (sampleId:SampleId) = if this.sampleConditions.ContainsKey sampleId then this.sampleConditions.[sampleId] else Array.empty
    member this.GetExperimentSignals (experimentId:ExperimentId) = if this.experimentSignals.ContainsKey experimentId then this.experimentSignals.[experimentId] else Array.empty
    member this.GetCellEntities (cellId:CellId) = if this.cellEntities.ContainsKey cellId then this.cellEntities.[cellId] else Array.empty

    member this.GetDnaOrFail (i:ReagentId) = 
        match this.reagentsMap.TryFind i with
        | Some reagent -> 
            match reagent with 
            | DNA r -> r
            | _ -> failwithf "Reagent %A is not DNA" i
        | None -> failwithf "Reagent %A was not found" i

    member this.GetReagentOrFail (i:ReagentId) = 
        match this.reagentsMap.TryFind i with
        | Some reagent -> reagent
        | None -> failwithf "Reagent %A was not found" i

     member this.GetPartOrFail (i:PartId) = 
        match this.partsMap.TryFind i with
        | Some part -> part
        | None -> failwithf "Part %A was not found" i
   
     member this.ReagentNames = 
        this.reagentsMap
        |> Map.map(fun _ r -> r.anthaName)

     member this.CellNames = 
        this.cellsMap 
        |> Map.map(fun _ r -> r.anthaName)

    member this.ShortReagentNames = 
        this.reagentsMap 
        |> Map.map(fun _ r -> r.getProperties.name)

     member this.ShortCellNames = 
        this.cellsMap 
        |> Map.map(fun _ c -> c.name)

     member this.experiments = this.experimentsMap |> Map.toArray |> Array.map snd
     
     member this.parts = this.partsMap |> Map.toArray |> Array.map snd
     
     member this.reagents = this.reagentsMap |> Map.toArray |> Array.map snd

     member this.dnaReagents = 
        this.reagentsMap 
        |> Map.toArray
        |> Array.choose(fun (_,reagent) -> 
            match reagent with 
            | DNA device -> Some device
            | _ -> None
        )      

     member this.media = 
        this.reagentsMap 
        |> Map.filter(fun _ reagent -> 
            match reagent with 
            | Chemical chem -> 
                match chem.Type with 
                | Media -> true 
                | _ -> false
            | _ -> false)         

     member this.smallMolecules = 
         this.reagentsMap 
         |> Map.filter(fun _ reagent -> 
             match reagent with 
             | Chemical chem -> 
                 match chem.Type with 
                 | SmallMolecule -> true 
                 | _ -> false
             | _ -> false)         

     member this.antibiotics = 
         this.reagentsMap 
         |> Map.filter(fun _ reagent -> 
             match reagent with 
             | Chemical chem -> 
                 match chem.Type with 
                 | Antibiotic -> true 
                 | _ -> false
             | _ -> false)         


     static member GetDnaDevices (devices:Map<ReagentId, Reagent>) =      
        devices 
        |> Map.toArray
        |> Array.choose (fun (_, d) -> 
            match d with 
            | DNA r -> Some r
            | _ -> None
            )

     member this.samples = this.samplesMap |> Map.toArray |> Array.map snd

     member this.cells = this.cellsMap |> Map.toArray |> Array.map snd
    
type TempEventTuple = string * System.DateTime * string * (string * string) list


module PlateTagging = 
    let maxVirtualWells = 100
    let columnHeaders = Array.init maxVirtualWells (fun i -> sprintf "%i" (i+1))
    let rowHeaders = //TODO: Fix bug with X; Y; Z; BA; ...
        let toChar n = System.Convert.ToChar(65+n) |>sprintf "%c"            
        let rec toString n :string = 
            let m = 26
            if n>=m then (toString (n/m)) + (toChar (n%m))
            else (toChar (n%m))
        Array.init maxVirtualWells toString                                            

    let allHeaders = Array.concat [columnHeaders;rowHeaders]

    let makeSequence (from:string) (until:string) =

        let fromIndexRow = Array.findIndex ((=) (from.Substring(0,1))) allHeaders
        let untilIndexRow = Array.findIndex ((=) (until.Substring(0,1))) allHeaders
        //let rangeRow = Array.sub allHeaders fromIndexRow (untilIndexRow - fromIndexRow)

        let fromIndexCol = Array.findIndex ((=) (from.[1..])) allHeaders
        let untilIndexCol = Array.findIndex ((=) (until.[1..])) allHeaders
        //let rangeCol = Array.sub allHeaders fromIndexCol (untilIndexCol - fromIndexCol)

        seq {
            for row in fromIndexRow..untilIndexRow do
                for col in fromIndexCol..untilIndexCol do
                    yield allHeaders.[row], allHeaders.[col]
        }

    let makeRowsSequence (from:string) (until:string) =

        let fromIndexRow = Array.findIndex ((=) (from.Substring(0,1))) allHeaders
        let untilIndexRow = Array.findIndex ((=) (until.Substring(0,1))) allHeaders
        //let rangeRow = Array.sub allHeaders fromIndexRow (untilIndexRow - fromIndexRow)

        let fromIndexCol = Array.findIndex ((=) (from.[1..])) allHeaders
        let untilIndexCol = Array.findIndex ((=) (until.[1..])) allHeaders
        //let rangeCol = Array.sub allHeaders fromIndexCol (untilIndexCol - fromIndexCol)

        [fromIndexRow..untilIndexRow]
        |> List.map(fun row -> 
            [fromIndexCol..untilIndexCol]
            |> List.map(fun col -> 
                allHeaders.[row], allHeaders.[col]
            ))
    
    type TreatmentMap = Map<string*string, Condition list>
    type Plate = 
        { Genotypes  : Map<string*string, List<CellId>>
          Treatments : TreatmentMap
          sampleIds  : Map<string*string, SampleId>
        }    
        member this.UsedWells =         
            let NonEmptyLabels (x:Map<'a,List<'b>>) = x |> Map.toArray |> Array.filter (snd>>List.isEmpty>>not) |> Array.map fst |> Set.ofArray        
            (NonEmptyLabels this.Treatments) + (NonEmptyLabels this.Genotypes)
            |> Set.toArray

        static member getWell96 n =         
            sprintf "%s%i" rowHeaders.[n/8] (1 + n%8)

        static member getWell12 n =         
            sprintf "%s1" rowHeaders.[n]

        static member FilterZeroSignals (p:Plate) = 
            let treatmentMap =                 
                p.Treatments
                |> Map.toArray
                |> Array.choose(fun (well,treatments) -> 
                    let treatments' = treatments |> List.filter(fun cond -> (Concentration.getValue cond.concentration)<>0.0)
                    if Seq.isEmpty treatments then 
                        None
                    else Some (well,treatments')
                    )
                |> Map.ofSeq
            {p with Treatments = treatmentMap}

        static member empty =
            { Genotypes  = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), List.empty))) |> Map.ofSeq
              Treatments = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), List.empty))) |> Map.ofSeq
              sampleIds  = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), SampleId.Create()))) |> Map.ofSeq
            } 


        member this.AllConditions = 
            this.Treatments |> Map.toList |> List.map snd |> List.concat

    
        member this.SerialDilution addZero (timed:Time option) from until treatment (initial:Concentration) factor = 
            let treatments =     
                let wells = makeSequence from until
                let n = (Seq.length wells) - 2
                
                wells
                |> Seq.indexed
                |> Seq.fold(fun (conc,acc:TreatmentMap) (i,well) ->                        
                    let conc' = 
                        if (addZero && i=n) then Concentration.zero else conc/factor
                    let acc'  = 
                        let sampleId = this.sampleIds.[well]
                        let condition = Condition.Create(treatment, sampleId, conc, timed) 
                        if acc.ContainsKey well then 
                            let t = acc.[well]
                            if t |> Seq.map (fun cond -> cond.reagentId) |> Set.ofSeq |> Set.contains treatment then
                                failwithf "Duplicate definition of treatment with %A" treatment                           
                            let t' = condition::t
                            (acc.Remove well).Add(well,t')
                        else
                            acc.Add(well,[condition])
                                
                    conc', acc'
                    ) (initial, this.Treatments)
                |> snd            
            {this with Treatments = treatments}

        member this.SerialDilution2D addZero (timed:Time option) from until treatment initial factor treatment' initial' factor' = 
            let _, treatments = 
                let rows = makeRowsSequence from until
                let n = (Seq.length rows) - 2
                let m = (rows |> Seq.head |> Seq.length) - 2

                rows
                |> Seq.indexed
                |> Seq.fold(fun (conc1,acc:TreatmentMap) (i,wells) -> 
                    let conc1' = if addZero && i=n then Concentration.zero else conc1/factor
                    let _, A = 
                        wells
                        |> Seq.indexed
                        |> Seq.fold(fun (conc2, acc:TreatmentMap) (j,well) -> 
                            let conc2' = if addZero && j=m then Concentration.zero else conc2/factor'
                            let acc2'  = 
                                let sampleId = this.sampleIds.[well]
                                let condition1 = Condition.Create(treatment, sampleId, conc1, timed)
                                let condition2 = Condition.Create(treatment', sampleId, conc2, timed)
                                if acc.ContainsKey well then 
                                    let t = acc.[well]
                                    if t |> Seq.map (fun cond -> cond.reagentId) |> Set.ofSeq |> Set.contains treatment then
                                        failwithf "Duplicate definition of treatment with %A" treatment
                                    let t' = condition1::condition2::t
                                    (acc.Remove well).Add(well,t')
                                else
                                    acc.Add(well,[condition1; condition2])
                            (conc2',acc2')
                            ) (initial',  acc)   
                                                        
                    conc1', A
                    ) (initial, this.Treatments)

            {this with Treatments = treatments}


                        
        member this.TreatBlock (from:string) (until:string) (reagentId, conc, time) =

            let fromIndexRow = Array.findIndex ((=) (from.Substring(0,1))) allHeaders
            let untilIndexRow = Array.findIndex ((=) (until.Substring(0,1))) allHeaders
            let rangeRow = Array.sub allHeaders fromIndexRow (untilIndexRow - fromIndexRow)

            let fromIndexCol = Array.findIndex ((=) (from.[1..])) allHeaders
            let untilIndexCol = Array.findIndex ((=) (until.[1..])) allHeaders
            let rangeCol = Array.sub allHeaders fromIndexCol (untilIndexCol - fromIndexCol)

            let rowInRange header =
                let rowIndex = Array.findIndex ((=) header) allHeaders
                rowIndex >= fromIndexRow && rowIndex <= untilIndexRow

            let colInRange header =
                let colIndex = Array.findIndex ((=) header) allHeaders
                colIndex >= fromIndexCol && colIndex <= untilIndexCol 

            let treatments' = 
                this.Treatments
                |> Map.map
                    (fun key value ->
                        let row,col = key
                        if rowInRange row && colInRange col then
                            let sampleId = this.sampleIds.[row,col]
                            let cond = Condition.Create(reagentId, sampleId, conc, time)
                            cond :: value
                        else
                            value)

            {this with Treatments = treatments'}


        member this.SetGenotypeBlock (from:string) (until:string) (genotype:CellId) =

            let fromIndexRow = Array.findIndex ((=) (from.Substring(0,1))) allHeaders
            let untilIndexRow = Array.findIndex ((=) (until.Substring(0,1))) allHeaders
            let rangeRow = Array.sub allHeaders fromIndexRow (untilIndexRow - fromIndexRow)

            let fromIndexCol = Array.findIndex ((=) (from.[1..])) allHeaders
            let untilIndexCol = Array.findIndex ((=) (until.[1..])) allHeaders
            let rangeCol = Array.sub allHeaders fromIndexCol (untilIndexCol - fromIndexCol)

            let rowInRange header =
                let rowIndex = Array.findIndex ((=) header) allHeaders
                rowIndex >= fromIndexRow && rowIndex <= untilIndexRow

            let colInRange header =
                let colIndex = Array.findIndex ((=) header) allHeaders
                colIndex >= fromIndexCol && colIndex <= untilIndexCol 

            let newGenotypes = 
                this.Genotypes
                |> Map.map
                    (fun key value ->
                        let row,col = key
                        if rowInRange row && colInRange col then
                            genotype :: value
                        else
                            value)

            {this with Genotypes = newGenotypes }

module CharacterizationSpec = 
    type Range = 
        { from  : string
        ; until : string
        }

    type SerialDilutionOp = 
        { treatment : ReagentId
        ; initial   : Concentration
        ; factor    : float
        }   

    type CharOp = 
        | SerialDilution of SerialDilutionOp
        | Conditions     of ReagentId * Concentration
        | Genotype       of CellId
        | Serial2D       of SerialDilutionOp * SerialDilutionOp //assume 2D are rows and cols
        | TimedSerialDilution of Time * SerialDilutionOp
        | TimedSerial2D       of Time * SerialDilutionOp * SerialDilutionOp //assume 2D are rows and cols
        | TimedConditions     of Time * ReagentId * Concentration

        static member toString (o:CharOp) = 
            match o with 
            | SerialDilution _ -> "SerialDilution"
            | Conditions     _ -> "Conditions"
            | Genotype       _ -> "Genotype"
            | Serial2D       _ -> "Serial2D"
            | TimedSerialDilution _ -> "TimedSerialDilution"
            | TimedSerial2D       _ -> "TimedSerial2D"
            | TimedConditions     _ -> "TimedConditions"
            
        static member fromString defaultReagent defaultDevice (ostr:string) =                   
            let defaultSerial = {treatment=defaultReagent; initial = UM 1.0; factor = 1.0}
            match ostr with 
            | "SerialDilution"   -> SerialDilution defaultSerial
            | "Conditions"       -> Conditions     (defaultReagent, UM 1.0)
            | "Genotype"         -> Genotype       defaultDevice
            | "Serial2D"         -> Serial2D       (defaultSerial, defaultSerial)
            | "TimedSerialDilution" -> TimedSerialDilution (Hours 0.0, defaultSerial)
            | "TimedSerial2D"       -> TimedSerial2D       (Hours 0.0, defaultSerial, defaultSerial)
            | "TimedConditions"     -> TimedConditions     (Hours 0.0, defaultReagent, UM 1.0)
            | _                  -> failwithf "Unknown Antha characterization operation %s" ostr        

        static member Available = ["SerialDilution"; "Conditions"; "Genotype"; "Serial2D"]

    type Op = 
        { guid    : System.Guid
        ; range   : Range
        ; content : CharOp
        }

        static member setRangeFrom (o:Op) (v:string)= {o with range = {o.range with from = v}}
        static member setRangeTo (o:Op) (v:string)  = {o with range = {o.range with until = v}}
        static member setCharOp  (o:Op) (v:CharOp)  = {o with content=v}

        static member CreateFromGenotype g = 
            { guid    = System.Guid.NewGuid()
            ; range   = {from = "A1"; until = "A1"}
            ; content = Genotype g
            }

        static member Create range content = 
            { guid    = System.Guid.NewGuid()
            ; range   = range
            ; content = content
            }

        static member toString (reagents:Map<ReagentId,string>) (cells:Map<CellId,string>) (o:Op) = 
            match o.content with 
            | SerialDilution sd  -> sprintf "SerialDilution \"%s\" \"%s\" \"%s\" %s %f" o.range.from o.range.until reagents.[sd.treatment] (Concentration.toString sd.initial) sd.factor
            | Conditions (r,v)   -> sprintf "Conditions \"%s\" \"%s\" \"%s=%s\"" o.range.from o.range.until reagents.[r] (Concentration.toString v)
            | Genotype g      -> sprintf "Genotype \"%s\" \"%s\" \"%s\"" o.range.from o.range.until cells.[g]
            | Serial2D (sd1, sd2) -> sprintf "2DSerialDilution \"%s\" \"%s\" \"%s\" %s %f \"%s\" %s %f" o.range.from o.range.until reagents.[sd1.treatment] (Concentration.toString sd1.initial) sd1.factor reagents.[sd2.treatment] (Concentration.toString sd2.initial) sd2.factor            
            | TimedSerialDilution (t,sd) -> sprintf "TimedSerialDilution \"%s\" \"%s\" \"%s\" %s %f %s" o.range.from o.range.until reagents.[sd.treatment] (Concentration.toString sd.initial) sd.factor (Time.toString t)
            | TimedSerial2D       (t, sd1, sd2) -> sprintf "Timed2DSerialDilution \"%s\" \"%s\" \"%s\" %s %f \"%s\" %s %f %s" o.range.from o.range.until reagents.[sd1.treatment] (Concentration.toString sd1.initial) sd1.factor reagents.[sd2.treatment] (Concentration.toString sd2.initial) sd2.factor (Time.toString t)
            | TimedConditions     (t, r,v) -> sprintf "TimedConditions \"%s\" \"%s\" \"%s=%s\" %s" o.range.from o.range.until reagents.[r] (Concentration.toString v) (Time.toString t)


        static member genPlate addZeros (ops:seq<Op>) =
            ops
            |> Seq.fold(fun (plate:PlateTagging.Plate) op ->             
                match op.content with
                | SerialDilution p -> plate.SerialDilution addZeros None op.range.from op.range.until p.treatment p.initial p.factor
                | Conditions (reagentId,conc)-> plate.TreatBlock op.range.from op.range.until (reagentId, conc, None)
                | Genotype g       -> plate.SetGenotypeBlock op.range.from op.range.until g
                | Serial2D (p, p') -> plate.SerialDilution2D addZeros None op.range.from op.range.until p.treatment p.initial p.factor p'.treatment p'.initial p'.factor                
                | TimedSerialDilution (t,p) -> plate.SerialDilution addZeros (Some t) op.range.from op.range.until p.treatment p.initial p.factor
                | TimedSerial2D     (t, p, p') -> plate.SerialDilution2D addZeros (Some t) op.range.from op.range.until p.treatment p.initial p.factor p'.treatment p'.initial p'.factor                
                | TimedConditions   (t, reagentId,conc)-> plate.TreatBlock op.range.from op.range.until (reagentId, conc, Some t)

            ) PlateTagging.Plate.empty 
            |> PlateTagging.Plate.FilterZeroSignals

        static member toPlate (ops:seq<Op>) = Op.genPlate true ops
        
        //Old-style serial dilution without zeros included
        static member toPlateNoZeros (ops:seq<Op>) = Op.genPlate false ops
