module Tagging

//TODO: add a typed command log
type Plate = 
    { Wells : Map<string*string, List<string>>
      Genotypes : Map<string*string, List<string>>
      Treatments : Map<string*string, List<string*float>>}

let columnHeaders = [|"1";"2";"3";"4";"5";"6";"7";"8";"9";"10";"11";"12"|]
let rowHeaders = [|"A";"B";"C";"D";"E";"F";"G";"H"|]

let allHeaders = Array.concat [columnHeaders;rowHeaders]

//Fix duplication
let emptyPlate = { Wells = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), List.empty))) |> Map.ofSeq
                   Genotypes = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), List.empty))) |> Map.ofSeq
                   Treatments = rowHeaders |> Seq.collect (fun row -> columnHeaders |> Seq.map (fun col -> ((row, col), List.empty))) |> Map.ofSeq} 

let mutable plateContext = emptyPlate

let ResetPlate() = plateContext <- emptyPlate


let TreatWell row col treatment =
    () 

let TreatColumn col treatment =
    ()

let TreatCol = TreatColumn

let TreatRow col treatment =
    ()

let makeSequence (from:string) (until:string) =

    let fromIndexRow = Array.findIndex ((=) (from.Substring(0,1))) allHeaders
    let untilIndexRow = Array.findIndex ((=) (until.Substring(0,1))) allHeaders
    let rangeRow = Array.sub allHeaders fromIndexRow (untilIndexRow - fromIndexRow)

    let fromIndexCol = Array.findIndex ((=) (from.[1..])) allHeaders
    let untilIndexCol = Array.findIndex ((=) (until.[1..])) allHeaders
    let rangeCol = Array.sub allHeaders fromIndexCol (untilIndexCol - fromIndexCol)

    seq {
        for row in fromIndexRow..untilIndexRow do
            for col in fromIndexCol..untilIndexCol do
                yield allHeaders.[row], allHeaders.[col]
    }

let inline Treat (from:string) (until:string) (masterTreatment:string*_) operation =

    //Need to thread through data, can we use something like initInfinite?
    let treatments =
        seq {
            let mutable afterOperation = snd masterTreatment
            while (true) do
                yield (fst masterTreatment, afterOperation)
                afterOperation <- operation afterOperation
        }

    let newTreatments =
        (plateContext.Treatments, makeSequence from until, treatments)
        |||> Seq.fold2
            (fun acc well treatment -> acc |> Map.add well (treatment :: acc.[well]))

    plateContext <- { Wells = plateContext.Wells; Genotypes = plateContext.Genotypes; Treatments = newTreatments }

(*Treat "A10" "A12" ("X3", 5000.0) (fun a -> a / 3.0)
plateContext
ResetPlate()*)

let inline SerialDilution from until treatment initial factor = Treat from until (treatment, initial) (fun a -> a / factor)

let TreatBlock (from:string) (until:string) treatment =

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

    let newWells = 
        plateContext.Wells
        |> Map.map
            (fun key value ->
                let row,col = key
                if rowInRange row && colInRange col then
                    treatment :: value
                else
                    value)

    plateContext <- { Wells = newWells; Treatments = plateContext.Treatments; Genotypes = plateContext.Genotypes }

//Duplicated from TreatBlock (fix this)
let SetGenotypeBlock (from:string) (until:string) treatment =

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
        plateContext.Genotypes
        |> Map.map
            (fun key value ->
                let row,col = key
                if rowInRange row && colInRange col then
                    treatment :: value
                else
                    value)

    plateContext <- { Wells = plateContext.Wells; Treatments = plateContext.Treatments; Genotypes = newGenotypes }

let Genotype = SetGenotypeBlock

let Note = SetGenotypeBlock

let Conditions = TreatBlock

let PlateToCSVArray() =
    rowHeaders
        |> Array.collect (fun rowH ->

            columnHeaders |> Array.map (fun colH ->

                let genotypeContent =
                    plateContext.Genotypes.[(rowH,colH)]
                    |> (String.concat ";")
                
                let conditionsContent =
                    plateContext.Wells.[(rowH,colH)]

                let treatmentContent =
                    plateContext.Treatments.[(rowH,colH)]
                    |> Seq.map (fun (name,amount) -> sprintf "%s=%f" name amount)
                    

                let totalConditions =
                    (conditionsContent, treatmentContent)
                    ||> Seq.append
                    |> (String.concat ";")

                sprintf "%s,,%s,%s,%s" genotypeContent colH rowH totalConditions
            )
        )

let PlateToCSV() =

    let timeString = PlateToCSVArray()
        
    let content = timeString |> (String.concat System.Environment.NewLine)

    "Genotype, Colony, Well Col, Well Row, Conditions" + System.Environment.NewLine +
    content