module PlateReaderFileParser

open FParsec

type UserState = unit
    
    type Parser<'t> = Parser<'t, UserState>

type Configuration =
    | Fluorescence
    | Absorbance

type PlateReaderRun = 
    { Name : string
      ReadingDateTime: System.DateTime
      Configuration : Configuration
      Filters : string array option
      Wavelengths : string array option
      Data : Option<int>[][][] }
    
let ReadDefault() = ""
let pvariable name = pstring (name + ":") >>. spaces >>. restOfLine true
let isDateChar c = isDigit c || c = '/' || c = ':'
let isFilterChar c = isDigit c || c = '-' || c = '/'
let isWavelengthChar c = isDigit c || c = 'n' || c = 'm'
let pdatetime : Parser<_> = pstring ("Date: ") >>. manyChars (satisfy isDateChar) .>> pstring "  Time: " .>>. manyChars (satisfy isDateChar) .>> restOfLine true
let pheadingunderline : Parser<_> = pstring "===" .>> restOfLine true
let skipUntilHeading heading = 
    skipCharsTillString heading true System.Int32.MaxValue .>> restOfLine true .>> pheadingunderline
let pdataitem : Parser<_> = spaces >>. many1Satisfy (fun c -> isDigit c || '-' = c)
let pdataline : Parser<_> = sepBy pdataitem (pchar ',')
let pmultipledataline = manyTill (pdataline .>> skipNewline) newline
let pfilter : Parser<_> = manyChars (satisfy isFilterChar)  
let pwavelength  : Parser<_> = manyChars (satisfy isWavelengthChar)  
let pfilterandgainfluorescene : Parser<_> = spaces >>. satisfy isDigit >>. pchar ':' >>. spaces >>. pfilter .>> spaces .>>. pint64 .>> restOfLine true
let pfilterandgainabsorbance : Parser<_> = spaces >>. satisfy isDigit >>. pchar ':' >>. spaces >>. pwavelength .>> spaces .>>. pint64 .>> spaces .>>. pfloat .>> restOfLine true

let pOneChromaticOfData =
    skipCharsTillString "Chromatic: " true System.Int32.MaxValue >>. restOfLine true >>.
    pvariable "Cycle" >>.
    pvariable "Time [s]" >>.
    restOfLine true >>. //Unicode pain T[°C]
    pmultipledataline

type FileRead = {
    Name : string
    ReadingDateTime : System.DateTime
}

let private monadicStyle : Parser<_,_> =
    //Despite statements http://www.quanttec.com/fparsec/users-guide/where-is-the-monad.html#why-the-monadic-syntax-is-slow benchmark.net tests suggest
    //this syntax is now fine performance-wise. Keep in mind
    parse {
        let! name = pvariable "Testname"
        let! (date,time) = pdatetime
        let! _ = restOfLine true
        let! chromatics = pvariable "No. of Channels / Multichromatics" 
        let! cycles = pvariable "No. of Cycles"
        let! configurationRaw = pvariable "Configuration"
        let! rawFilters =
            parse {
                let! ((wavelength1,gain1),pathlengthcorrection) =
                    attempt(pstring "Used wavelengths, gain values and path length correction factor") >>.
                    restOfLine true >>.
                    pfilterandgainabsorbance
                let! ((wavelength2,gain2),pathlengthcorrection) = pfilterandgainabsorbance
                return [|wavelength1;wavelength2|]
            } <|> parse {
                let! (filter1,gain1) =
                    attempt(pstring "Used filters (excitation/emission) and gain values") >>.
                    restOfLine true >>.
                    pfilterandgainfluorescene
                let! (filter2,gain2) = pfilterandgainfluorescene
                let! (filter3,gain3) = pfilterandgainfluorescene
                let filters = [|filter1;filter2;filter3|] |> Array.filter (fun filterstring -> filterstring <> "-")
                return filters
            }
        let! chromaticData = manyTill (pOneChromaticOfData) eof

        let optionallyParse = 
            function 
            | "-" -> None
            | reading -> Some(System.Int32.Parse reading)
        let reshaped =
            chromaticData |> Seq.map (fun x -> x |> Seq.map (fun outer -> outer |> Seq.map optionallyParse |> Array.ofSeq) |> Array.ofSeq)
            |> Array.ofSeq

        let configuration = match configurationRaw with "Absorbance" -> Absorbance | "Fluorescence" -> Fluorescence | c -> failwithf "Unknwown configuration: %s" c

        return
            { Name = name
              Configuration = configuration
              Filters = match configuration with Absorbance -> None | Fluorescence -> Some rawFilters
              Wavelengths = match configuration with Absorbance -> Some rawFilters | Fluorescence -> None
              ReadingDateTime = System.DateTime.ParseExact(date + " " + time, "dd/MM/yyyy HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture)
              Data = reshaped }
    }

let private unpack result = 
    match result with
    | Success (plateReaderRun, _, _) -> plateReaderRun
    | Failure(errorMsg, _, _) -> failwithf "Failure: %s" errorMsg

let parseFrom string : PlateReaderRun =
    (run monadicStyle string) |> unpack