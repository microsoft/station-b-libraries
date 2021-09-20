module BCKG.Test.Main

open Argu
open BCKG.Test.APITests
open BCKG.Test.RunTests
open BCKG.Domain
open BCKG.Events
open Thoth.Json.Net
open BCKG.Test.Entities

type TestArguments =
    | [<Mandatory>] EndPoint of string
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | EndPoint _ -> "URL Endpoint for BCKG REST API."

[<EntryPoint>]
let main argv =
    printfn "Starting BCKG REST API Tests"

    let parser = ArgumentParser.Create<TestArguments>(programName = "BckgTest.exe")
    let parserResults = parser.Parse(argv)

    let endpoint = parserResults.GetResult EndPoint

    let rest_api = RestAPI(endpoint)
    runPartTests rest_api
    runReagentTests rest_api
    runCellTests rest_api
    runExperimentTests rest_api
    runSampleTests rest_api
    0 // return an integer exit code
