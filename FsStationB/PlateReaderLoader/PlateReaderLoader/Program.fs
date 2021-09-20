open System
open System.IO

open Argu

open BCKG

open FParsec
open System.Net

//--local --monitorpath "C:\PlateReaderOutputs" --machineidentifier "zebra"
type PlateReaderLoaderArguments =
    | ConnectionString of string
    | Local
    | MonitorPath of string
    | MachineIdentifier of string
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | ConnectionString _ -> "Backend connection string"
            | Local _ -> "Use a local Azure emulator"
            | MonitorPath _ -> """Path to monitor, deployed is usually C:\PlateReaderOutputs\"""
            | MachineIdentifier _ -> """Unique identifier for the machine aardvark,badger,C..."""

//https://github.com/Azure/azure-storage-net-data-movement#increase-net-http-connections-limit
ServicePointManager.DefaultConnectionLimit <- Environment.ProcessorCount * 4

let rawFileToUniformFile machineIdentifier path =
    let allText = File.ReadAllText path
    let parsed = PlateReaderFileParser.parseFrom allText
    let newFilename = sprintf "%s_%s.txt" (parsed.ReadingDateTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss")) machineIdentifier
    { BCKG.Storage.UniformFileUpload.name = newFilename; BCKG.Storage.UniformFileUpload.contents = allText }

let initialUpload connectionString fullPath machineIdentifier =
    let rawFilesUploaded = connectionString |> Storage.ListPlateReaderRawFile |> Set.ofSeq
    let filesOnDisc = Directory.EnumerateFiles fullPath |> Seq.map Path.GetFileName

    let newFileNames = Set.difference (filesOnDisc |> Set.ofSeq) rawFilesUploaded
    let newFilePaths = newFileNames |> Seq.map (fun justFileName -> Path.Combine(fullPath,justFileName))

    let newFileCount = newFilePaths |> Seq.length

    if newFileCount > 0 then
        
        //TODO: Add throttling and progress reporting 
        let asyncs =
            newFilePaths
            |> Seq.map (Storage.SavePlateReaderRawFile connectionString)

        printfn "Uploading existing %i files to \"raw\" from %s " newFileCount fullPath

        asyncs
        |> Async.Parallel
        |> Async.RunSynchronously
        |> ignore

    //The filenames are a bit eractic and are when the scan completes rather than starts. We make them uniform for comparison.
    let localAdditions =
        fullPath
        |> Directory.EnumerateFiles 
        |> Seq.map (fun filename -> filename, FileInfo filename)
        |> Seq.filter (
            fun (filename, fileInfo) ->
                if fileInfo.Length > 0L then true
                else
                    printfn "File not long enough for processing: %s" filename
                    false)
        |> Seq.map (fun (filename, _) -> filename)
        |> Seq.map (rawFileToUniformFile machineIdentifier)
    
    let uniformFilesUploaded = connectionString |> Storage.ListPlateReaderUniformFile |> Set.ofSeq

    let newAdditions =
        localAdditions
        |> Seq.filter (fun addition -> not (Set.contains addition.name uniformFilesUploaded))

    let newAdditionsCount = newAdditions |> Seq.length

    if newAdditionsCount > 0 then

        let asyncs =
            newAdditions
            |> Seq.map (Storage.SavePlateReaderUniformFile connectionString)

        printfn "Uploading existing %i files to \"uniform\"" newAdditionsCount

        asyncs
        |> Async.Parallel
        |> Async.RunSynchronously
        |> ignore

let monitorAndUpload (cancellationToken:System.Threading.CancellationTokenSource) (fileSystemWatcher:FileSystemWatcher) connectionString fullPath machineIdentifier =
    printfn "Monitoring %s (press any key to exit)" fullPath
    
    fileSystemWatcher.Path <- fullPath
    fileSystemWatcher.EnableRaisingEvents <- true
    fileSystemWatcher.Created.Add(fun file -> 
        
        let millisecondsToWait =
            match connectionString with
            | @"UseDevelopmentStorage=true" -> 100
            | _ -> 10000
        printfn "File created %s scheduling upload in %i seconds." file.FullPath (millisecondsToWait / 1000)

        let delayedWork =
            async {
                do! Async.Sleep millisecondsToWait
                printf "Uploading %s to \"raw\" now..." file.FullPath
                do! Storage.SavePlateReaderRawFile connectionString file.FullPath

                printfn "done"

                let newUniform = rawFileToUniformFile machineIdentifier file.FullPath
                printf "Uploading %s to \"uniform\" now..." newUniform.name
                do! Storage.SavePlateReaderUniformFile connectionString newUniform

                printfn "done"
            }
        
        Async.Start (delayedWork, cancellationToken.Token)
    )

[<EntryPoint>]
let main argv =

    //Plate reader should be configured as <date:yymmdd>_<time:hhmm>

    //Maybe during development should just default to emulator, or auto fetch credentials by keyvault?
    let parser = ArgumentParser.Create<PlateReaderLoaderArguments>()
    let parserResults = parser.Parse(argv)
    
    let connectionString =
        if parserResults.Contains Local then @"UseDevelopmentStorage=true"
        else
            parserResults.GetResult ConnectionString
    if connectionString = "%BCKG_CONNECTION_STRING%" || connectionString = "" then
        failwith "Connection string not set correctly. Try setting %BCKG_CONNECTION_STRING%"
    
    //Existing files
    let argumentPath = parserResults.GetResult MonitorPath
    let machineIdentifier = parserResults.GetResult MachineIdentifier

    let fullPath = Path.GetFullPath argumentPath

    printf "Initialising databases..."

    (Storage.initialiseDatabase connectionString).Wait()

    printfn "Done"

    initialUpload connectionString fullPath machineIdentifier
    
    use cancellationToken = new System.Threading.CancellationTokenSource()
    use fileSystemWatcher = new FileSystemWatcher()
    monitorAndUpload cancellationToken fileSystemWatcher connectionString fullPath machineIdentifier
    
    //Wait indefinitely
    Console.ReadKey() |> ignore
    printfn "%A" argv
    0