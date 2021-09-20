module BCKG.Admin.Copy


open BCKG.Admin.Utilities
open System
open Microsoft.Azure.Storage
open Microsoft.Azure.Storage.DataMovement
open Microsoft.Azure.Storage.Blob
open System.Net

let CopyBlobsInContainer (source:CloudStorageAccount) (target:CloudStorageAccount) container = 
    let t0 = System.DateTime.Now
    let progress =
        MailboxProcessor<TransferStatus>.Start(fun inbox ->
        let rec loop _ = async {
            let! transferStatus = inbox.Receive()
            printf("\r%i files transferred (%i files skipped)") transferStatus.NumberOfFilesTransferred transferStatus.NumberOfFilesSkipped

            return! loop 0
        }    
        loop 0 )

    let blobClient = source.CreateCloudBlobClient()

    let blobContainerSource = blobClient.GetContainerReference(container)
    blobContainerSource.CreateIfNotExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore

    let directoryReferenceSource = blobContainerSource.GetDirectoryReference(String.Empty)

    let blobClientDestination = target.CreateCloudBlobClient()

    let blobContainerDestination = blobClientDestination.GetContainerReference(container)
    blobContainerDestination.CreateIfNotExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore

    let directoryReferenceDestination = blobContainerDestination.GetDirectoryReference(String.Empty)

    TransferManager.Configurations.ParallelOperations <- 64
    ServicePointManager.DefaultConnectionLimit <- Environment.ProcessorCount*8
    ServicePointManager.Expect100Continue <- false

    let context = DirectoryTransferContext()

    context.ProgressHandler <- Progress<_>(fun (transferStatus:TransferStatus) ->
        progress.Post(transferStatus)
    )

    context.ShouldOverwriteCallbackAsync <- ShouldOverwriteCallbackAsync(fun _ _ -> async { return true } |> Async.StartImmediateAsTask )

    let copyDirectoryOptions = CopyDirectoryOptions()
    copyDirectoryOptions.Recursive <- true    

    //CopyMethod.ServiceSideAsyncCopy
    TransferManager.CopyDirectoryAsync(
        sourceBlobDir = directoryReferenceSource,
        destBlobDir = directoryReferenceDestination,
        //isServiceCopy = true,
        copyMethod = CopyMethod.ServiceSideAsyncCopy,
        options = copyDirectoryOptions,
        context = context).Wait()
    ((System.DateTime.Now - t0).TotalSeconds)

 

let CopyTableContainer (source:Microsoft.WindowsAzure.Storage.CloudStorageAccount) (target:Microsoft.WindowsAzure.Storage.CloudStorageAccount) container = 
    let t0 = System.DateTime.Now
    let tableSource = source.CreateCloudTableClient().GetTableReference(container)
    let tableTarget = target.CreateCloudTableClient().GetTableReference(container)
    let query = Microsoft.WindowsAzure.Storage.Table.TableQuery()    
        
    tableTarget.CreateIfNotExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously |> ignore

    let mutable error = false;
    
    let entries = 
        (false, null)
        |> Seq.unfold(fun (isNotInitial, token) -> 
            if isNotInitial && token = null then None 
            else    
                let queryResponse = tableSource.ExecuteQuerySegmentedAsync(query, token) |> Async.AwaitTask |> Async.RunSynchronously
                Some(queryResponse.Results, (true, queryResponse.ContinuationToken))
            )
        |> Seq.concat

    entries
    |> Seq.chunkBySize 50
    |> Seq.iter(fun tableEntries ->         
        let batchOperations = Microsoft.WindowsAzure.Storage.Table.TableBatchOperation()
        tableEntries |> Seq.iter batchOperations.InsertOrReplace
        let flag = 
            batchOperations
            |> tableTarget.ExecuteBatchAsync
            |> Async.AwaitTask
            |> Async.RunSynchronously
            |> Seq.exists (fun r -> r.HttpStatusCode <> (int)System.Net.HttpStatusCode.NoContent)
        error <- error && flag
        )

    if error then 
        printfn "ERROR: Table copy error in container %s" container
        -1, -1.0
    else           
        (entries |> Seq.length), ((System.DateTime.Now - t0).TotalSeconds)

    
let CopyAccountTables sourceConnectionString destinationConnectionString enableCleaning = 
    let tableContainers = getTableContainers sourceConnectionString 
    let accountSourceTable = getAccountTable sourceConnectionString
    let accountDestinationTable = getAccountTable destinationConnectionString
    printfn "Table Containers:"
    tableContainers
    |> Seq.iter (fun tableContainer ->             
        if enableCleaning then 
            printf "\t - Cleaning %s..." tableContainer
            CleanTableContainer accountDestinationTable tableContainer
            printfn "done"

        printf "\t - Copying %s..." tableContainer
        let entries, time = CopyTableContainer accountSourceTable accountDestinationTable tableContainer            
        printfn "%i entries copied (%.1f sec)"  entries time
        )
    printfn "------------------------------"

let CopyAccountBlobs sourceConnectionString destinationConnectionString enableCleaning =  
    let accountSource = CloudStorageAccount.Parse(sourceConnectionString)
    let accountDestination = CloudStorageAccount.Parse(destinationConnectionString)
    let blobContainers = accountSource.CreateCloudBlobClient().ListContainers() |> Seq.map(fun container -> container.Name)  
    printfn "Blob Containers:"
    blobContainers
    |> Seq.iter (fun blobContainer ->                                 
        if enableCleaning then 
            printf "\t - Cleaning %s..." blobContainer
            CleanBlobContainer accountDestination blobContainer
            printfn "done"

        printfn "\t - Copying %s..." blobContainer        
        let time = CopyBlobsInContainer accountSource accountDestination blobContainer            
        printfn " (%.1f sec)" time
        )
    printfn "------------------------------"


let GetAccountName (connectionString:string) = 
    let options = 
        connectionString.Split ';'
        |> Array.map (fun x -> x.Trim())
        |> Array.filter ((<>) "")
        |> Array.map(fun opt -> 
            let fields = opt.Split '='
            fields.[0], fields.[1]) 
        |> Map.ofArray
    options.["AccountName"]