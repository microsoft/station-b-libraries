// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
module BCKG.Storage

open System
open System.IO.Compression

open Microsoft.WindowsAzure.Storage
open Microsoft.WindowsAzure.Storage.Table
open Microsoft.WindowsAzure.Storage.Blob

open BCKG.Domain
open BCKG.Events
open BCKG.Entities
open FSharp.Control.Tasks
open System.Text

let private rawPlateReaderContainerID = "rawplatereader"
let private uniformPlateReaderContainerID = "uniformplatereader"
let private reagentsContainerID = "reagents"
let private experimentsContainerID = "experiments"
let private filesContainerID = "files"
let private timeSeriesContainerID = "timeseries"
let private filesMapContainerID = "filesmap"
let private partsContainerID = "parts"
let private eventsContainerID = "events"
let private experimentEventsContainerID = "experimentevents"
let private observationsContainerID = "observations"

let private samplesContainerID = "samples"
let private sampleConditionsContainerID = "sampleconditions"
let private sampleDevicesContainerID = "sampledevices" //Map which devices were measured in each sample
let private sampleReplicatesContainerID = "samplereplicates" //Map replicates to samples
let private signalsContainerID = "signals"

let private tagsContainerID = "tags"

let private eventsV2ContainerID = "eventsV2"

let private cellsContainerID = "cells"
let private cellEntitiesContainerID = "cellentities"

let private interactionsContainerID = "interactions"
let private interactionEntitiesContainerID = "interactionentities"
let private derivedFromContainerID = "derivedfrom"

let private crnServerBuildsID = "crnserver"
let private classicDSDServerBuildsID = "classicdsdserverbuilds"

type FilesIndex = Map<System.Guid, FileRef list>

let private initializeTables storageCredentials =
    //TODO: switch to Cosmos DB api
    let storageAccount = CloudStorageAccount.Parse storageCredentials      
    let tableClient = storageAccount.CreateCloudTableClient()
    
    let eventsTable = tableClient.GetTableReference eventsContainerID //This should be removed
    
    let reagentsTable = tableClient.GetTableReference reagentsContainerID
    let filesMapTable = tableClient.GetTableReference filesMapContainerID
    let partsTable = tableClient.GetTableReference partsContainerID
    
    let experimentsTable = tableClient.GetTableReference experimentsContainerID
    let signalsTable = tableClient.GetTableReference signalsContainerID
    let experimentEventsTable = tableClient.GetTableReference experimentEventsContainerID
    
    let samplesTable = tableClient.GetTableReference samplesContainerID    
    let sampleDevicesTable = tableClient.GetTableReference sampleDevicesContainerID
    let sampleConditionsTable = tableClient.GetTableReference sampleConditionsContainerID
    let sampleReplicatesTable = tableClient.GetTableReference sampleReplicatesContainerID

    let cellsTable = tableClient.GetTableReference cellsContainerID
    let cellEntitiesTable = tableClient.GetTableReference cellEntitiesContainerID

    let interactionsTable = tableClient.GetTableReference interactionsContainerID
    let interactionEntitiesTable = tableClient.GetTableReference interactionEntitiesContainerID
    
    let tagsTable = tableClient.GetTableReference tagsContainerID

    let observationsTable = tableClient.GetTableReference observationsContainerID

    let derivedFromTable = tableClient.GetTableReference derivedFromContainerID
    let eventlogTable = tableClient.GetTableReference eventsV2ContainerID
    task {
        let! results = 
            [                                                
                reagentsTable.CreateIfNotExistsAsync()
                experimentsTable.CreateIfNotExistsAsync()                
                filesMapTable.CreateIfNotExistsAsync()
                partsTable.CreateIfNotExistsAsync()
                eventsTable.CreateIfNotExistsAsync()
                experimentEventsTable.CreateIfNotExistsAsync()
                samplesTable.CreateIfNotExistsAsync()                
                signalsTable.CreateIfNotExistsAsync()
                sampleDevicesTable.CreateIfNotExistsAsync()
                sampleConditionsTable.CreateIfNotExistsAsync()
                sampleReplicatesTable.CreateIfNotExistsAsync()
                eventlogTable.CreateIfNotExistsAsync()
                cellsTable.CreateIfNotExistsAsync()
                cellEntitiesTable.CreateIfNotExistsAsync()
                interactionsTable.CreateIfNotExistsAsync()
                interactionEntitiesTable.CreateIfNotExistsAsync()
                tagsTable.CreateIfNotExistsAsync()
                derivedFromTable.CreateIfNotExistsAsync()
                observationsTable.CreateIfNotExistsAsync()
            ]
            |> System.Threading.Tasks.Task.WhenAll
                
        return Seq.forall id results
    }

let private initializeBlobs storageCredentials =
    //TODO: switch to Cosmos DB api
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()    
    let rawPlateReaderContainerReference = blobClient.GetContainerReference rawPlateReaderContainerID
    let uniformPlateReaderContainerReference = blobClient.GetContainerReference uniformPlateReaderContainerID    
    let experimentFilesContainerReference = blobClient.GetContainerReference filesContainerID
    let timeSeriesContainerReference = blobClient.GetContainerReference timeSeriesContainerID
    
    task {
        let! results = 
            [                
                rawPlateReaderContainerReference.CreateIfNotExistsAsync()
                uniformPlateReaderContainerReference.CreateIfNotExistsAsync()         
                experimentFilesContainerReference.CreateIfNotExistsAsync()            
                timeSeriesContainerReference.CreateIfNotExistsAsync()    
            ]
            |> System.Threading.Tasks.Task.WhenAll
                
        return Seq.forall id results
    }

let public initialiseDatabase storageCredentials =    
    task {
        let! results = 
            [                
                initializeTables storageCredentials
                initializeBlobs storageCredentials                
            ]
            |> System.Threading.Tasks.Task.WhenAll
                
        return Seq.forall id results
    }

let internal clearTables storageCredentials = 
    let rec RepeatedInit (counter:int) =                
        task {
            if counter < 0  then
                failwith "Max number of iterations reached" 
            try                                
                return! initializeTables storageCredentials
            with 
            | e -> 
            //:? StorageException as e ->             
              //  if (e.RequestInformation.HttpStatusCode = 409) && ((e.RequestInformation.ExtendedErrorInformation.ErrorCode.Equals(Microsoft.WindowsAzure.Storage.Table.Protocol.TableErrorCodeStrings.TableBeingDeleted))) then                                
                    printfn "Waiting for an Azure table delete...(%i left)\n%s" counter (e.Message)
                    do! System.Threading.Tasks.Task.Delay 1000          
                    return! RepeatedInit(counter - 1)                
            }

     //TODO: switch to Cosmos DB api
    let storageAccount = CloudStorageAccount.Parse storageCredentials                
    let tableClient = storageAccount.CreateCloudTableClient()
    let reagentsTable = tableClient.GetTableReference reagentsContainerID
    let experimentsTable = tableClient.GetTableReference experimentsContainerID
    let filesMapTable = tableClient.GetTableReference filesMapContainerID
    let partsTable = tableClient.GetTableReference partsContainerID
    let eventsTable = tableClient.GetTableReference eventsContainerID
    let experimentEventsTable = tableClient.GetTableReference experimentEventsContainerID
    let samplesTable = tableClient.GetTableReference samplesContainerID    
    let signalsTable = tableClient.GetTableReference signalsContainerID
    let sampleDevicesTable = tableClient.GetTableReference sampleDevicesContainerID
    let sampleConditionsTable = tableClient.GetTableReference sampleConditionsContainerID
    let sampleReplicatesTable = tableClient.GetTableReference sampleReplicatesContainerID
    let eventlogTable = tableClient.GetTableReference eventsV2ContainerID

    let cellsTable = tableClient.GetTableReference cellsContainerID
    let cellEntitiesTable = tableClient.GetTableReference cellEntitiesContainerID

    let interactionsTable = tableClient.GetTableReference interactionsContainerID
    let interactionEntitiesTable = tableClient.GetTableReference interactionEntitiesContainerID
    let tagsTable = tableClient.GetTableReference tagsContainerID
    let derivedFromTable = tableClient.GetTableReference derivedFromContainerID

    let observationsTable = tableClient.GetTableReference observationsContainerID

    task {
        let! results = 
            [                                                
                reagentsTable.DeleteIfExistsAsync()
                experimentsTable.DeleteIfExistsAsync()
                filesMapTable.DeleteIfExistsAsync()
                partsTable.DeleteIfExistsAsync()
                eventsTable.DeleteIfExistsAsync()
                experimentEventsTable.DeleteIfExistsAsync()
                samplesTable.DeleteIfExistsAsync()
                signalsTable.DeleteIfExistsAsync()
                sampleDevicesTable.DeleteIfExistsAsync()
                sampleConditionsTable.DeleteIfExistsAsync()
                sampleReplicatesTable.DeleteIfExistsAsync()
                eventlogTable.DeleteIfExistsAsync()
                cellsTable.DeleteIfExistsAsync()
                cellEntitiesTable.DeleteIfExistsAsync()
                interactionsTable.DeleteIfExistsAsync()
                interactionEntitiesTable.DeleteIfExistsAsync()
                tagsTable.DeleteIfExistsAsync()
                derivedFromTable.DeleteIfExistsAsync()
                observationsTable.DeleteIfExistsAsync()
            ]
            |> System.Threading.Tasks.Task.WhenAll
                        
        do! System.Threading.Tasks.Task.Delay 5000
        let! initResult = RepeatedInit 60
        return initResult && (Seq.forall id results)                        
    }
    
let internal clearBCKGBlobs storageCredentials = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()    
    let filesContainerReference = blobClient.GetContainerReference filesContainerID
    let timeSeriesContainerReference = blobClient.GetContainerReference timeSeriesContainerID
    task {
        let! results = 
            [                
                filesContainerReference.DeleteIfExistsAsync()
                timeSeriesContainerReference.DeleteIfExistsAsync()
            ]
            |> System.Threading.Tasks.Task.WhenAll
                
        return Seq.forall id results
    }

type UniformFile =
    {
        dateTime : DateTime
        machineIdentifier : string
        fullFileName : string
    }

type UniformFileUpload = {
    name : string
    contents : String
}

let private ListBlobs storageCredentials containerID =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let container = blobClient.GetContainerReference containerID

    //Switch to asyncSeq?
    async {
        //Adapted from https://docs.microsoft.com/en-us/dotnet/fsharp/using-fsharp-on-azure/blob-storage
        let blobs = ResizeArray()

        //TODO: rewrite this to tasks, iterative not recursive
        let rec loop continuationToken (depth:int) =
            async {
                let! ct = Async.CancellationToken

                let! resultSegment = 
                    container.ListBlobsSegmentedAsync(
                        "",
                        true,
                        Blob.BlobListingDetails.None,
                        Nullable(), //Implicitly 5000 at time of writing
                        continuationToken,
                        null,
                        null,
                        ct) 
                    |> Async.AwaitTask
                    
                blobs.AddRange resultSegment.Results

                // Get the continuation token.
                let continuationToken = resultSegment.ContinuationToken
                if (continuationToken <> null) then
                    do! loop continuationToken (depth+1)
            }
    
        do! loop null 1

        return blobs
    }

let private ListTable storageCredentials containerID (tableQ:TableQuery<_>) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference containerID

    //Switch to asyncSeq?
    async {
        let rows = ResizeArray()                
        let mutable continuationToken = TableContinuationToken()
        while (continuationToken <> null) do    
            let! query = table.ExecuteQuerySegmentedAsync(tableQ,continuationToken) |> Async.AwaitTask
            rows.AddRange(query.Results)
            continuationToken <- query.ContinuationToken
        return rows
    }


let internal dowloadBlobs storageCredentials containerID =         
    async {
        use memstream = new System.IO.MemoryStream()
        use zip = new ZipArchive(memstream, ZipArchiveMode.Create, true)
        let! blobs = ListBlobs storageCredentials containerID        

        blobs.ToArray()
        |> Array.iteri(fun i blob -> 
            let cloudBlockBlob = blob :?> CloudBlockBlob                        
            printfn "Retreiving blob (%i/%i) %s..." i blobs.Count cloudBlockBlob.Name
            //let cloudBlockBlob = container.GetBlockBlobReference (blob :> CloudBlockBlob).Name
            
            let entry = zip.CreateEntry cloudBlockBlob.Name
            use entryStream = entry.Open()        
            cloudBlockBlob.DownloadToStreamAsync entryStream |> Async.AwaitTask |> Async.RunSynchronously            
            )        
        zip.Dispose() //Early dispose, force checksums, needed?        
        return memstream.ToArray()
        }

let internal getFileRef storageCredentials (FileId fileGuid) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference filesMapContainerID
    
    async {
        let retrieve = TableOperation.Retrieve<FileRefStore>("", fileGuid.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let store = res.Result :?> FileRefStore            
            return FileRefStore.toFileRef(store) |> Some
        else
            return None
                
    }


let internal getExperimentEvent storageCredentials (ExperimentOperationId eventGuid) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentEventsContainerID
    
    async {
        let retrieve = TableOperation.Retrieve<ExperimentOperationStore>("", eventGuid.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let store = res.Result :?> ExperimentOperationStore            
            return ExperimentOperationStore.toExperimentEvent(store) |> Some
        else
            return None
                
    }

    

//NOTE: Slow! Don't use for repeated operations (e.g. loading a list of X with associated files)
let internal getFileRefs storageCredentials (sourceGuid:System.Guid) =     
    async { 
        let query = (new TableQuery<FileRefStore>()).Where(TableQuery.GenerateFilterCondition("source", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! fileRefStores = ListTable storageCredentials filesMapContainerID query                                    
        let fileRefs = 
            fileRefStores.ToArray()
            |> Array.map FileRefStore.toFileRef
        return fileRefs
    }

let internal getReagent storageCredentials (reagentId:ReagentId) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference reagentsContainerID
    
    async {
        let retrieve = TableOperation.Retrieve<ReagentStore>("", reagentId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let store = res.Result :?> ReagentStore            
            return ReagentStore.toReagent(store) |> Some
        else
            return None
                
    }

let internal storeFileRef connectionString (source:System.Guid)  (file:FileRef)= 
    let storageAccount = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse connectionString
    let tableClient = storageAccount.CreateCloudTableClient()        
    let filesTable = tableClient.GetTableReference filesMapContainerID    
    async {
        let fileRef = FileRefStore(source, file)
        let fileUpdateOperation = TableOperation.InsertOrReplace(fileRef)
        let! res = filesTable.ExecuteAsync(fileUpdateOperation) |> Async.AwaitTask                   
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal deleteFileRef connectionString (source:System.Guid) (file:FileRef)= 
    let fileRef = FileRefStore(source, file)
    fileRef.ETag <- "*"
    let storageAccount = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse connectionString
    let tableClient = storageAccount.CreateCloudTableClient()        
    let filesTable = tableClient.GetTableReference filesMapContainerID    
    async {        
        let fileUpdateOperation = TableOperation.Delete(fileRef)
        let! res = filesTable.ExecuteAsync(fileUpdateOperation) |> Async.AwaitTask                 
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal storeTag connectionString (source:System.Guid) (tag:Tag) = 
    let storageAccount = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse connectionString
    let tableClient = storageAccount.CreateCloudTableClient()
    let tagTable = tableClient.GetTableReference tagsContainerID
    async {
        let tagstore = TagStore(source,tag)
        let tagUpdateOperation = TableOperation.InsertOrReplace(tagstore)
        let! res = tagTable.ExecuteAsync(tagUpdateOperation) |> Async.AwaitTask   
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }
let internal getTags storageCredentials (sourceGuid:System.Guid) =     
    async { 
        let query = (new TableQuery<TagStore>()).Where(TableQuery.GenerateFilterCondition("source", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! tagStores = ListTable storageCredentials tagsContainerID query                      
        let tags = 
            tagStores.ToArray()
            |> Array.map (fun store -> Tag store.tag)
        return tags
    }

let private getTag storageCredentials (sourceGuid:System.Guid) (tag:Tag) =     
    async { 
        let query = (new TableQuery<TagStore>()).Where(TableQuery.GenerateFilterCondition("source", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! tagStores = ListTable storageCredentials tagsContainerID query                                    
        let tag = 
            tagStores.ToArray()
            |> Array.tryFind (fun store -> store.tag = tag.ToString())
        return tag
    }

let internal deleteTag connectionString (source:System.Guid) (tag:Tag) = 
    let tagstoreopt = getTag connectionString source tag |> Async.RunSynchronously
    let tagstore = match tagstoreopt with | Some(tstore) -> tstore | None -> failwith "Tag doesn't exist."
    tagstore.ETag <- "*"
    let storageAccount = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse connectionString
    let tableClient = storageAccount.CreateCloudTableClient()        
    let tagTable = tableClient.GetTableReference tagsContainerID    
    async {        
        let tagUpdateOperation = TableOperation.Delete(tagstore)
        let! res = tagTable.ExecuteAsync(tagUpdateOperation) |> Async.AwaitTask                 
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    } 

//let private storeFileRefs (source:System.Guid) (filesTable:CloudTable) (files:seq<FileRef>)= 
//    async {
//        files
//        |> Seq.iter(fun f -> 
//            let fileRef = FileRefStore(source, f)
//            let fileUpdateOperation = TableOperation.InsertOrReplace(fileRef)
//            filesTable.ExecuteAsync(fileUpdateOperation) |> Async.AwaitTask |> ignore
//            )        
//    }


let internal saveReagent storageCredentials (reagent:Reagent) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference reagentsContainerID            
    let store = ReagentStore(reagent)                
    async {      
        let reagentUpdateOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(reagentUpdateOperation) |> Async.AwaitTask
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

////To-Do: Should this be available for events? 
//let private saveReagentStore  storageCredentials (store:ReagentStore) =
//    let storageAccount = CloudStorageAccount.Parse storageCredentials    
//    let tableClient = storageAccount.CreateCloudTableClient()
//    let table = tableClient.GetTableReference reagentsContainerID        
//    async {
//        let updateOperation = TableOperation.InsertOrReplace(store)
//        let! res = table.ExecuteAsync(updateOperation) |> Async.AwaitTask

//        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
//    }

let internal savePart storageCredentials (part:Part) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference partsContainerID   
    let store = PartStore(part)        
    async {
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask                        
        //https://docs.microsoft.com/en-us/rest/api/storageservices/Insert-Or-Replace-Entity?redirectedfrom=MSDN
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

//TODO: avoid code reuse with savePart, etc?
let internal saveSample storageCredentials (sample:Sample) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()        
    let table = tableClient.GetTableReference samplesContainerID   
    let store = SampleStore(sample)    
    async {      
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask            
        
        //https://docs.microsoft.com/en-us/rest/api/storageservices/Insert-Or-Replace-Entity?redirectedfrom=MSDN
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }


let internal saveCell storageCredentials (cell:Cell) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let cellsTable = tableClient.GetTableReference cellsContainerID
    let cellStore = CellStore(cell)
    async{
        let tableOperation = TableOperation.InsertOrReplace(cellStore)
        let! res = cellsTable.ExecuteAsync(tableOperation) |> Async.AwaitTask        
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal saveCellEntity storageCredentials (cellEntity:CellEntity) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference cellEntitiesContainerID
    let store = CellEntityStore(cellEntity)
    async{
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask        
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let private getCellEntity storageCredentials (cellEntity:CellEntity) = 
    async{
        let query = (new TableQuery<CellEntityStore>()).Where(TableQuery.GenerateFilterCondition("cellId", QueryComparisons.Equal, cellEntity.cellId.ToString()));            
        let! cellEntityStores = ListTable storageCredentials cellEntitiesContainerID query                                    
        let cellEntity = 
            cellEntityStores.ToArray()
            |> Array.tryFind(fun store -> 
                let ce = CellEntityStore.toCellEntity store
                (ce.compartment = cellEntity.compartment) && (ce.entity = cellEntity.entity) && (ce.cellId = cellEntity.cellId))                
        return cellEntity
    
    }

let internal deleteCellEntity storageCredentials (cellEntity:CellEntity) = 
    let cellEntityStoreopt = getCellEntity storageCredentials cellEntity |> Async.RunSynchronously
    let cellEntityStore = match cellEntityStoreopt with | Some(cestore) -> cestore | None -> failwith "Cell Entity doesn't exist."
    cellEntityStore.ETag <- "*"
    let storageAccount = Microsoft.WindowsAzure.Storage.CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()        
    let ceTable = tableClient.GetTableReference cellEntitiesContainerID    
    async {        
        let ceUpdateOperation = TableOperation.Delete(cellEntityStore)
        let! res = ceTable.ExecuteAsync(ceUpdateOperation) |> Async.AwaitTask                 
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }


let private storeInteractionEntities (intEntitiesTable:CloudTable) (entities:seq<InteractionEntityStore>)= 
    async {
        entities
        |> Seq.iter(fun iestore -> 
            let entityUpdateOperation = TableOperation.InsertOrReplace(iestore)
            intEntitiesTable.ExecuteAsync(entityUpdateOperation) |> Async.AwaitTask |> ignore
            )        
    }

let internal saveInteraction storageCredentials (interaction:Interaction) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let interactionsTable = tableClient.GetTableReference interactionsContainerID
    let intEntitiesTable = tableClient.GetTableReference interactionEntitiesContainerID
    let interactionStore = InteractionStore(interaction.getProperties,interaction.getType)
    let intEntities = 
        match interaction with 
        | CodesFor(cf) -> 
            let cds = match cf.cds with CDSId guid -> guid
            let prot = match cf.protein with ProteinId guid -> guid
             
            [
                InteractionEntityStore(interaction.id,InteractionNodeType.Template,cds,PartId.GetType (cf.cds |> CDSPartId),0)
                InteractionEntityStore(interaction.id,InteractionNodeType.Product,prot,ReagentId.GetType (cf.protein |> ProteinReagentId), 0);
            ]
        | GeneticActivation(ga) -> 
            let activated = 
                let guid = match ga.activated with PromoterId guid -> guid
                InteractionEntityStore(interaction.id,InteractionNodeType.Activated,guid, PartId.GetType (ga.activated |> PromoterPartId), 0)
            let activator = 
                ga.activator |> List.map (fun r -> 
                    InteractionEntityStore(interaction.id,InteractionNodeType.Activator,r.guid,ReagentId.GetType r,0))
            activated::activator
        | GeneticInhibition(gi) -> 
            let inhibited = 
                let guid = match gi.inhibited with PromoterId guid -> guid
                InteractionEntityStore(interaction.id,InteractionNodeType.Inhibited,guid, PartId.GetType (gi.inhibited |> PromoterPartId),0)
            let inhibitor = 
                gi.inhibitor |> List.map (fun r -> 
                    InteractionEntityStore(interaction.id,InteractionNodeType.Inhibitor,r.guid,ReagentId.GetType r,0))
            inhibited::inhibitor
        | Reaction(rxn) -> 
            let reactants = 
                rxn.reactants
                |> List.mapi (fun i complex -> 
                    complex |> List.map (fun r -> InteractionEntityStore(interaction.id,InteractionNodeType.Reactant,r.guid,ReagentId.GetType r,i)))
                |> List.fold (fun acc x -> acc@x) []

            let enzyme = 
                rxn.enzyme
                |> List.map (fun r -> 
                    InteractionEntityStore(interaction.id,InteractionNodeType.Enzyme,r.guid, ReagentId.GetType r,0))

            let products = 
                rxn.products
                |> List.mapi (fun i complex -> 
                    complex |> List.map (fun r -> 
                        InteractionEntityStore(interaction.id,InteractionNodeType.Product,r.guid,ReagentId.GetType r,i)))
                |> List.fold (fun acc x -> acc@x) []
            reactants@enzyme@products
        | GenericActivation(gact) -> 
            let regulators = 
                gact.regulator 
                |> List.map (fun r -> 
                    InteractionEntityStore(interaction.id,InteractionNodeType.Regulator,r.guid,ReagentId.GetType r,0))
            let regulated = 
                let r = gact.regulated 
                InteractionEntityStore(interaction.id,InteractionNodeType.Regulated,r.guid,ReagentId.GetType r,0)
            regulated::regulators
        | GenericInhibition(ginh) -> 
            let regulators = 
                ginh.regulator 
                |> List.map (fun r -> 
                    InteractionEntityStore(interaction.id,InteractionNodeType.Regulator,r.guid,ReagentId.GetType r,0))
            let regulated = 
                let r = ginh.regulated 
                InteractionEntityStore(interaction.id,InteractionNodeType.Regulated,r.guid,ReagentId.GetType r,0)
            regulated::regulators
    async{
        let tableOperation = TableOperation.InsertOrReplace(interactionStore)
        let! res = interactionsTable.ExecuteAsync(tableOperation) |> Async.AwaitTask        
    
        let! _ = storeInteractionEntities intEntitiesTable intEntities
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
        
    }

let internal saveInteractionConstituents storageCredentials (iprops:InteractionProperties) (iType:string) (entities:(InteractionId*InteractionNodeType*System.Guid*string*int)list) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let interactionsTable = tableClient.GetTableReference interactionsContainerID
    let intEntitiesTable = tableClient.GetTableReference interactionEntitiesContainerID
    let interactionStore = InteractionStore(iprops,iType)
    let entityStores = entities |> List.map (fun (a,b,c,d,e) -> InteractionEntityStore(a,b,c,d,e)) |> List.toSeq
    async{
        let tableOperation = TableOperation.InsertOrReplace(interactionStore)
        let! res = interactionsTable.ExecuteAsync(tableOperation) |> Async.AwaitTask        
    
        let! _ = storeInteractionEntities intEntitiesTable entityStores
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
        
    }
   
let internal saveDerivedFrom storageCredentials (derivedFrom:DerivedFrom) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference derivedFromContainerID
    let dfstore = DerivedFromStore(derivedFrom)
    async{
        let tableOperation = TableOperation.InsertOrReplace(dfstore)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal getDerivedFrom storageCredentials (derivedFrom:DerivedFrom) = 
    //let storageAccount = CloudStorageAccount.Parse storageCredentials        
    //let tableClient = storageAccount.CreateCloudTableClient()
    let typeFilter = TableQuery.GenerateFilterCondition("Type",QueryComparisons.Equal,DerivedFrom.GetType derivedFrom)
    let sourceFilter = TableQuery.GenerateFilterCondition("source",QueryComparisons.Equal,(DerivedFrom.GetSourceGuid derivedFrom).ToString())
    let targetFilter = TableQuery.GenerateFilterCondition("target",QueryComparisons.Equal,(DerivedFrom.GetTargetGuid derivedFrom).ToString())
    let combineEntityFilter = TableQuery.CombineFilters(sourceFilter,TableOperators.And,targetFilter)
    let combinedFilter = TableQuery.CombineFilters(combineEntityFilter,TableOperators.And,typeFilter)
    async { 
        let query = (new TableQuery<DerivedFromStore>()).Where(combinedFilter);            
        let! dfstores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivedFroms = dfstores.ToArray()
        let derivedFrom =     
            match derivedFroms with 
            | [||] -> None
            | [|dfstore|] -> dfstore |> DerivedFromStore.toDerivedFrom |> Some
            | _ -> 
                printfn "[WARNING] Multiple Derived Froms observed. Returning the first one. Please fix this."
                derivedFroms.[1] |> DerivedFromStore.toDerivedFrom |> Some
        return derivedFrom
    }

let internal deleteDerivedFrom storageCredentials (derivedFrom:DerivedFrom) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference derivedFromContainerID    
    let typeFilter = TableQuery.GenerateFilterCondition("Type",QueryComparisons.Equal,DerivedFrom.GetType derivedFrom)
    let sourceFilter = TableQuery.GenerateFilterCondition("source",QueryComparisons.Equal,(DerivedFrom.GetSourceGuid derivedFrom).ToString())
    let targetFilter = TableQuery.GenerateFilterCondition("target",QueryComparisons.Equal,(DerivedFrom.GetTargetGuid derivedFrom).ToString())
    let combineEntityFilter = TableQuery.CombineFilters(sourceFilter,TableOperators.And,targetFilter)
    let combinedFilter = TableQuery.CombineFilters(combineEntityFilter,TableOperators.And,typeFilter)
    async {
        let query = (new TableQuery<DerivedFromStore>()).Where(combinedFilter);            
        let! dfstores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivedFroms = dfstores.ToArray()
        let derivedFromStore =     
            match derivedFroms with 
            | [||] -> None
            | [|dfstore|] -> dfstore |> Some
            | _ -> 
                printfn "[WARNING] Multiple Derived Froms observed. Returning the first one. Please fix this."
                derivedFroms.[1] |> Some
        match derivedFromStore with 
        | Some(existing) -> 
            existing.ETag <- "*"
            let tableOperation = TableOperation.Delete(existing)
            let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
            return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
        | None -> return true
    }


let internal saveSampleDevice storageCredentials (sampleId:SampleId) (cellId:CellId) (cellDensity:float option) (cellPreSeeding:float option) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleDevicesContainerID    
    let store = SampleCellStore(sampleId,cellId, cellDensity, cellPreSeeding)
    async {
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal deletesampleDeviceStore storageCredentials (store:SampleCellStore) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleDevicesContainerID    
    store.ETag <- "*"

    async {
        
        let tableOperation = TableOperation.Delete(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal getSampleCellStore storageCredentials (sampleId:SampleId) (cellId:CellId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleDevicesContainerID    
    
    async {
        let retrieve = TableOperation.Retrieve<SampleCellStore>("", sampleId.ToString() + cellId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let store = res.Result :?> SampleCellStore            
            return store |> Some
        else
            return None                
    }

let internal getSampleDevice storageCredentials (sampleId:SampleId) (cellId:CellId) : Async<SampleDevice option>= 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleDevicesContainerID    
    
    async {
        let retrieve = TableOperation.Retrieve<SampleDeviceStore>("", sampleId.ToString() + cellId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let store = res.Result :?> SampleDevice            
            return store |> Some
        else
            return None                
    }

let internal getSampleDevices storageCredentials (sampleId:SampleId): Async<SampleDevice[]>= 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()

    async {
        let query = (new TableQuery<SampleDeviceStore>()).Where(TableQuery.GenerateFilterCondition("sampleId", QueryComparisons.Equal, sampleId.ToString()))
        let! results = ListTable storageCredentials sampleDevicesContainerID query
        return results.ToArray() |> Array.map (fun x ->  SampleDeviceStore.toSampleDevice x )

    }

let internal getSampleCondition storageCredentials (sampleId:SampleId) (reagentId:ReagentId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleConditionsContainerID    
    
    async {
        let retrieve = TableOperation.Retrieve<ConditionStore>("", sampleId.ToString() + reagentId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let condition = res.Result :?> ConditionStore |> ConditionStore.toCondition    
            return Some (condition)
        else
            return None                
    }


let internal saveSampleCondition storageCredentials (sampleId:SampleId) (reagentId:ReagentId) (conc:Concentration) (time: Time option) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleConditionsContainerID 
    let condition = Condition.Create(reagentId, sampleId, conc, time)
    let store = ConditionStore(condition)
    async {
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal deletesampleConditionStore storageCredentials  (sampleId:SampleId) (reagentId:ReagentId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleConditionsContainerID 
    let key = sampleId.ToString() + reagentId.ToString()
    
    async {
        let! res = TableOperation.Retrieve<ConditionStore>("",key) |> table.ExecuteAsync |> Async.AwaitTask
        let store = res.Result:?> ConditionStore  
        store.ETag <- "*"
        let tableOperation = TableOperation.Delete(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal saveReplicate storageCredentials (sampleId:SampleId) (replicateId:ReplicateId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleReplicatesContainerID
    let store = SampleReplicateStore(sampleId, replicateId)
    async {
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal unlinkReplicate storageCredentials (replicateId:ReplicateId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference sampleReplicatesContainerID
    async {
        let! res = TableOperation.Retrieve<SampleReplicateStore>("",replicateId.ToString()) |> table.ExecuteAsync |> Async.AwaitTask
        let store = res.Result:?> SampleReplicateStore  
        store.ETag <- "*"
        let tableOperation = TableOperation.Delete(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

//TODO: Batch operations have a limit of 100 entries. This will fail for many samples/conditions/etc
let private saveSamples storageCredentials (samples:seq<Sample>) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let samplesTable = tableClient.GetTableReference samplesContainerID   
    let filesTable = tableClient.GetTableReference filesMapContainerID    
    let conditionsTable = tableClient.GetTableReference sampleConditionsContainerID    
    let devicesTable = tableClient.GetTableReference sampleDevicesContainerID    
    
    async {              
        let saveSampleOperations = new TableBatchOperation()
        let saveSampleDataOperations = new TableBatchOperation()        
        let saveSampleDevicesOperations = new TableBatchOperation()
        let saveSampleConditionsOperations = new TableBatchOperation()
        let flush_limit = 50
        let mutable error = false;
        let FlushBatchOperations (table:CloudTable) (ops:TableBatchOperation) = 
            if ops.Count > 0 then 
                let flag = 
                    ops
                    |> table.ExecuteBatchAsync
                    |> Async.AwaitTask
                    |> Async.RunSynchronously
                    |> Seq.exists (fun r -> r.HttpStatusCode <> (int)System.Net.HttpStatusCode.NoContent)
                error <- error && flag
                ops.Clear()
       
        samples 
        |> Seq.iter (fun sample ->             
            let sampleGuid = match sample.id with SampleId guid -> guid
            
            //store the actual sample
            sample |> SampleStore |> saveSampleOperations.InsertOrReplace

            if saveSampleOperations.Count > flush_limit then
                FlushBatchOperations samplesTable saveSampleOperations                                     
            
           )
        
        //execute the remaining batch operations        
        if saveSampleConditionsOperations.Count > 0 then 
                FlushBatchOperations conditionsTable saveSampleConditionsOperations                    
            
        if saveSampleDataOperations.Count > 0 then            
            FlushBatchOperations filesTable saveSampleDataOperations                    

        if saveSampleDevicesOperations.Count > 0 then
            FlushBatchOperations devicesTable saveSampleDevicesOperations                    

        if saveSampleOperations.Count > 0 then
            FlushBatchOperations samplesTable saveSampleOperations                    
                
        //https://docs.microsoft.com/en-us/rest/api/storageservices/Insert-Or-Replace-Entity?redirectedfrom=MSDN
        return (not error) //TODO: Check return logic?
    }

let internal saveObservation storageCredentials (observation:Observation) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials    
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference observationsContainerID 
    let store = ObservationStore(observation)
    async {
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let private getConditionsForSample storageCredentials (sampleId:SampleId) =         
    async { 
        let query = (new TableQuery<ConditionStore>()).Where(TableQuery.GenerateFilterCondition("sampleId", QueryComparisons.Equal, sampleId.ToString()));            
        let! conditionStores = ListTable storageCredentials sampleConditionsContainerID query                                    
        let conditions = 
            conditionStores.ToArray()
            |> Array.map (fun store -> 
                let reagentId = ReagentId.FromType (store.reagentId |> System.Guid) (store.reagentType)
                reagentId, store.value
                )  
            |> List.ofArray
        return conditions
    }

    
let internal getSampleCells storageCredentials (sampleId:SampleId) =         
    async { 
        let query = (new TableQuery<SampleCellStore>()).Where(TableQuery.GenerateFilterCondition("sampleId", QueryComparisons.Equal, sampleId.ToString()));            
        let! deviceStores = ListTable storageCredentials sampleDevicesContainerID query                                    
        let devices = 
            deviceStores.ToArray()
            |> Array.map (fun store -> 
                let cellId = store.cellId |> System.Guid |> CellId
                let cellDensity = if store.cellDensity < 0.0 then None else Some store.cellDensity
                let cellPreSeeding = if store.cellDensity < 0.0 then None else Some store.cellPreSeeding
                cellId, (cellDensity, cellPreSeeding)
                )                          
        return devices
    }

let internal getSampleConditions storageCredentials (sampleId:SampleId) = 
    async {
        let query = (new TableQuery<ConditionStore>()).Where(TableQuery.GenerateFilterCondition("sampleId", QueryComparisons.Equal, sampleId.ToString()));            
        let! conditionStores = ListTable storageCredentials sampleConditionsContainerID query
        let conditions = 
            conditionStores.ToArray()
            |> Array.map (fun store -> store |> ConditionStore.toCondition)
        return conditions     
    }

let internal getSamplesForExperiment storageCredentials (ExperimentId sourceGuid) =         
    async {         
        let query = (new TableQuery<SampleStore>()).Where(TableQuery.GenerateFilterCondition("experimentId", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! sampleStores = ListTable storageCredentials samplesContainerID query                          
        let samples = 
            sampleStores.ToArray()
            |> Array.map (fun store -> SampleStore.toSample(store))                   
        return samples
    }

let internal getExperimentOperations storageCredentials (ExperimentId sourceGuid) = 
    async  {
        let query = (new TableQuery<ExperimentOperationStore>()).Where(TableQuery.GenerateFilterCondition("source",QueryComparisons.Equal,sourceGuid.ToString()));
        let! experimentOperationStores = ListTable storageCredentials experimentEventsContainerID query
        let experimentOperations = 
            experimentOperationStores.ToArray()
            |> Array.map (fun estore -> ExperimentOperationStore.toExperimentEvent(estore))            
        return experimentOperations
    }

let internal getExperimentSignals storageCredentials (ExperimentId sourceGuid) =     
    async { 
        let query = (new TableQuery<SignalStore>()).Where(TableQuery.GenerateFilterCondition("experimentId", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! signalStores = ListTable storageCredentials signalsContainerID query                                    
        let signals = 
            signalStores.ToArray()
            |> Array.map SignalStore.toSignal                           
        return signals
    }

let internal getSignals storageCredentials (SignalId sourceGuid) = 
    async {
        let query = (new TableQuery<SignalStore>()).Where(TableQuery.GenerateFilterCondition("id", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! signalStores = ListTable storageCredentials signalsContainerID query                                    
        let signal_map = 
            signalStores.ToArray()
            |> Array.map (fun sig_store -> ((sig_store.experimentId |> System.Guid |> ExperimentId), SignalStore.toSignal sig_store))
        return signal_map
    }

let internal getEntitiesOfCell storageCredentials (CellId sourceGuid) = 
    async{
        let query = (new TableQuery<CellEntityStore>()).Where(TableQuery.GenerateFilterCondition("cellId", QueryComparisons.Equal, sourceGuid.ToString()));            
        let! cellEntityStores = ListTable storageCredentials cellEntitiesContainerID query                                    
        let cellEntities = 
            cellEntityStores.ToArray()
            |> Array.map CellEntityStore.toCellEntity               
        return cellEntities
    
    }

let internal getInteractionEntities storageCredentials (interactionId:InteractionId) =   
    let guid = match interactionId with InteractionId guid -> guid
    async { 
        let query = (new TableQuery<InteractionEntityStore>()).Where(TableQuery.GenerateFilterCondition("interactionId", QueryComparisons.Equal, guid.ToString()));            
        let! entityStores = ListTable storageCredentials interactionEntitiesContainerID query                                    
        let entityStores = 
            entityStores.ToArray() 
            |> Array.map (fun iestore -> (iestore.Type |> InteractionNodeType.fromString,iestore.entityId |> System.Guid, iestore.entityType,iestore.complexIndex))            
            |> List.ofArray
        return entityStores
    }
    

let internal getAllInteractions storageCredentials =    
    async {
        let tableQ = TableQuery<InteractionStore>()
        let t0 = System.DateTime.Now;
        System.Console.Error.Write "Loading interactions..."
        let parts = 
            ListTable storageCredentials interactionsContainerID tableQ
            |> Async.RunSynchronously
            |> Seq.map (fun store -> 
                let interactionId = store.RowKey |> System.Guid |> InteractionId
                let entities = getInteractionEntities storageCredentials interactionId |> Async.RunSynchronously
                InteractionStore.toInteraction entities store
                )
                
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return parts
            
    }

let internal getAllParts storageCredentials =    
    async {
        let tableQ = TableQuery<PartStore>()
        let t0 = System.DateTime.Now;
        System.Console.Error.Write "Loading parts..."
        let parts = 
            ListTable storageCredentials partsContainerID tableQ
            |> Async.RunSynchronously
            |> Seq.map PartStore.toPart                                      
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return parts
            
    }

let internal getAllReagents storageCredentials =
    async {
        let tableQ = TableQuery<ReagentStore>()
        let t0 = System.DateTime.Now;                
        System.Console.Error.Write "Loading reagents..."
        let reagents =             
            ListTable storageCredentials reagentsContainerID tableQ
            |> Async.RunSynchronously                        
            |> Seq.map ReagentStore.toReagent                
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return reagents
            
    }

let internal getAllCells storageCredentials =    
    async {
        let tableQ = TableQuery<CellStore>()
        let t0 = System.DateTime.Now;
        System.Console.Error.Write "Loading cells..."
        let parts = 
            ListTable storageCredentials cellsContainerID tableQ
            |> Async.RunSynchronously
            |> Seq.map CellStore.toCell                                      
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return parts
            
    }

let internal getAllSamples storageCredentials = 
    async {
        let tableQ = TableQuery<SampleStore>()
        let t0 = System.DateTime.Now;                
        System.Console.Error.Write "Loading samples..."
        let result = 
            ListTable storageCredentials samplesContainerID tableQ
            |> Async.RunSynchronously
            |> Seq.map SampleStore.toSample
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return result
    }


let private FidOptToString (fid:FileId option) = 
    match fid with 
    | Some (FileId guid) -> guid.ToString()
    | None -> ""       

let internal getAllExperiments storageCredentials =    
    async {        
        let tableQ = TableQuery<ExperimentStore>()      
        let t0 = System.DateTime.Now;        
        System.Console.Error.Write "Loading experiments..."
        let result = 
            ListTable storageCredentials experimentsContainerID tableQ
            |> Async.RunSynchronously
            |> Seq.map ExperimentStore.toExperiment                
            |> Seq.toArray
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return result
    }


let internal getExperiment storageCredentials (ExperimentId experimentId) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentsContainerID
    
    async {
        let retrieve = TableOperation.Retrieve<ExperimentStore>("", experimentId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        //res.HttpStatusCode better?
        if res.Result <> null then
            let lookup = res.Result :?> ExperimentStore                        
            return ExperimentStore.toExperiment(lookup) |> Some
        else
            return None
        }

let getPart storageCredentials (partId:PartId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference partsContainerID

    async {
        let retrieve = TableOperation.Retrieve<PartStore>("",partId.ToString())
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask
        
        if res.Result <>  null then 
            return PartStore.toPart(res.Result :?> PartStore) |> Some
        else 
            return None
    }


let internal saveExperiment storageCredentials (experiment:Experiment) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentsContainerID    
    let store = ExperimentStore(experiment)
    async {
        //Store core experiment details
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask                        
        return res.HttpStatusCode = (int) System.Net.HttpStatusCode.NoContent
    }

let storeExperimentSignal storageCredentials (experimentId:ExperimentId) (signal:Signal) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference signalsContainerID
    let store = SignalStore(experimentId,signal)
    async {
        //Store Experiment Signal
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }    

let internal getExperimentSignal storageCredentials (experimentId:ExperimentId) (signalId:SignalId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference signalsContainerID

    async {
        let retrieve = TableOperation.Retrieve<SignalStore>("",experimentId.ToString() + signalId.ToString())
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask
        
        if res.Result <>  null then 
            return SignalStore.toSignal(res.Result :?> SignalStore) |> Some
        else 
            return None
    }

let deleteExperimentSignal storageCredentials (experimentId:ExperimentId) (signal:Signal) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference signalsContainerID
    let store = SignalStore(experimentId,signal)
    store.ETag <- "*"
    async {
        //Store Experiment Signal
        let tableOperation = TableOperation.Delete (store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask
        
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal storeExperimentEvent storageCredentials (experimentId: ExperimentId) (event: ExperimentOperation) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials      
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentEventsContainerID
    let store = ExperimentOperationStore(experimentId, event)
    async {      
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask                        
        //https://docs.microsoft.com/en-us/rest/api/storageservices/Insert-Or-Replace-Entity?redirectedfrom=MSDN
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let internal deleteExperimentEvent storageCredentials (experimentId:ExperimentId) (event:ExperimentOperation) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials      
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentEventsContainerID
    let store = ExperimentOperationStore(experimentId, event)
    store.ETag <- "*"
    
    async {      
        let tableOperation = TableOperation.Delete(store)
        let! res = table.ExecuteAsync(tableOperation) |> Async.AwaitTask                        
        //https://docs.microsoft.com/en-us/rest/api/storageservices/Insert-Or-Replace-Entity?redirectedfrom=MSDN
        return res.HttpStatusCode = (int)System.Net.HttpStatusCode.NoContent
    }

let private uploadFileBytesToContainer storageCredentials (containerID) (FileId fileGuid) (contents:byte[]) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let filesContainerReference = blobClient.GetContainerReference containerID
    let blob = filesContainerReference.GetBlockBlobReference (fileGuid.ToString())            
    blob.UploadFromByteArrayAsync(contents, 0, contents.Length) |> Async.AwaitTask
  
let internal uploadFileBytesToFilesContainer storageCredentials fileId (contents:byte[]) =
    uploadFileBytesToContainer storageCredentials filesContainerID fileId contents

let internal uploadFileBytesToTimeSeriesContainer storageCredentials fileId (contents:byte[]) =
    uploadFileBytesToContainer storageCredentials timeSeriesContainerID fileId contents

let rec private recFileBytesToTimeSeriesContainer (blob:CloudBlockBlob) fileId (contents:byte[]) (waitTime) = 
    if waitTime > 300000 then
        printfn "Number of retries to upload File %s exceeded." (fileId.ToString())
    else
        printfn "Issue uploading blob %s. Retrying after %d seconds" (fileId.ToString()) (waitTime/1000)
        Async.Sleep(waitTime) |> Async.RunSynchronously
        let res = blob.UploadFromByteArrayAsync(contents, 0, contents.Length) |> Async.AwaitTask |> Async.RunSynchronously
        Async.Sleep(10) |> Async.RunSynchronously
        let fileExists = blob.ExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously 
        match (fileExists) with 
        | true -> ()
        | false -> recFileBytesToTimeSeriesContainer blob fileId contents (waitTime*2)
            

let rec internal safeUploadFileBytesToTimeSeriesContainer storageCredentials fileId (contents:byte[]) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let filesContainerReference = blobClient.GetContainerReference timeSeriesContainerID
    let blob = filesContainerReference.GetBlockBlobReference (fileId.ToString())            
    printfn "Uploading file %s to the timeseries blobstore" (fileId.ToString())
    let res = blob.UploadFromByteArrayAsync(contents, 0, contents.Length) |> Async.AwaitTask |> Async.RunSynchronously
    Async.Sleep(10) |> Async.RunSynchronously
    let fileExists = blob.ExistsAsync() |> Async.AwaitTask |> Async.RunSynchronously
    match fileExists with
    | true -> async{return ()}
    | false -> 
        recFileBytesToTimeSeriesContainer blob fileId contents 100
        async{return ()}

let private uploadFileBytes storageCredentials (FileId fileId) (contents:byte[]) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let filesContainerReference = blobClient.GetContainerReference filesContainerID
    let blob = filesContainerReference.GetBlockBlobReference (fileId.ToString())            
    blob.UploadFromByteArrayAsync(contents, 0, contents.Length) |> Async.AwaitTask    

let private uploadFile storageCredentials fileId (contents:string) =    
    contents    
    |> System.Text.Encoding.UTF8.GetBytes
    |> uploadFileBytes storageCredentials fileId
    

let private uploadFilesBytes containerId storageCredentials (files:(FileId*byte[])[]) =
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let filesContainerReference = blobClient.GetContainerReference containerId
        
    files
    |> Array.map(fun (fileId,contents) ->
        let blob = filesContainerReference.GetBlockBlobReference (fileId.ToString())            
        blob.UploadFromByteArrayAsync(contents, 0, contents.Length)        
        )
    |> Threading.Tasks.Task.WhenAll
    |> Async.AwaitTask  


let private uploadFiles containerId storageCredentials (files:(FileId*string)[]) =    
    files
    |> Array.map(fun (fid,contents) -> 
        let contents' = 
            contents
            |> System.Text.Encoding.UTF8.GetBytes            
        fid, contents')
    |> uploadFilesBytes containerId storageCredentials
        

let internal getFile storageCredentials (FileId fileId) =         
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let fileContainerReference = blobClient.GetContainerReference filesContainerID
    (fileContainerReference.GetBlockBlobReference (fileId.ToString())).DownloadTextAsync() |> Async.AwaitTask

let internal getTimeSeriesFile storageCredentials (FileId fileId) =     
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let fileContainerReference = blobClient.GetContainerReference timeSeriesContainerID
    (fileContainerReference.GetBlockBlobReference (fileId.ToString())).DownloadTextAsync() |> Async.AwaitTask

let private generateLink (container:CloudBlobContainer) blobName fileName= 
    //Does this work without touching blob store?

    let blob = container.GetBlockBlobReference blobName

    let sasConstraints = SharedAccessBlobPolicy()
    sasConstraints.SharedAccessStartTime <- Some (System.DateTime.UtcNow.AddMinutes -5.0 |> System.DateTimeOffset) |> Option.toNullable //Clock drift
    sasConstraints.SharedAccessExpiryTime <- Some (System.DateTime.UtcNow.AddMinutes 6.0 |> System.DateTimeOffset) |> Option.toNullable
    sasConstraints.Permissions <- SharedAccessBlobPermissions.Read

        
    let headers = SharedAccessBlobHeaders()
    headers.ContentDisposition <- "attachment; filename=" + fileName

    let sasBlobToken = blob.GetSharedAccessSignature(sasConstraints, headers)
    blob.Uri.ToString() + sasBlobToken


let internal generateFileDownloadLink storageCredentials (FileId fileId) fileName =    
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let fileBlobContainerReference = blobClient.GetContainerReference filesContainerID
    let fileIdStr = fileId.ToString()
    async {                
        return (generateLink fileBlobContainerReference fileIdStr fileName)
    }

let internal generateTimeSeriesFileDownloadLink storageCredentials (FileId fileId) fileName =    
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let fileBlobContainerReference = blobClient.GetContainerReference timeSeriesContainerID
    let fileIdStr = fileId.ToString()
    async {                
        return (generateLink fileBlobContainerReference fileIdStr fileName)
    }


let internal generateAllDownloadLinks storageCredentials (fileIDs:FileId[]) =    
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let genbankBlobContainerReference = blobClient.GetContainerReference filesContainerID
    async {    
        let links = 
            fileIDs
            |> Array.map(fun (FileId fileId) -> 
                let idStr = fileId.ToString()
                (FileId fileId, generateLink genbankBlobContainerReference idStr idStr)
            )
            |> Map.ofSeq
        return links
    }



//TODO: avoid indexing files?

let internal getSample storageCredentials (sampleId:SampleId ) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let samplesTable = tableClient.GetTableReference samplesContainerID

    async {                      
        let! res = samplesTable.ExecuteAsync(TableOperation.Retrieve<SampleStore>("", SampleId.toString sampleId)) |> Async.AwaitTask        
        //res.HttpStatusCode better?
        if res.Result <> null then           
            let sampleStore = (res.Result :?> SampleStore)
            return Some (SampleStore.toSample(sampleStore))
        else
            return None
    }            

let internal getCell storageCredentials (cellId:CellId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let cellsTable = tableClient.GetTableReference cellsContainerID
    
    async {                      
        let! res = cellsTable.ExecuteAsync(TableOperation.Retrieve<CellStore>("", CellId.toString cellId)) |> Async.AwaitTask
        //res.HttpStatusCode better?
        if res.Result <> null then           
            let cellStore = (res.Result :?> CellStore)
            return Some (CellStore.toCell(cellStore))
        else
            return None
    }


let internal getInteraction storageCredentials (interactionId:InteractionId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials        
    let tableClient = storageAccount.CreateCloudTableClient()
    let interactionsTable = tableClient.GetTableReference interactionsContainerID
    async{
        let! res = interactionsTable.ExecuteAsync(TableOperation.Retrieve<InteractionStore>("", InteractionId.toString interactionId)) |> Async.AwaitTask
        //res.HttpStatusCode better?
        if res.Result <> null then           
            let store = (res.Result :?> InteractionStore)            
            let! entities = getInteractionEntities storageCredentials interactionId
            let interaction = InteractionStore.toInteraction entities store
            return (interaction |> Some)
        else
            return None           
    }

let internal getTimeSeries storageCredentials (sampleId:SampleId ) = 
    async {                
        let! sampleOpt = getSample storageCredentials sampleId    
        match sampleOpt with 
        | Some sample ->
            let guid = match sampleId with SampleId guid -> guid
            //printfn "%s" (sampleId.ToString())
            let! dataRefs = getFileRefs storageCredentials guid
            match dataRefs with 
            | [||] -> return None 
            | _ -> 
                let dataRef = dataRefs |> Seq.head //TODO: only single file is considered 
                let! data = getTimeSeriesFile storageCredentials dataRef.fileId            
                return Some data            //TODO: handle missing files?
        | None -> 
            return None    
    }

//let getFile storageCredentials  (FileId fileId) = 
//    let storageAccount = CloudStorageAccount.Parse storageCredentials
//    let blobClient = storageAccount.CreateCloudBlobClient()
//    let filesContainerReference = blobClient.GetContainerReference filesContainerID
//    let blob = filesContainerReference.GetBlobReference(fileId.ToString())               
//    async {        
//        let! exists = blob.ExistsAsync() |> Async.AwaitTask        
//        if exists then 
//            use ms = new System.IO.MemoryStream()
//            do! blob.DownloadToStreamAsync ms |> Async.AwaitTask                                   
//            let bytes = ms.ToArray()                                                            
//            let text = System.Text.Encoding.UTF8.GetString bytes
//            printfn "\n\n File on server: %s" text.[0..20]
//            return Some text
//        else    
//            return None
//    } 
    

//let exportDatabase storageCredentials =
//    failwith "Not implemented"
    //let storageAccount = CloudStorageAccount.Parse storageCredentials
    //let tableClient = storageAccount.CreateCloudTableClient()
    //let table = tableClient.GetTableReference barcodesContainerID

    //let tableQ = TableQuery<BarcodeStore>()

    //task {
    //    let cont = null

    //    let! result = table.ExecuteQuerySegmentedAsync(tableQ, cont)

    //    //TODO: loop for >1000
    //    let cont2 = result.ContinuationToken

    //    //TODO: do this with a StringBuilder
    //    let header = "Barcode\tFilename\tNotes\tSpecies\r\n"
    //    let resultsString = 
    //        result.Results
    //        |> Seq.map (fun lookup ->
    //            sprintf "%s\t%s\t%s\t%s" lookup.RowKey lookup.Filename lookup.Notes lookup.Species 
    //            )
    //        |> String.concat "\r\n"
    //    return header + resultsString
    //}

//let exportGenbank storageCredentials =
//    failwith "exportGenbank to be re-implemented"
//    let storageAccount = CloudStorageAccount.Parse storageCredentials

//    task {        
//        use memstream = new System.IO.MemoryStream()        
//        use zip = new ZipArchive(memstream, ZipArchiveMode.Create, true)
//        //TODO: investigate moving nested async to task as well
//        let! results = 
//            genBankBlobs
//            |> Seq.map (fun blob ->
//                let blob2 = CloudBlockBlob(blob.Uri, storageAccount.Credentials)
//                async {
//                    let name = blob.Uri.Segments |> Array.last |> Uri.UnescapeDataString
//                    use ms = new System.IO.MemoryStream()
//                    let! _ = (blob2.DownloadToStreamAsync ms) |> Async.AwaitTask
//                    return (name, ms.ToArray())
//                } )
//            |> Async.Parallel
//            |> Async.StartAsTask


//        //Not sure if ZipArchive is thread safe, so download everything first which is a shame
//        //https://msdn.microsoft.com/en-us/library/system.io.compression.zipfile(v=vs.110).aspx#Anchor_6

//        results |> Seq.iter (fun (name, data) ->
//            printfn "Adding: %s" name
//            let entry = zip.CreateEntry name
//            use entryStream = entry.Open()
//            entryStream.Write(data,0,data.Length))

//        printfn "Disposing"
//        zip.Dispose() //Early dispose, force checksums, needed?

//        //We might be able to stream responses but need to check rules on ZipArchive
//        return memstream.ToArray()
//    }





let ListPlateReaderRawFile storageCredentials =
    //https://github.com/Azure/azure-storage-net/issues/352
    //Can't switch to SAS until that's resolved, issues with workaround
    ListBlobs storageCredentials rawPlateReaderContainerID |> Async.RunSynchronously |> Seq.map (fun blob -> blob.Uri.Segments |> Array.last |> Uri.UnescapeDataString)

let ListPlateReaderUniformFile storageCredentials =
    //https://github.com/Azure/azure-storage-net/issues/352
    //Can't switch to SAS until that's resolved, issues with workaround
    ListBlobs storageCredentials uniformPlateReaderContainerID |> Async.RunSynchronously |> Seq.map (fun blob -> blob.Uri.Segments |> Array.last |> Uri.UnescapeDataString)

let ListPlateReaderUniformFileBetween storageCredentials startDateTime endDateTime =
    //https://github.com/Azure/azure-storage-net/issues/352
    //Can't switch to SAS until that's resolved, issues with workaround
   
    ListBlobs storageCredentials uniformPlateReaderContainerID
    |> Async.RunSynchronously
    |> Seq.map (fun blob ->
        let filename = blob.Uri.Segments |> Array.last |> Uri.UnescapeDataString

        let stem = filename.[0..(filename.Length - 5)].Split('_')

        let dateTime = DateTime.Parse stem.[0]        
        let machineIdentifier = stem.[1]
        
        { dateTime = dateTime;  machineIdentifier = machineIdentifier; fullFileName = filename}
    )
    |> Seq.filter(fun uniformFile ->
        //TODO: Discuss inclusivity...
        uniformFile.dateTime > startDateTime &&
        uniformFile.dateTime < endDateTime &&
        uniformFile.machineIdentifier = "aardvark" //Currently we only supply data read from the machine labelled as "aardvark"
        )

let LoadPlateReaderRawFile storageCredentials path =

    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let container = blobClient.GetContainerReference rawPlateReaderContainerID

    let reference = container.GetBlockBlobReference path
    //TODO: expose async to client, switch to task?
    async {
        let! exists = reference.ExistsAsync() |> Async.AwaitTask
        if not exists then
            failwithf "Not found %s" path

        let! text = reference.DownloadTextAsync() |> Async.AwaitTask
        return text
        
    } |> Async.RunSynchronously

let LoadPlateReaderUniformFile storageCredentials path =

    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let container = blobClient.GetContainerReference uniformPlateReaderContainerID

    let reference = container.GetBlockBlobReference path
    //TODO: expose async to client, switch to task?
    async {
        let! exists = reference.ExistsAsync() |> Async.AwaitTask
        if not exists then
            failwithf "Not found %s" path

        let! text = reference.DownloadTextAsync() |> Async.AwaitTask
        return text
        
    } |> Async.RunSynchronously

let SavePlateReaderRawFile storageCredentials (path:string) =

    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let container = blobClient.GetContainerReference rawPlateReaderContainerID

    let blobName = System.IO.Path.GetFileName path

    async {
        let blobReference = container.GetBlockBlobReference blobName
        do! 
            blobReference.UploadFromFileAsync path
            |> Async.AwaitTask
    }



let SavePlateReaderUniformFile storageCredentials uniformFileUpload =

    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let blobClient = storageAccount.CreateCloudBlobClient()
    let container = blobClient.GetContainerReference uniformPlateReaderContainerID

    async {
        let blobReference = container.GetBlockBlobReference uniformFileUpload.name
        do! 
            blobReference.UploadTextAsync uniformFileUpload.contents
            |> Async.AwaitTask
    }

let internal saveEvent storageCredentials (event:Event) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let eventsTable = tableClient.GetTableReference eventsV2ContainerID
    let store = EventStore(event)
    
    async {      
        let tableOperation = TableOperation.InsertOrReplace(store)
        let! res = eventsTable.ExecuteAsync(tableOperation) |> Async.AwaitTask                               
        res |> ignore
    }
    |> Async.Start


let internal GetEvents storageCredentials (targetGuid:System.Guid option) = 
    let query = 
       match targetGuid with
       | Some guid -> (new TableQuery<EventStore>()).Where(TableQuery.GenerateFilterCondition("targetId", QueryComparisons.Equal, guid.ToString()))
       | None -> new TableQuery<EventStore>()
    
    //TODO: order of events?
    async {
        let! events = ListTable storageCredentials eventsV2ContainerID query
        let result = 
            events.ToArray()
            |> Array.sortBy(fun e -> e.timestamp) //NOTE: sorting by original timestamp and not the Azure tables one
            |> Array.map EventStore.toEvent            
        return result
    }

let internal GetEventsLog storageCredentials =     
    async {
        let! events = GetEvents storageCredentials None
        let result = 
            events
            |> Array.map Event.encode
            |> String.concat ","
            |> sprintf "[%s]"
        return result
    }


//Interface functions
let internal UploadBundle uploadFile (fileId:FileId, content:(string*string)[]) =           
       use memstream = new System.IO.MemoryStream()
       use zip = new ZipArchive(memstream, ZipArchiveMode.Create, true)    
       content
       |> Seq.iter(fun (name,dataString) ->         
           let entry = zip.CreateEntry name
           use entryStream = entry.Open()        
           let data = System.Text.Encoding.UTF8.GetBytes dataString        
           entryStream.Write(data,0,data.Length)
           ) 
       zip.Dispose() //Early dispose, force checksums, needed?        
       uploadFile (fileId, memstream.ToArray())  

let ListCRNserverbuilds connectionString = ListBlobs connectionString crnServerBuildsID
let ListClassicdsdserverbuilds connectionString = ListBlobs connectionString classicDSDServerBuildsID

let GetCRNServerLink connectionString named =
    let storageAccount = CloudStorageAccount.Parse connectionString
    let blobClient = storageAccount.CreateCloudBlobClient()
    let containerReference = blobClient.GetContainerReference crnServerBuildsID

    generateLink containerReference named named

let GetClassicDSDServerLink connectionString named =
    let storageAccount = CloudStorageAccount.Parse connectionString
    let blobClient = storageAccount.CreateCloudBlobClient()
    let containerReference = blobClient.GetContainerReference classicDSDServerBuildsID

    generateLink containerReference named named

module DataProcessor = 
    let TryParseExperimentLayout tryGetExperiment (getFileRefs:System.Guid -> Async<FileRef[]>) getFile (getSamplesForExperiment:ExperimentId -> Async<Sample[]>) (experimentId: ExperimentId) = 
        let ParseAnthaLayout (content:string) = 
            let lines = content.Split([|'\n'; '\r'|], System.StringSplitOptions.RemoveEmptyEntries)
            let headers = lines.[0].Split(',')
            let sid = headers |> Array.findIndex ((=) "Run")
            let lid = headers |> Array.findIndex ((=) "Location")
            let pid = headers |> Array.findIndex ((=) "Plate")            
            lines.[1..]
            |> Array.map(fun line ->                 
                let fields = line.Split(',')                                                
                let sampleId = fields.[sid] |> System.Guid |> SampleId
                let well = fields.[lid]
                let plate = fields.[pid]
                    
                let wellPos = 
                    { row = (int well.[0]) - 65
                      col = (int well.[1..]) - 1
                    }
                (sampleId, (wellPos, plate))
                )
            |> Map.ofSeq
    
        async {    
            let! experiment = tryGetExperiment experimentId
            match experiment with
            | Some e ->
                let guid = match experimentId with | ExperimentId guid -> guid
                let! filerefs = getFileRefs guid
                match filerefs |> Array.tryFind (fun f -> f.Type = FileType.AnthaPlateLayout) with
                | Some plateRef -> 
                    let! anthaLayoutFile = getFile plateRef.fileId
                    let anthaSamples = ParseAnthaLayout anthaLayoutFile                    
                    let! experimentSamples = getSamplesForExperiment experimentId
                    
                    let updatedSamples = 
                        experimentSamples
                        |> Array.choose(fun s -> 
                            match anthaSamples.TryFind s.id with 
                            | Some (well,plateName) -> 
                                match s.meta with 
                                | PlateReaderMeta meta ->
                                    let meta' = 
                                        {meta with 
                                            physicalWell = Some well
                                            physicalPlateName = Some plateName
                                        }                    
                                    Some {s with meta = PlateReaderMeta meta'}                        
                                | MissingMeta -> None                        
                            | None -> None
                            )
                    return Some(updatedSamples)
                | None -> return None
            | None -> return None
            }      


    let private LoadDataBetween listPlateReaderUniformFileBetween loadPlateReaderUniformFile (fromUTC:System.DateTime) (toUTC:System.DateTime) = 
        printfn "Parsing data in range %A to %A" fromUTC toUTC
        
        listPlateReaderUniformFileBetween (fromUTC, toUTC)
        |> Seq.map (fun uniformFileName ->       
            let content = loadPlateReaderUniformFile uniformFileName.fullFileName            
            content
            )
        |> Seq.map PlateReaderFileParser.parseFrom
        |> Array.ofSeq


    let private TryProcessData (signals:Signal[]) (data:PlateReaderFileParser.PlateReaderRun[]) =
        
        if Array.isEmpty data then 
            printfn "WARNING: No data for processing"
            None
        else
            let startDateTime = data.[0].ReadingDateTime
            //TODO: Extract temperature?
            let fluorescenceReads = data |> Seq.filter (fun read -> read.Configuration = PlateReaderFileParser.Configuration.Fluorescence) |> Array.ofSeq
            let absorbanceReads = data |> Seq.filter (fun read -> read.Configuration = PlateReaderFileParser.Configuration.Absorbance) |> Array.ofSeq
        
            //TODO: merge times from absorbance, etc
            let times =
                fluorescenceReads
                |> Seq.map (fun measurement -> (measurement.ReadingDateTime - startDateTime).TotalHours)        
                |> Array.ofSeq
    
            //TODO: Crop frames for consistency (observed one frame short on absorbance)
            let max_frames = [fluorescenceReads.Length; absorbanceReads.Length; times.Length] |> Seq.min
            let fluorescenceReads = fluorescenceReads |> Array.take max_frames
            let absorbanceReads = absorbanceReads |> Array.take max_frames        
            let times = times |> Array.take max_frames
    
            //TODO: Assuming that all fluorescence and absorbance reads match on wells
            let nRows = data.[0].Data.[0].Length
            let nCols = data.[0].Data.[0].[0].Length
        
            //TODO: assume the same signals for all reads
            let fluorescenceSignals = 
                fluorescenceReads.[0].Filters.Value 
                |> Array.map(fun s -> 
                    signals 
                    |> Array.find(fun s' -> 
                        let str = 
                            match s'.settings with 
                            | PlateReaderFluorescence o -> sprintf "%s/%s" (PlateReaderFilter.toString o.excitationFilter) (PlateReaderFilter.toString o.emissionFilter) //TODO: implement a less fragile matching of signals  
                            | _ -> ""
                        s = str
                        ))        
            let absorbanceSignals = 
                absorbanceReads.[0].Wavelengths.Value
                |> Array.map(fun w -> 
                    let matchedSignal = 
                        signals 
                        |> Array.tryFind(fun w' -> 
                            let str = 
                                match w'.settings with 
                                | PlateReaderAbsorbance o -> (sprintf "%.0fnm" o.wavelength) //TODO: implement a less fragile matching of signals  
                                | _ -> ""
                            w = str
                            )
                    match matchedSignal with 
                    | Some s -> s
                    | None -> failwithf "ERROR: Signal %s not match to BCKG experiment definition"  w                    
                    )        
    
            [0..nRows-1]
            |> Seq.map(fun row -> 
                [0..nCols-1]
                |> Seq.map(fun col -> 
                    let fluorObs = 
                        fluorescenceSignals
                        |> Seq.mapi(fun i signal -> 
                            let data = fluorescenceReads |> Array.map(fun measurements -> float measurements.Data.[i].[row].[col].Value)                                                       
                            (signal.id, data)
                            )
                    let absObs = 
                        absorbanceSignals
                        |> Seq.mapi(fun i signal -> 
                            let data = absorbanceReads |> Array.map(fun measurements -> float measurements.Data.[i].[row].[col].Value)                                                       
                            (signal.id, data)
                            )
                    
                    let observations = 
                        Seq.append fluorObs absObs                                
                        |> Map.ofSeq
    
                    let ts = 
                        { times = times |> Array.map Hours //TODO: assuming data is in hours
                          observations = observations
                        }
                    let well = {row=row; col=col}
                    (well, ts)
                )
            )
            |> Seq.concat
            |> Map.ofSeq
            |> Some
    

//    TryProcessData

    let ProcessExperimentData (tryGetExperiment:ExperimentId -> Async<Experiment option>) (getExperimentOperations:ExperimentId -> Async<ExperimentOperation[]>) (getExperimentSignals:ExperimentId -> Async<Signal[]>) getSamplesForExperiment listPlateReaderUniformFileBetween loadPlateReaderUniformFile (experimentId:ExperimentId) =                            
        async {        
            printfn "\nProcessing data for experiment %s" (ExperimentId.toString experimentId)
            let! experimentOpt = tryGetExperiment experimentId 
            match experimentOpt with 
            | Some experiment -> 
                let! experimentOperations = getExperimentOperations experimentId
                let! experimentSignals = getExperimentSignals experimentId
                let readStartedEvent = experimentOperations |> Array.tryFind(fun e -> e.Type = PlateReaderStarted)
                let readFinishedEvent = experimentOperations |> Array.tryFind(fun e -> e.Type = ExperimentFinished)
                match readStartedEvent, readFinishedEvent with 
                | Some started, Some finished ->                                    
                    //(System.DateTime.Parse "2018-11-22 14:50", System.DateTime.Parse "2018-11-23 11:40")                
                    //printfn "Loading data from %s to %s" (started.timestamp.ToString("yyyy-MM-dd hh:mm:ss")) (finished.timestamp.ToString("yyyy-MM-dd hh:mm:ss"))                
                    let dataOption = //NOTE: data is labeled by physical wells
                        let started' = started.timestamp.Subtract(System.TimeSpan(0,10,0)) //10min buffer
                        let finished' = finished.timestamp.Add(System.TimeSpan(0,10,0)) //10min buffer
                        
                        LoadDataBetween listPlateReaderUniformFileBetween loadPlateReaderUniformFile started' finished'
                        |> TryProcessData experimentSignals 
                    match dataOption with 
                    | Some data -> 
                        let! samples =  getSamplesForExperiment experimentId
                            
                        let sampleDataMap =
                            samples
                            |> Array.choose(fun sample -> 
                                match sample.meta with 
                                | PlateReaderMeta m ->  
                                    match m.physicalWell with 
                                    | Some well ->                                 
                                        let sampleData = data.[well]
                                        let dataFile = sampleData |> TimeSeries.toString                                    
                                        let dataFileRef = FileRef.Create(sprintf "%s.csv" (sample.id.ToString()), FileType.CharacterizationData)
                                        
                                        //TODO: assume that there is a single characterization file. If a file is already associated, then replace it
                                        //TODO? printfn "WARNING: Multiple characterization files detected for sample %A...skipping" sample.id
                                        Some(sample,(dataFileRef.fileId, dataFile))
                                        
                                            
                                    | None -> None
                            
                                | MissingMeta -> None                                        
                                )   
                            
                        
                        return Some(sampleDataMap)
                    
                    | None -> return None                                   
                | _ -> return None
            | None -> return None            
    
            //TODO: updates to model?
        }
    
  
let internal tryGetCellParent storageCredentials (CellId cellGuid) =         
    async { 
        let query = (new TableQuery<DerivedFromStore>()).Where(TableQuery.GenerateFilterCondition("target", QueryComparisons.Equal, cellGuid.ToString()));            
        let! derivedFromStores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivations = 
            derivedFromStores.ToArray()
            |> Array.map DerivedFromStore.toDerivedFrom
            |> Array.choose (fun x -> match x with | DerivedFrom.CellLineage (source, _ ) -> Some source | _ -> None)
           
        //TODO: Generalize for multiple derivations        
        if Array.isEmpty derivations then 
            return None
        else
            return (Some derivations.[0])
    }

let internal tryGetObservation storageCredentials (observationId: ObservationId) = 
    let storageAccount = CloudStorageAccount.Parse storageCredentials
    let tableClient = storageAccount.CreateCloudTableClient()
    let table = tableClient.GetTableReference experimentsContainerID
  
    async {
        let retrieve = TableOperation.Retrieve<ObservationStore>("", observationId.ToString())        
        let! res = table.ExecuteAsync(retrieve) |> Async.AwaitTask

        if res.Result <> null then
            let lookup = res.Result :?> ObservationStore                        
            return ObservationStore.toObservation(lookup) |> Some
        else
            return None
        }


let internal getObservations storageCredentials (SampleId sampleGuid) (SignalId signalGuid)=            
    let sampleCond = TableQuery.GenerateFilterCondition("sample", QueryComparisons.Equal, sampleGuid.ToString())
    let signalCond = TableQuery.GenerateFilterCondition("signal", QueryComparisons.Equal, signalGuid.ToString())
    let query = (new TableQuery<ObservationStore>()).Where(TableQuery.CombineFilters(sampleCond, TableOperators.And, signalCond))
    
    async {
            let! observations = ListTable storageCredentials observationsContainerID query
            let result = 
                observations.ToArray()
                |> Array.map ObservationStore.toObservation            
            return result
        }

let internal getSampleObservations storageCredentials (sampleId)  =            
    let query = (new TableQuery<ObservationStore>()).Where(TableQuery.GenerateFilterCondition( "sample", QueryComparisons.Equal, sampleId.ToString()))
    
    async {
            let! observations = ListTable storageCredentials observationsContainerID query
            let result = 
                observations.ToArray()
                |> Array.map ObservationStore.toObservation            
            return result
        }

let internal tryGetCellSourceExperiment  storageCredentials (CellId cellGuid)  = 
    async { 
        let query = (new TableQuery<DerivedFromStore>()).Where(TableQuery.GenerateFilterCondition("target", QueryComparisons.Equal, cellGuid.ToString()));            
        let! derivedFromStores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivations = 
            derivedFromStores.ToArray()
            |> Array.map DerivedFromStore.toDerivedFrom
            |> Array.choose (fun x -> match x with | DerivedFrom.CellTransformation (source, _ ) -> Some source | _ -> None)
            
        //TODO: Generalize for multiple derivations        
        if Array.isEmpty derivations then 
            return None
        else
            return (Some derivations.[0])
    }


let internal tryGetDeviceSourceExperiment storageCredentials (DNAId dnaGuid)  = 
    async { 
        let query = (new TableQuery<DerivedFromStore>()).Where(TableQuery.GenerateFilterCondition("target", QueryComparisons.Equal, dnaGuid.ToString()));            
        let! derivedFromStores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivations = 
            derivedFromStores.ToArray()
            |> Array.map DerivedFromStore.toDerivedFrom
            |> Array.choose (fun x -> match x with | DerivedFrom.DNAAssembly (source, _ ) -> Some source | _ -> None)
            
        
        //TODO: Generalize for multiple derivations        
        if Array.isEmpty derivations then 
            return None
        else
            return (Some derivations.[0])
    }

let internal getDeviceComponents storageCredentials (DNAId dnaGuid)  = 
    async { 
        let query = (new TableQuery<DerivedFromStore>()).Where(TableQuery.GenerateFilterCondition("target", QueryComparisons.Equal, dnaGuid.ToString()));            
        let! derivedFromStores = ListTable storageCredentials derivedFromContainerID query                                    
        let derivations = 
            derivedFromStores.ToArray()
            |> Array.map DerivedFromStore.toDerivedFrom
            |> Array.choose (fun x -> match x with | DerivedFrom.DNAComponent (source, _ ) -> Some source | _ -> None)
    
        return derivations
    }



(*
        INDEXING METHODS (TODO)

*)


//NOTE: getFileRefs is slow when called repeatedly with different guids. Instead, build the complete map once
//TODO: cache index? Cache invalidation?
let internal IndexFiles storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing files..."
        let! queryResult = ListTable storageCredentials filesMapContainerID (TableQuery<FileRefStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.source)
            |> Array.map(fun (key, entries) -> 
                System.Guid.Parse key, entries |> Array.map FileRefStore.toFileRef)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }

let internal IndexCellEntities storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing cell entities..."
        let! queryResult = ListTable storageCredentials cellEntitiesContainerID (TableQuery<CellEntityStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.cellId)
            |> Array.map(fun (key, entries) -> 
                CellId (System.Guid.Parse key), entries |> Array.map CellEntityStore.toCellEntity)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }
    
let internal IndexTags storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing tags..."
        let! queryResult = ListTable storageCredentials tagsContainerID (TableQuery<TagStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.source)
            |> Array.map(fun (key, entries) -> 
                System.Guid.Parse key, entries |> Array.map (fun store -> Tag store.tag))
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }
    
let internal IndexExperimentOperations storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing experiment events..."
        let! queryResult = ListTable storageCredentials experimentEventsContainerID (TableQuery<ExperimentOperationStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.source) //TODO: Can the source be something other than an ExperimentId? (maybe in the future)
            |> Array.map(fun (key, entries) -> 
                 ExperimentId (System.Guid.Parse key), entries |> Array.map ExperimentOperationStore.toExperimentEvent)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }

let internal IndexExperimentSignals storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing experiment signals..."
        let! queryResult = ListTable storageCredentials signalsContainerID (TableQuery<SignalStore>())
        let index= 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.experimentId)
            |> Array.map(fun (key, entries) -> 
                ExperimentId (System.Guid.Parse key), entries |> Array.map SignalStore.toSignal)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }

let internal IndexExperimentSamples storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing experiment samples..."
        let! queryResult = ListTable storageCredentials samplesContainerID (TableQuery<SampleStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.experimentId)
            |> Array.map(fun (key, entries) -> 
                ExperimentId (System.Guid.Parse key), entries |> Array.map SampleStore.toSample)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }

let internal IndexSampleConditions storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing sample conditions..."        
        let! queryResult = ListTable storageCredentials sampleConditionsContainerID (TableQuery<ConditionStore>())
        let index = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.sampleId)
            |> Array.map(fun (key, entries) -> 
                let sampleId = System.Guid.Parse key
                let conditions = Array.map ConditionStore.toCondition entries                                     
                (SampleId sampleId), conditions                
                )
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }

let internal IndexSampleCells storageCredentials = 
    async {
        let t0 = System.DateTime.Now 
        System.Console.Error.Write "Indexing sample cells..."
        let! queryResult = ListTable storageCredentials sampleDevicesContainerID (TableQuery<SampleCellStore>())
        let index  = 
            queryResult.ToArray()
            |> Array.groupBy (fun r -> r.sampleId)
            |> Array.map(fun (key, entries) -> 
                let key = SampleId (System.Guid.Parse key)
                let cells = 
                    entries 
                    |> Array.map (fun dstore -> 
                        let cellId = dstore.cellId |> System.Guid |> CellId
                        let cellDensity = if dstore.cellDensity < 0.0 then None else Some dstore.cellDensity
                        let cellPreSeeding = if dstore.cellPreSeeding < 0.0 then None else Some dstore.cellPreSeeding
                        cellId, (cellDensity, cellPreSeeding)
                    )
                (key, cells))
            |> Map.ofSeq 
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }


let internal IndexObservations storageCredentials = 
    async{
        let t0 = System.DateTime.Now
        System.Console.Error.Write "Indexing observations..."
        let! queryResult = ListTable storageCredentials observationsContainerID (TableQuery<ObservationStore>())
        let index = 
            queryResult.ToArray()
            |> Array.map(fun store -> 
                let obs = store |> ObservationStore.toObservation
                (obs.sampleId, obs.signalId), obs
                )
            |> Array.groupBy fst
            |> Array.map (fun (x,L) -> x, L |> Array.map snd)
            |> Map.ofSeq      
        System.Console.Error.WriteLine ("\tdone in {0:F2} seconds", (System.DateTime.Now-t0).TotalSeconds)
        return index
    }