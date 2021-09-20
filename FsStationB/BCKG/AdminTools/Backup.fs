module BCKG.Admin.Backup


let GetBckgBackup (db:BCKG.API.Instance) =
    failwith "Not implemented yet."
   
let BackupStateTo (db:BCKG.API.Instance) filename = 
    let file = GetBckgBackup db |> Async.RunSynchronously
    System.IO.File.WriteAllText(filename, file)

let BackupEventsTo (db:BCKG.API.Instance) filename =
    let content = db.GetLogJson() |> Async.RunSynchronously
    System.IO.File.WriteAllText(filename, content)    

let DownloadBlobs (db:BCKG.API.Instance) container filename = 
    let bytes = db.GetBlobsZip container |> Async.RunSynchronously
    System.IO.File.WriteAllBytes(filename, bytes)