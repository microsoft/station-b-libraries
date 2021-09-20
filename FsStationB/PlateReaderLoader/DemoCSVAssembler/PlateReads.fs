module PlateReads

open System.IO

type Measurement = 
    { dateTime : System.DateTime
      name : string
      content : unit -> string }