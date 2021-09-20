module FSTL.Signal


type Trace = (double * double) list

type Signal = {signal:string;trace:Trace}

let isValid (trace:Trace) = 
    let nonNegative = 
        trace 
        |> List.map(fun (t,_) -> (t < 0.0))
        |> List.fold (fun acc x -> (acc && x)) true
    let stopTime = 
        let timepoints = trace |> List.map (fun (t,_) -> t)
        let distinct = timepoints |> List.distinct
        (distinct.Length) = (timepoints.Length)
    
    (nonNegative && stopTime)    

let arrange (trace:Trace):Trace = 
    trace |> List.sortBy (fun (t,_) -> t)

let timepoints (trace:Trace) (start:double) (stop:double) = 
    trace 
    |> arrange
    |> List.filter (fun (t,_) ->  (t >= start) && (t<=stop))

let interpolate ((x1,y1):double*double) ((x2,y2):double*double) (x:double) = 
    let m = (y2 - y1)/(x2-x1)
    let y = (m * (x - x1)) + y1
    (x,y)

let atTime (trace:Trace) (time:double) = 
    let sortedTrace = trace |> arrange
    let pointAtTime = 
        sortedTrace
        |> List.tryFind(fun (t,v) -> t = time)
    match pointAtTime with 
    | Some(t,v) -> Some(t,v)
    | None -> 
        match time with 
        | t when (t < (fst sortedTrace.Head)) -> None 
        | t when (t > (fst (sortedTrace |> List.rev).Head )) -> None
        | _ -> 
            let p1 = sortedTrace |> List.findBack (fun (t,v) -> t < time)
            let p2 = sortedTrace |> List.find (fun (t,v) -> t > time)
            Some(interpolate p1 p2 time)