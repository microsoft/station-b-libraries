module FSTL.BoundedSTL

open FSTL
open FSTL.Signal
open FSTL.Lib


type BoundedSpace = {signal:string;lowerBound:double;upperBound:double}

type BoundedRegion = {
    //lowerTimeBound:double;
    //upperTimeBound:double;
    signalBounds:BoundedSpace list
    traceInterval:double
    }
with
    member this.maxSignalRobustness (signal:string) = 
        let x = this.signalBounds |> List.find (fun b -> b.signal = signal)
        x.upperBound - x.lowerBound
    member this.minSignalRobustness (signal:string) =
        (this.maxSignalRobustness signal) * (-1.0)
    member this.maximumRobustness = 
        (this.signalBounds 
        |> List.map (fun b -> 
            b.upperBound - b.lowerBound)
        |> List.max)
    member this.minimumRobustness = 
        this.maximumRobustness * (-1.0) 
    member this.getSignal (signals:Signal list) (signalName:string) = 
        signals
        |> List.tryFind(fun s -> s.signal = signalName)
    
    member this.robustness (stl:Operator) (signals:Signal list) (time:double)= 
        match stl with 
        | Conjunction(o1,o2) -> min( (this.robustness o1 signals time),(this.robustness o2 signals time))
        | Disjunction(o1,o2) -> max( (this.robustness o1 signals time),(this.robustness o2 signals time))
        | Negation(o1) -> (-1.0) * (this.robustness o1 signals time) 
        | BooleanNode (o1) -> 
            match o1 with 
            | true -> this.maximumRobustness
            | false -> this.minimumRobustness
        | LinearPredicate(lp) ->
            let sigopt = this.getSignal signals lp.signal
            match sigopt with 
            | Some(s) -> 
                let pointOption = Signal.atTime (s.trace) time
                match pointOption with 
                | Some(t,v) ->
                    match lp.op with 
                    | GreaterThan -> v - lp.value
                    | GreaterThanEqualTo -> v - lp.value
                    | LessThan -> lp.value - v 
                    | LessThanEqualTo -> lp.value - v
                    | Equals -> 
                        match (v = lp.value) with 
                        | true -> this.maxSignalRobustness lp.signal
                        | false -> this.minSignalRobustness lp.signal
                    | NotEquals -> 
                        match (v = lp.value) with 
                        | false -> this.maxSignalRobustness lp.signal
                        | true -> this.minSignalRobustness lp.signal
                | None -> 
                    this.minimumRobustness
            | None -> this.minSignalRobustness (lp.signal)
            
        | Always (o1) -> 
            [(o1.interval.startTime)..(this.traceInterval)..(o1.interval.endTime)]
            |> List.map (fun i -> this.robustness (o1.operator) (signals) (i + time))
            |> List.min
        | Eventually (o1) -> 
            [(o1.interval.startTime)..(this.traceInterval)..(o1.interval.endTime)]
            |> List.map (fun i -> this.robustness (o1.operator) (signals) (i + time))
            |> List.max
        | Until(o1) -> 
            [(o1.interval.startTime)..(this.traceInterval)..(o1.interval.endTime)]
            |> List.map (fun i -> 
                [
                    (this.robustness (o1.right) (signals) (i+time));
                    [time..(this.traceInterval)..(i+time)]
                    |> List.map (fun j -> this.robustness (o1.left) (signals) (j))
                    |> List.min
                ])
            |> List.fold (fun acc x -> acc@x) []
            |> List.max

            