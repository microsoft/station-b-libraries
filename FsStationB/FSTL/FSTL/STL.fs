// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
namespace FSTL

open FSTL.Lib

type Interval = {startTime:double;endTime:double}

type RelativeOperator = 
    | LessThan
    | LessThanEqualTo
    | GreaterThan
    | GreaterThanEqualTo
    | Equals
    | NotEquals

type LinearPredicate = {signal:string;op:RelativeOperator;value:double}

type Operator = 
    | BooleanNode of bool 
    | LinearPredicate of LinearPredicate
    | Always of AlwaysNode
    | Eventually of EventuallyNode
    | Until of UntilNode
    | Negation of Operator
    | Conjunction of (Operator*Operator)
    | Disjunction of (Operator*Operator)
    member this.negate = 
        match this with
        | BooleanNode(b) -> BooleanNode(not b)
        | LinearPredicate (lp) -> 
            let rop = 
                match lp.op with 
                | LessThan -> GreaterThanEqualTo
                | LessThanEqualTo -> GreaterThan
                | GreaterThan -> LessThanEqualTo
                | GreaterThanEqualTo -> LessThan
                | Equals -> NotEquals
                | NotEquals -> Equals
            LinearPredicate({signal=lp.signal;op = rop;value=lp.value})
        | Always(an) -> Always({interval=an.interval;operator=an.operator.negate})
        | Eventually(en) -> Eventually({interval=en.interval;operator=en.operator.negate})
        | Until(un) -> Until({interval=un.interval;left=un.left.negate;right=un.right.negate})
        | Conjunction(l,r) -> Disjunction(l.negate,r.negate)
        | Disjunction(l,r) -> Conjunction(l.negate,r.negate)
        | Negation o -> o
    member this.horizon = 
        match this with 
        | BooleanNode _ -> 0.0
        | LinearPredicate _ -> 0.0
        | Always (o) -> o.interval.endTime + (o.operator.horizon)
        | Eventually (o) -> o.interval.endTime + (o.operator.horizon)
        | Until (o) -> (o.interval.endTime) + max(o.left.horizon,o.right.horizon)
        | Conjunction(l,r) -> max(l.horizon,r.horizon)
        | Disjunction(l,r) -> max(l.horizon,r.horizon)
        | Negation (o) -> o.horizon
and AlwaysNode = {interval:Interval;operator:Operator}
and EventuallyNode = {interval:Interval;operator:Operator}
and UntilNode = {interval:Interval;left:Operator;right:Operator}

