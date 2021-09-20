module BCKG.Tests

open Expecto
open BCKG.Domain
open BCKG.Events
open Thoth.Json.Net

let tests =
    let db1 = BCKG.API.Instance(BCKG.API.MemoryInstance,"TestUser")
    let db2 = BCKG.API.Instance(BCKG.API.MemoryInstance,"TestUser2")
    let db3 = BCKG.API.Instance(BCKG.API.MemoryInstance,"TestUser3")
    

    testList "BCKG Tests" [
        
        testAsync "Part Events" { 
            let db = db1    
            let part1 = 
                UserDefined({   
                    id = UserDefinedId.Create();
                    properties = {
                        name = "Terminator1"
                        sequence = "ATGCGGCCCGGAATTAGAGA"
                        deprecated = false
                    } 
                })

            let! _ = db.SavePart part1
            let! part1opt = db.TryGetPart part1.id
            Expect.isSome part1opt "Part has a value"
            let part1' = match part1opt with | Some(p) -> p | None -> failwith "Part Option really can't be none at this point."
            
            Expect.equal part1.id part1'.id "Part ID"
            Expect.equal part1.name part1'.name "Part Name"
            Expect.equal part1.getProperties.sequence part1'.getProperties.sequence "Part Sequence"
            Expect.equal part1.getType part1'.getType "Part Type"            
            
            let! parts = db.GetParts()
            Expect.equal parts.Length 1 "Part count"
            Expect.equal parts.[0] part1 "Correct part created"
            
            let! events = db.GetEvents()
            let partEvents = events |> Array.filter (fun e -> e.target = PartEvent(part1.id))
            Expect.equal partEvents.Length 1 "Part Event Count"
            
            let partEvent = partEvents.[0]
            let changes = Decode.Auto.unsafeFromString<(string*string)list>(partEvent.change)
            Expect.hasLength changes 4 "Add Part change length"

            let (paddName,pname) = changes |> List.find (fun (c,cstring) -> c = Part.addName)
            let (paddType,ptype) = changes |> List.find (fun (c,cstring) -> c = Part.addType)
            let (paddSequence,psequence) = changes |> List.find (fun (c,cstring) -> c = Part.addSequence)
            let (paddDeprecated,isDeprecated) = changes |> List.find (fun (c,cstring) -> c = Part.addDeprecate)
            
            Expect.equal pname part1.name "Change string part name"
            Expect.equal ptype (part1.getType) "Change string part type"
            Expect.equal psequence part1.getProperties.sequence "Change string part sequence"
            
            let part2 = Terminator({id = part1.guid |> TerminatorId; properties = part1.getProperties})
            let! _ = db.SavePart part2
            let! part2opt = db.TryGetPart part1.id
            Expect.isSome part2opt "Try Get Part has a value - After modification."
            let part2' = match part2opt with | Some(p) -> p | None -> failwith "Part Option really can't be none at this point."
            
            Expect.equal part2'.getType "Terminator" "Modified Part Type"
            
            let! events = db.GetEvents()
            let partEvents = events |> Array.filter (fun e -> e.target = PartEvent(part1.id))
            Expect.equal partEvents.Length 2 "Part Event Count - After modification"


            let modifiedPartEvents = partEvents |> Array.filter (fun e -> e.operation = EventOperation.Modify)
            Expect.hasLength modifiedPartEvents 1 "Number of modify part events."
            let modifiedPartEvent = modifiedPartEvents.[0]
            let changes = Decode.Auto.unsafeFromString<(string*string)list>(modifiedPartEvent.change)
            Expect.hasLength changes 2 "Modified Part - Changes" 

            let (mremoveType,rtype) = changes |> List.find (fun (c,cstring) -> c = Part.removeType)
            let (maddType,atype) = changes |> List.find (fun (c,cstring) -> c = Part.addType)
            
            Expect.equal rtype ("UserDefined") "Modify part remove type"
            Expect.equal atype ("Terminator") "Modify part add type"
            

        }

        testAsync "Reagent Events" {
            let db = db1
            let reagentProperties = {
                name = "Reagent1234"
                barcode = None
                notes = "Got this Reagent from DB."
                deprecated = false
            }
            let reagent1 = DNA{
                    id = DNAId.Create()
                    properties = reagentProperties
                    sequence = "TAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTGAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTGGGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACATCAGCCACAACGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATAATACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATAACGCTCTGTAGGTCCAGTTTGACCCTCCACTTGGTCAAGTGATATCCTGGTAAGGTAAGCTCGTACCGTGATTCATGCGGCAGGGGTAAGACCATTAGAAGTAGGGATAGTCCCAAACCTCACTTACCACTGTTAGCCGAAGTTGCACGGGGTGCCCACCGTGGACTCCTCCCCGGGTGTCGCTCCTTCATCTGACAATATGCAGCCGCTACCACCATCGATTAATACAACGCATTAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAACTCGGTACCAAATTCCAGAAAAGAGGCCTCCCGAAAGGGGGGCCTTTTTTCGTTTTGGTCCTTTCCAATAAGGGGTCCTTATCTGAAGGATGAGTGTCAGCCAGTGTAACCCGATGAGGAACCCAGAAGCCGAACTGGGCCAGACAACCCGGCGCTAACGCACTCAAAGCCGGGACGCGACGCGACATAACGGGGGTAGCACCAGAAGTCTATAGCACGTGCATCCCAACGTGGCGTGCGTACACCTTAATCACCGCTTCATGCTAAGGTCCTGGCTGCATGCTATGTTGATAGGTTGAGAATTCTGTACACTCGAGGGTCTCACCCCAAGGGCGACACCCCCTAATTAGCCCGGGCGAAAGGCCCAGTCTTTCGACTGAGCCTTTCGTTTTATTTGATGCCTGGCAGTTCCCTACTCTCGCATGGGGAGTCCCCACACTACCATCGGCGCTACGGCGTTTCACTTCTGAGTTCGGCATGGGGTCAGGTGGGACCACCGCGCTACTGCCGCCAGGCAAACAAGGGGTGTTATGAGCCATATTCAGGTATAAATGGGCTCGCGATAATGTTCAGAATTGGTTAATTGGTTGTAACACTGACCCCTAATGGAAGTACTAGTAGCGGCCGCTGCAGTCCGGCAAAAAAACGGGCAAGGTGTCACCACCCTGCCCTTTTTCTTTAAAACCGAAAAGATTACTTCGCGTTATGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATCTCGAGTCCCGTCAAGTCAGCGTAATGCTCTGCCAGTGTTACAACCAATTAACCAATTCTGATTAGAAAAACTCATCGAGCATCAAATGAAACTGCAATTTATTCATATCAGGATTATCAATACCATATTTTTGAAAAAGCCGTTTCTGTAATGAAGGAGAAAACTCACCGAGGCAGTTCCATAGGATGGCAAGATCCTGGTATCGGTCTGCGATTCCGACTCGTCCAACATCAATACAACCTATTAATTTCCCCTCGTCAAAAATAAGGTTATCAAGTGAGAAATCACCATGAGTGACGACTGAATCCGGTGAGAATGGCAAAAGCTTATGCATTTCTTTCCAGACTTGTTCAACAGGCCAGCCATTACGCTCGTCATCAAAATCACTCGCATCAACCAAACCGTTATTCATTCGTGATTGCGCCTGAGCGAGACGAAATACGCGATCGCTGTTAAAAGGACAATTACAAACAGGAATCGAATGCAACCGGCGCAGGAACACTGCCAGCGCATCAACAATATTTTCACCTGAATCAGGATATTCTTCTAATACCTGGAATGCTGTTTTCCCGGGGATCGCAGTGGTGAGTAACCATGCATCATCAGGAGTACGGATAAAATGCTTGATGGTCGGAAGAGGCATAAATTCCGTCAGCCAGTTTAGTCTGACCATCTCATCTGTAACATCATTGGCAACGCTACCTTTGCCATGTTTCAGAAACAACTCTGGCGCATCGGGCTTCCCATACAATCGATAGATTGTCGCACCTGATTGCCCGACATTATCGCGAGCCCATTTATACCCATATAAATCAGCATCCATGTTGGAATTTAATCGCGGCCTCGAGCAAGACGTTTCCCGTTGAATATGGCTCATAACACCCCTTGTATTACTGTTTATGTAAGCAGACAGTTTTATTGTTCATGATGATATATTTTTATCTTGTGCAATGTAACATCAGAGATTTTGAGACACAACGTGGCTTTGTTGAATAAATCGAACTTTTGCTGAGTTGAAGGATCAGATCACGCATCTTCCCGACAACGCAGACCGTTCCGTGGCAAAGCAAAAGTTCAAAATCACCAACTGGTCCACCTACAACAAAGCTCTCATCAACCGTGGCTCCCTCACTTTCTGGCTGGATGATGGGGCGATTCAGGCCTGGTATGAGTCAGCAACACCTTCTTCACGAGGCAGACCTCAGCGCTAGCGGAGTGTATACTGGCTTACTATGTTGGCACTGATGAGGGTGTCAGTGAAGTGCTTCATGTGGCAGGAGAAAAAAGGCTGCACCGGTGCGTCAGCAGAATATGTGATACAGGATATATTCCGCTTCCTCGCTCACTGACTCGCTACGCTCGGTCGTTCGACTGCGGCGAGCGGAAATGGCTTACGAACGGGGCGGAGATTTCCTGGAAGATGCCAGGAAGATACTTAACAGGGAAGTGAGAGGGCCGCGGCAAAGCCGTTTTTCCATAGGCTCCGCCCCCCTGACAAGCATCACGAAATCTGACGCTCAAATCAGTGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCTGGCGGCTCCCTCGTGCGCTCTCCTGTTCCTGCCTTTCGGTTTACCGGTGTCATTCCGCTGTTATGGCCGCGTTTGTCTCATTCCACGCCTGACACTCAGTTCCGGGTAGGCAGTTCGCTCCAAGCTGGACTGTATGCACGAACCCCCCGTTCAGTCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGAAAGACATGCAAAAGCACCACTGGCAGCAGCCACTGGTAATTGATTTAGAGGAGTTAGTCTTGAAGTCATGCGCCGGTTAAGGCTAAACTGAAAGGACAAGTTTTGGTGACTGCGCTCCTCCAAGCCAGTTACCTCGGTTCAAAGAGTTGGTAGCTCAGAGAACCTTCGAAAAACCGCCCTGCAAGGCGGTTTTTTCGTTTTCAGAGCAAGAGATTACGCGCAGACCAAAACGATCTCAAGAAGATCATCTTATTAAGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTCCATGGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCTTCTAGAGAATTTTGTGTCGCCCTTGAA"
                    concentration = None
                    Type = DNAType.SourceLinearDNA
                } 

           

            let! _ = db.SaveReagent reagent1
            let! reagent1opt = db.TryGetReagent reagent1.id
            Expect.isSome reagent1opt "Reagent has a value" 
            let reagent1' = match reagent1opt with | Some(r) -> r | None -> failwith "Reagent Option really can't be none at this point."
            let! reagents = db.GetReagents()
            Expect.equal reagents.Length 1 "Reagent Count"
            Expect.equal reagent1.name reagent1'.name "Reagent Name"

            let dnaReagent1 = match reagent1 with | DNA(d) -> d | _ -> failwith "Unknown Reagent type"
            let dnaReagent1' = match reagent1' with | DNA(d) -> d | _ -> failwith "Unknown Reagent type"
            
            Expect.equal dnaReagent1.sequence dnaReagent1'.sequence "Reagent sequence"
            let! events = db.GetEvents()
            let reagentEvents = events |> Array.filter (fun e -> e.target = ReagentEvent(reagent1.id))
            Expect.hasLength reagentEvents 1 "Reagent event exists"
            let reagentEvent = reagentEvents.[0]
            Expect.equal reagentEvent.operation (EventOperation.Add) "Reagent Event Type"
            let change = Decode.Auto.unsafeFromString<(string*string)list>(reagentEvent.change)
            Expect.hasLength change 6 "Add Reagent Event Change string"
            
            let (addname) = change |> List.filter (fun (c,cstring) -> c = Reagent.addName)
            Expect.hasLength addname 1 "Reagent Event - 1 Add Name"
            let (raddname,rname) = addname.[0]
            Expect.equal rname reagent1'.name "Reagent name matches change string"
            
            let (addSequence) = change |> List.filter (fun (c,cstring) -> c = Reagent.addSequence)
            Expect.hasLength addSequence 1 "Reagent Event - 1 Add Sequence"
            let (raddsequence,rsequence) = addSequence.[0]
            Expect.equal rsequence "TAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTGAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTGGGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACATCAGCCACAACGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATAATACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATAACGCTCTGTAGGTCCAGTTTGACCCTCCACTTGGTCAAGTGATATCCTGGTAAGGTAAGCTCGTACCGTGATTCATGCGGCAGGGGTAAGACCATTAGAAGTAGGGATAGTCCCAAACCTCACTTACCACTGTTAGCCGAAGTTGCACGGGGTGCCCACCGTGGACTCCTCCCCGGGTGTCGCTCCTTCATCTGACAATATGCAGCCGCTACCACCATCGATTAATACAACGCATTAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAACTCGGTACCAAATTCCAGAAAAGAGGCCTCCCGAAAGGGGGGCCTTTTTTCGTTTTGGTCCTTTCCAATAAGGGGTCCTTATCTGAAGGATGAGTGTCAGCCAGTGTAACCCGATGAGGAACCCAGAAGCCGAACTGGGCCAGACAACCCGGCGCTAACGCACTCAAAGCCGGGACGCGACGCGACATAACGGGGGTAGCACCAGAAGTCTATAGCACGTGCATCCCAACGTGGCGTGCGTACACCTTAATCACCGCTTCATGCTAAGGTCCTGGCTGCATGCTATGTTGATAGGTTGAGAATTCTGTACACTCGAGGGTCTCACCCCAAGGGCGACACCCCCTAATTAGCCCGGGCGAAAGGCCCAGTCTTTCGACTGAGCCTTTCGTTTTATTTGATGCCTGGCAGTTCCCTACTCTCGCATGGGGAGTCCCCACACTACCATCGGCGCTACGGCGTTTCACTTCTGAGTTCGGCATGGGGTCAGGTGGGACCACCGCGCTACTGCCGCCAGGCAAACAAGGGGTGTTATGAGCCATATTCAGGTATAAATGGGCTCGCGATAATGTTCAGAATTGGTTAATTGGTTGTAACACTGACCCCTAATGGAAGTACTAGTAGCGGCCGCTGCAGTCCGGCAAAAAAACGGGCAAGGTGTCACCACCCTGCCCTTTTTCTTTAAAACCGAAAAGATTACTTCGCGTTATGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATCTCGAGTCCCGTCAAGTCAGCGTAATGCTCTGCCAGTGTTACAACCAATTAACCAATTCTGATTAGAAAAACTCATCGAGCATCAAATGAAACTGCAATTTATTCATATCAGGATTATCAATACCATATTTTTGAAAAAGCCGTTTCTGTAATGAAGGAGAAAACTCACCGAGGCAGTTCCATAGGATGGCAAGATCCTGGTATCGGTCTGCGATTCCGACTCGTCCAACATCAATACAACCTATTAATTTCCCCTCGTCAAAAATAAGGTTATCAAGTGAGAAATCACCATGAGTGACGACTGAATCCGGTGAGAATGGCAAAAGCTTATGCATTTCTTTCCAGACTTGTTCAACAGGCCAGCCATTACGCTCGTCATCAAAATCACTCGCATCAACCAAACCGTTATTCATTCGTGATTGCGCCTGAGCGAGACGAAATACGCGATCGCTGTTAAAAGGACAATTACAAACAGGAATCGAATGCAACCGGCGCAGGAACACTGCCAGCGCATCAACAATATTTTCACCTGAATCAGGATATTCTTCTAATACCTGGAATGCTGTTTTCCCGGGGATCGCAGTGGTGAGTAACCATGCATCATCAGGAGTACGGATAAAATGCTTGATGGTCGGAAGAGGCATAAATTCCGTCAGCCAGTTTAGTCTGACCATCTCATCTGTAACATCATTGGCAACGCTACCTTTGCCATGTTTCAGAAACAACTCTGGCGCATCGGGCTTCCCATACAATCGATAGATTGTCGCACCTGATTGCCCGACATTATCGCGAGCCCATTTATACCCATATAAATCAGCATCCATGTTGGAATTTAATCGCGGCCTCGAGCAAGACGTTTCCCGTTGAATATGGCTCATAACACCCCTTGTATTACTGTTTATGTAAGCAGACAGTTTTATTGTTCATGATGATATATTTTTATCTTGTGCAATGTAACATCAGAGATTTTGAGACACAACGTGGCTTTGTTGAATAAATCGAACTTTTGCTGAGTTGAAGGATCAGATCACGCATCTTCCCGACAACGCAGACCGTTCCGTGGCAAAGCAAAAGTTCAAAATCACCAACTGGTCCACCTACAACAAAGCTCTCATCAACCGTGGCTCCCTCACTTTCTGGCTGGATGATGGGGCGATTCAGGCCTGGTATGAGTCAGCAACACCTTCTTCACGAGGCAGACCTCAGCGCTAGCGGAGTGTATACTGGCTTACTATGTTGGCACTGATGAGGGTGTCAGTGAAGTGCTTCATGTGGCAGGAGAAAAAAGGCTGCACCGGTGCGTCAGCAGAATATGTGATACAGGATATATTCCGCTTCCTCGCTCACTGACTCGCTACGCTCGGTCGTTCGACTGCGGCGAGCGGAAATGGCTTACGAACGGGGCGGAGATTTCCTGGAAGATGCCAGGAAGATACTTAACAGGGAAGTGAGAGGGCCGCGGCAAAGCCGTTTTTCCATAGGCTCCGCCCCCCTGACAAGCATCACGAAATCTGACGCTCAAATCAGTGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCTGGCGGCTCCCTCGTGCGCTCTCCTGTTCCTGCCTTTCGGTTTACCGGTGTCATTCCGCTGTTATGGCCGCGTTTGTCTCATTCCACGCCTGACACTCAGTTCCGGGTAGGCAGTTCGCTCCAAGCTGGACTGTATGCACGAACCCCCCGTTCAGTCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGAAAGACATGCAAAAGCACCACTGGCAGCAGCCACTGGTAATTGATTTAGAGGAGTTAGTCTTGAAGTCATGCGCCGGTTAAGGCTAAACTGAAAGGACAAGTTTTGGTGACTGCGCTCCTCCAAGCCAGTTACCTCGGTTCAAAGAGTTGGTAGCTCAGAGAACCTTCGAAAAACCGCCCTGCAAGGCGGTTTTTTCGTTTTCAGAGCAAGAGATTACGCGCAGACCAAAACGATCTCAAGAAGATCATCTTATTAAGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTCCATGGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCTTCTAGAGAATTTTGTGTCGCCCTTGAA" "Reagent Sequence"  
            
            //First Reagent change
            let reagent2 = 
                let dnareagent = match reagent1' with | DNA d -> d | _ -> failwithf "This shouldn't be happening."
                let sequence = "ATGC"
                let updatedReagentProperties = {reagent1'.getProperties with name = "Reagent2"}
                DNA{dnareagent with properties = updatedReagentProperties;sequence = sequence}
            let! _ = db.SaveReagent reagent2
            let! reagent2opt = db.TryGetReagent reagent1.id
            let reagent2' = match reagent2opt with | Some(r) -> r | None -> failwith "Reagent Option really can't be none at this point."            
            let dnareagent2' = match reagent2' with | DNA d -> d | _ -> failwithf "This shouldn't be happening."

            Expect.equal reagent2'.name "Reagent2" "Modified Reagent Name"
            Expect.equal dnareagent2'.sequence ("ATGC") "Modified Reagent Sequence"

            let! events = db.GetEvents()
            let allreagentevents = events |> Array.filter (fun e -> e.target = ReagentEvent(reagent1.id))
            Expect.hasLength allreagentevents 2 "All Reagent Events"
            let modifyReagentEvents = allreagentevents |> Array.filter (fun e -> e.operation = EventOperation.Modify)
            Expect.hasLength modifyReagentEvents 1 "Modify Reagent Events Length"
            let modifyReagentEvent = modifyReagentEvents.[0]
            let modifiedchange = Decode.Auto.unsafeFromString<(string*string)list>(modifyReagentEvent.change)

            Expect.hasLength modifiedchange 4 "Modify Reagent changes"

            let (removeSequence,rseq) = modifiedchange |> List.find (fun (c,cstring) -> c = Reagent.removeSequence)
            let (addSequence,aseq) = modifiedchange |> List.find (fun (c,cstring) -> c = Reagent.addSequence)
            let (removeName,rname) = modifiedchange |> List.find (fun (c,cstring) -> c = Reagent.removeName)
            let (addName,aname) = modifiedchange |> List.find (fun (c,cstring) -> c = Reagent.addName)
            
            Expect.equal rseq "TAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTGAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTGGGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACATCAGCCACAACGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATAATACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATAACGCTCTGTAGGTCCAGTTTGACCCTCCACTTGGTCAAGTGATATCCTGGTAAGGTAAGCTCGTACCGTGATTCATGCGGCAGGGGTAAGACCATTAGAAGTAGGGATAGTCCCAAACCTCACTTACCACTGTTAGCCGAAGTTGCACGGGGTGCCCACCGTGGACTCCTCCCCGGGTGTCGCTCCTTCATCTGACAATATGCAGCCGCTACCACCATCGATTAATACAACGCATTAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAACTCGGTACCAAATTCCAGAAAAGAGGCCTCCCGAAAGGGGGGCCTTTTTTCGTTTTGGTCCTTTCCAATAAGGGGTCCTTATCTGAAGGATGAGTGTCAGCCAGTGTAACCCGATGAGGAACCCAGAAGCCGAACTGGGCCAGACAACCCGGCGCTAACGCACTCAAAGCCGGGACGCGACGCGACATAACGGGGGTAGCACCAGAAGTCTATAGCACGTGCATCCCAACGTGGCGTGCGTACACCTTAATCACCGCTTCATGCTAAGGTCCTGGCTGCATGCTATGTTGATAGGTTGAGAATTCTGTACACTCGAGGGTCTCACCCCAAGGGCGACACCCCCTAATTAGCCCGGGCGAAAGGCCCAGTCTTTCGACTGAGCCTTTCGTTTTATTTGATGCCTGGCAGTTCCCTACTCTCGCATGGGGAGTCCCCACACTACCATCGGCGCTACGGCGTTTCACTTCTGAGTTCGGCATGGGGTCAGGTGGGACCACCGCGCTACTGCCGCCAGGCAAACAAGGGGTGTTATGAGCCATATTCAGGTATAAATGGGCTCGCGATAATGTTCAGAATTGGTTAATTGGTTGTAACACTGACCCCTAATGGAAGTACTAGTAGCGGCCGCTGCAGTCCGGCAAAAAAACGGGCAAGGTGTCACCACCCTGCCCTTTTTCTTTAAAACCGAAAAGATTACTTCGCGTTATGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATCTCGAGTCCCGTCAAGTCAGCGTAATGCTCTGCCAGTGTTACAACCAATTAACCAATTCTGATTAGAAAAACTCATCGAGCATCAAATGAAACTGCAATTTATTCATATCAGGATTATCAATACCATATTTTTGAAAAAGCCGTTTCTGTAATGAAGGAGAAAACTCACCGAGGCAGTTCCATAGGATGGCAAGATCCTGGTATCGGTCTGCGATTCCGACTCGTCCAACATCAATACAACCTATTAATTTCCCCTCGTCAAAAATAAGGTTATCAAGTGAGAAATCACCATGAGTGACGACTGAATCCGGTGAGAATGGCAAAAGCTTATGCATTTCTTTCCAGACTTGTTCAACAGGCCAGCCATTACGCTCGTCATCAAAATCACTCGCATCAACCAAACCGTTATTCATTCGTGATTGCGCCTGAGCGAGACGAAATACGCGATCGCTGTTAAAAGGACAATTACAAACAGGAATCGAATGCAACCGGCGCAGGAACACTGCCAGCGCATCAACAATATTTTCACCTGAATCAGGATATTCTTCTAATACCTGGAATGCTGTTTTCCCGGGGATCGCAGTGGTGAGTAACCATGCATCATCAGGAGTACGGATAAAATGCTTGATGGTCGGAAGAGGCATAAATTCCGTCAGCCAGTTTAGTCTGACCATCTCATCTGTAACATCATTGGCAACGCTACCTTTGCCATGTTTCAGAAACAACTCTGGCGCATCGGGCTTCCCATACAATCGATAGATTGTCGCACCTGATTGCCCGACATTATCGCGAGCCCATTTATACCCATATAAATCAGCATCCATGTTGGAATTTAATCGCGGCCTCGAGCAAGACGTTTCCCGTTGAATATGGCTCATAACACCCCTTGTATTACTGTTTATGTAAGCAGACAGTTTTATTGTTCATGATGATATATTTTTATCTTGTGCAATGTAACATCAGAGATTTTGAGACACAACGTGGCTTTGTTGAATAAATCGAACTTTTGCTGAGTTGAAGGATCAGATCACGCATCTTCCCGACAACGCAGACCGTTCCGTGGCAAAGCAAAAGTTCAAAATCACCAACTGGTCCACCTACAACAAAGCTCTCATCAACCGTGGCTCCCTCACTTTCTGGCTGGATGATGGGGCGATTCAGGCCTGGTATGAGTCAGCAACACCTTCTTCACGAGGCAGACCTCAGCGCTAGCGGAGTGTATACTGGCTTACTATGTTGGCACTGATGAGGGTGTCAGTGAAGTGCTTCATGTGGCAGGAGAAAAAAGGCTGCACCGGTGCGTCAGCAGAATATGTGATACAGGATATATTCCGCTTCCTCGCTCACTGACTCGCTACGCTCGGTCGTTCGACTGCGGCGAGCGGAAATGGCTTACGAACGGGGCGGAGATTTCCTGGAAGATGCCAGGAAGATACTTAACAGGGAAGTGAGAGGGCCGCGGCAAAGCCGTTTTTCCATAGGCTCCGCCCCCCTGACAAGCATCACGAAATCTGACGCTCAAATCAGTGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCTGGCGGCTCCCTCGTGCGCTCTCCTGTTCCTGCCTTTCGGTTTACCGGTGTCATTCCGCTGTTATGGCCGCGTTTGTCTCATTCCACGCCTGACACTCAGTTCCGGGTAGGCAGTTCGCTCCAAGCTGGACTGTATGCACGAACCCCCCGTTCAGTCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGAAAGACATGCAAAAGCACCACTGGCAGCAGCCACTGGTAATTGATTTAGAGGAGTTAGTCTTGAAGTCATGCGCCGGTTAAGGCTAAACTGAAAGGACAAGTTTTGGTGACTGCGCTCCTCCAAGCCAGTTACCTCGGTTCAAAGAGTTGGTAGCTCAGAGAACCTTCGAAAAACCGCCCTGCAAGGCGGTTTTTTCGTTTTCAGAGCAAGAGATTACGCGCAGACCAAAACGATCTCAAGAAGATCATCTTATTAAGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTCCATGGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCTTCTAGAGAATTTTGTGTCGCCCTTGAA" "Modify Reagent Remove Sequence"
            Expect.equal aseq "ATGC" "Modify Reagent add sequence"
            Expect.equal rname "Reagent1234" "Modify Reagent remove name"
            Expect.equal aname "Reagent2" "Modify Reagent add name"

            //Second Reagent change
            
            
            let reagentfiletext = "Hello reagent!"
            let reagentfileref = { 
                fileId = FileId.Create()
                fileName = "ReagentDesc.txt"
                Type = FileType.MiscFile
            }
            
            let! _ = db.UploadFile(reagentfileref.fileId,reagentfiletext)
            let! _ = db.AddReagentFile(reagent2.id,reagentfileref)
            let! events = db.GetEvents()
            let allreagentfileevents = events |> Array.filter (fun e -> e.target = ReagentFileEvent(reagent1.id))
            Expect.hasLength allreagentfileevents 1 "All Reagent Events"
            
            let! reagentfiles = db.GetReagentFiles reagent1.id
            Expect.hasLength reagentfiles 1 "Reagent Files should be = 1"

        }
        
        testAsync "Experiment, ExperimentOperation, and Signals" {
            let db = db1
            let experimentfiletext = "a,b,c"
            let experimentfileref = { 
                fileId = FileId.Create()
                fileName = "somelayout.csv"
                Type = FileType.MiscFile
            }
            let signal1 = {
                id = SignalId.Create()
                settings = PlateReaderFluorescence({emissionFilter = PlateReaderFilter.PlateFilter_430_10; excitationFilter = PlateReaderFilter.PlateFilter_530; gain = 3.0})
                units = None
            }
            
            let operationStarted:ExperimentOperation = {
                id = ExperimentOperationId.Create()
                timestamp = System.DateTime.Now
                Type = ExperimentOperationType.ExperimentStarted
            }

            let operationFinished:ExperimentOperation = {
                id = ExperimentOperationId.Create()
                timestamp = System.DateTime.Now
                Type = ExperimentOperationType.ExperimentFinished
            }

            let experiment1 = {
                id = ExperimentId.Create()
                name = "Test Experiment"
                notes = "I lost my notes"
                Type = ExperimentType.TestExperiment
                deprecated = false
            }


            let! _ = db.SaveExperiment experiment1
            let! _ = db.UploadFile(experimentfileref.fileId,experimentfiletext)
            let! _ = db.AddExperimentFile(experiment1.id,experimentfileref)
            let! _ = db.SaveExperimentSignals(experiment1.id, [|signal1|])
            let! _ = db.SaveExperimentOperation (experiment1.id, operationStarted)
            let! _ = db.SaveExperimentOperation (experiment1.id, operationFinished)
            
            let! experiment1opt = db.TryGetExperiment experiment1.id
            Expect.isSome experiment1opt "Reagent has a value" 
            let experiment1' = match experiment1opt with | Some(e) -> e | None -> failwith "Experiment Option really can't be none at this point."
            
            Expect.isFalse experiment1'.deprecated ""
            Expect.equal experiment1.id experiment1'.id ""
            Expect.equal experiment1.deprecated experiment1'.deprecated ""
            

            let! exptSignals = db.GetExperimentSignals experiment1.id
            Expect.hasLength exptSignals 1 ""
            let exptSignal = exptSignals.[0]
            Expect.equal signal1.id exptSignal.id ""
            Expect.equal signal1.settings exptSignal.settings ""
        
            let! exptOps = db.GetExperimentOperations experiment1.id
            Expect.hasLength exptOps 2 ""
            
            let experiment2 = {experiment1' with notes = "I found my notes"}            
            let! _ = db.SaveExperiment experiment2

            let! events = db.GetEvents()
            let experimentevents = events |> Array.filter (fun e -> match e.target with | ExperimentEvent _ -> true | _ -> false)
            Expect.hasLength experimentevents 2 ""
            let exptEventsAdd = experimentevents |> Array.filter (fun e -> match e.operation with | EventOperation.Add -> true | _ -> false)
            let exptEventsModify = experimentevents |> Array.filter (fun e -> match e.operation with | EventOperation.Modify -> true | _ -> false)
            Expect.hasLength exptEventsAdd 1 ""
            Expect.hasLength exptEventsModify 1 ""
            
            let exptEventAdd = exptEventsAdd.[0]
            let exptEventModify = exptEventsModify.[0]
            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(exptEventAdd.change)) 4 ""
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(exptEventModify.change)) 2 ""     
            
            let! experiment2opt = db.TryGetExperiment experiment1.id
            Expect.isSome experiment2opt "Reagent has a value" 
            let experiment2' = match experiment2opt with | Some(e) -> e | None -> failwith "Experiment Option really can't be none at this point."
            Expect.equal experiment2'.notes "I found my notes" ""

            let exptopEvents = events |> Array.filter (fun e -> match e.target with | ExperimentOperationEvent _ -> true | _ -> false)
            Expect.hasLength exptopEvents 2 ""
            
            let exptFileEvents = events |> Array.filter (fun e -> match e.target with | ExperimentFileEvent _ -> true | _ -> false)
            Expect.hasLength exptFileEvents 1 ""
        
        }

        testAsync "Samples and Cells Events in a separate Memory instance" {
            let db = db2
            let chemreagent = 
                let rprops = {
                    ReagentProperties.name = "Chemical1"
                    ReagentProperties.barcode = None
                    ReagentProperties.deprecated = false
                    ReagentProperties.notes = "This is a chemical."
                }
                Chemical({id = ChemicalId.Create(); properties = rprops;Type = ChemicalType.SmallMolecule})
            

            let dnaReagent1 = 
                let rprops = {
                    ReagentProperties.name = "Some DNA"
                    ReagentProperties.barcode = None
                    ReagentProperties.deprecated = false
                    ReagentProperties.notes = "This is DNA."
                }
                DNA({id = DNAId.Create(); properties = rprops;Type=DNAType.SourceLinearDNA;sequence="ATTGGTTATTTAGGGCCGAGCA";concentration=None})
            
            let dnaReagent2 = 
                let rprops = {
                    ReagentProperties.name = "Some DNA"
                    ReagentProperties.barcode = None
                    ReagentProperties.deprecated = false
                    ReagentProperties.notes = "This is DNA."
                }
                DNA({id = DNAId.Create(); properties = rprops;Type=DNAType.SourceLinearDNA;sequence="ATTGGTTATGGGGGGGGGGGGGTTAGGGCCGAGCA";concentration=None})
            
            

            let expt1 = {
                id = ExperimentId.Create()
                name = "Experiment for Samples and Cells"
                notes = "No notes"
                Type = ExperimentType.TestExperiment
                deprecated = false
            }

            let sample1 = {
                id = SampleId.Create()
                experimentId = expt1.id
                meta = PlateReaderMeta({virtualWell = {col = 1;row = 2}; physicalPlateName = "Plate0" |> Some;physicalWell = {col = 2;row = 3} |> Some})
                deprecated = false
            }

            let cell1 = 
                let cellProps = 
                    {
                        id = CellId.Create()
                        genotype = ""
                        name = "OG Coli"
                        notes = "This is the original E.Cloni"
                        barcode = "FD1001" |> Barcode |> Some
                        deprecated = false
                    }
                {Type = ProkaryoteType.Bacteria;properties = cellProps} |> Prokaryote

            let cell2 = 
                let cellProps = 
                    {
                        id = CellId.Create()
                        genotype = ""
                        name = "Strain Cell2"
                        notes = "This was derived from OG Coli"
                        barcode = "FD1002" |> Barcode |> Some
                        deprecated = false
                    }
                {Type = ProkaryoteType.Bacteria;properties = cellProps} |> Prokaryote
            
            let cell2entity1 = {
                entity = dnaReagent1.id
                cellId = cell2.id
                compartment = CellCompartment.Plasmid
                }
            
            let cell2entity2 = {
                entity = dnaReagent2.id
                cellId = cell2.id
                compartment = CellCompartment.Plasmid
                }

            let derivedFrom = CellLineage(cell1.id,cell2.id)
            
            let condition:Condition = {
                reagentId = chemreagent.id
                sampleId = sample1.id
                concentration = NM 10.0
                time = (Min 30.0) |> Some
                }
            
            let! _ = db.SaveReagent chemreagent
            let! _ = db.SaveReagent dnaReagent1
            let! _ = db.SaveReagent dnaReagent2
            let! _ = db.SaveExperiment expt1
            let! _ = db.SaveSamples [|sample1|]
            
            let! _ = db.SaveCell cell1
            let! _ = db.SaveCell cell2
            
            let! _ = db.SaveDerivedFrom derivedFrom
            let! _ = db.SaveCellEntity cell2entity1
            let! _ = db.SaveCellEntity cell2entity2
            

            let sampleDevice = {
                sampleId = sample1.id
                cellId = cell2.id
                cellPreSeeding = None
                cellDensity = None
            }

            let! _ = db.SaveSampleDevices(sample1.id,[|sampleDevice|])
            let! _ = db.SaveSampleConditions(sample1.id,[|condition|])
            

            let! events = db.GetEvents()
            let reagentEvents = events |> Array.filter (fun e -> match e.target with | ReagentEvent _ -> true | _ -> false)
            let sampleEvents = events |> Array.filter (fun e -> e.target = SampleEvent(sample1.id))
            let sampleDataEvents = events |> Array.filter (fun e -> e.target = SampleDataEvent(sample1.id))
            let sampleDeviceEvents = events |> Array.filter (fun e -> e.target = SampleDeviceEvent(sample1.id))
            let sampleConditionsEvents = events |> Array.filter (fun e -> e.target = SampleConditionEvent(sample1.id))
            let cellEvents = events |> Array.filter (fun e -> match e.target with | CellEvent _ -> true | _ -> false)
            let smChemEvents = reagentEvents |> Array.filter (fun e -> match e.target with | ReagentEvent (rid) -> (match rid with | ChemicalReagentId _ -> true | _ -> false) | _ -> false)
            let dfEvents = events |> Array.filter (fun e -> match e.target with | DerivedFromEvent -> true | _ -> false)
            
            Expect.hasLength reagentEvents 3 ""
            Expect.hasLength smChemEvents 1 ""
            
            Expect.hasLength sampleEvents 1 "There should just be 1 sample event."
            Expect.hasLength sampleDataEvents 0 "There should be 0 sample data event."
            Expect.hasLength sampleDeviceEvents 1 "There should be 1 sample data event."
            Expect.hasLength sampleConditionsEvents 1 "There should be 0 sample data event."
            
            Expect.hasLength cellEvents 2 ""
            Expect.hasLength dfEvents 1 ""
            
            let motherCellEvents = cellEvents |> Array.filter (fun e -> e.target = CellEvent(cell1.id))
            let daughterCellEvents = cellEvents |> Array.filter (fun e -> e.target = CellEvent(cell2.id))
            
            Expect.hasLength motherCellEvents 1 ""
            Expect.hasLength daughterCellEvents 1 ""
            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(smChemEvents.[0].change)) 5 "Chemical (Small Molecule) should have just 5 changes."            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(sampleEvents.[0].change)) 8 "Sample should have just 8 changes"
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(sampleDeviceEvents.[0].change)) 1 "Sample Device should have just 1 change"
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(sampleConditionsEvents.[0].change)) 5 "Sample conditions should have just 5 changes"
            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(motherCellEvents.[0].change)) 7 ""
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(daughterCellEvents.[0].change)) 7 ""
            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(dfEvents.[0].change)) 3 ""
            
        }

        testAsync "Tags & Deprecate" {
            
            let db = db3
            let dnaReagent = DNA{
                    id = DNAId.Create()
                    properties = {
                        name = "Reagent1"
                        barcode = None
                        notes = "Got this Reagent from DB."
                        deprecated = false
                    }
                    sequence = "ATGC"
                    concentration = None
                    Type = DNAType.SourceLinearDNA
                }
            let chemicalReagent = Chemical{
                id = ChemicalId.Create()
                properties = {
                    name = "Media"
                    barcode = None
                    notes = "This is a media"
                    deprecated = false
                }
                Type = ChemicalType.Media
            } 

            let expt1 = {
                id = ExperimentId.Create()
                name = "Experiment for Tags and deprecation"
                notes = "No notes"
                Type = ExperimentType.BuildExperiment
                deprecated = false
            }

            let! _ = db.SaveReagent dnaReagent
            let! _ = db.SaveReagent chemicalReagent
            
            let! _ = db.SaveExperiment expt1
            
            let! _ = db.AddReagentTag(dnaReagent.id,"Project1" |> Tag)
            let! _ = db.AddReagentTag(dnaReagent.id,"DNAReagents" |> Tag)
            
            let! _ = db.AddReagentTag(chemicalReagent.id,"DNAReagents" |> Tag)
            let! _ = db.AddReagentTag(chemicalReagent.id,"Project1" |> Tag)
            
            let! _ = db.AddExperimentTag(expt1.id,"Project1" |> Tag)
            
            let! _ = db.RemoveReagentTag(chemicalReagent.id,"DNAReagents" |> Tag)
            let! _ = db.AddReagentTag(chemicalReagent.id,"ChemicalReagents" |> Tag)
            let! _ = db.AddReagentTag(chemicalReagent.id,"ChemicalReagents" |> Tag)
            
            let depExpt1 = {expt1 with deprecated = true}
            let! _ = db.SaveExperiment depExpt1
             
            let! events = db.GetEvents()

            Expect.hasLength events 11 ""

            let reagentTagEvents = events |> Array.filter (fun e -> match  e.target with | ReagentTagEvent _ -> true | _ -> false)
            Expect.hasLength reagentTagEvents 6 ""
            
            let exptTagEvents = events |> Array.filter (fun e -> match  e.target with | ExperimentTagEvent _ -> true | _ -> false)
            Expect.hasLength exptTagEvents 1 ""

            let! dnaTags = db.GetReagentTags(dnaReagent.id)
            Expect.hasLength dnaTags 2 ""

            let! chemicalTags = db.GetReagentTags(chemicalReagent.id)
            Expect.hasLength chemicalTags 2 ""

            let! exptTags = db.GetExperimentTags(expt1.id)
            Expect.hasLength exptTags 1 ""


            Expect.contains dnaTags ("DNAReagents" |> Tag) ""
            Expect.contains dnaTags ("Project1" |> Tag) ""
            
            Expect.contains chemicalTags ("ChemicalReagents" |> Tag) ""
            Expect.contains chemicalTags ("Project1" |> Tag) ""
            
            Expect.contains exptTags ("Project1" |> Tag) ""
            
            let! experiment1opt = db.TryGetExperiment expt1.id
            Expect.isSome experiment1opt "Reagent has a value" 
            let expt' = match experiment1opt with | Some(e) -> e | None -> failwith "Experiment Option really can't be none at this point."
            
            Expect.isTrue expt'.deprecated ""

            let depExptEvents = events |> Array.filter (fun e -> match e.target with | ExperimentEvent _ -> (match e.operation with | EventOperation.Modify _ -> true | _ -> false ) | _ -> false)
            Expect.hasLength depExptEvents 1 ""
            
            Expect.hasLength (Decode.Auto.unsafeFromString<(string*string)list>(depExptEvents.[0].change)) 2 ""
            

        }
            


        testAsync "Signals" {
            
            let db = BCKG.API.Instance(BCKG.API.MemoryInstance,"TestUser")
           
            let expt1 = {
                id = ExperimentId.Create()
                name = "Experiment for Tags and deprecation"
                notes = "No notes"
                Type = ExperimentType.BuildExperiment
                deprecated = false
            }

            let! _ = db.SaveExperiment expt1

            let signal1 = 
               { id       = System.Guid.NewGuid() |> SignalId
                 settings = GenericSignal "Signal1"
                 units    = None
               }

            let signal2 = 
                { id       = System.Guid.NewGuid() |> SignalId
                  settings = GenericSignal "Signal1"
                  units    = None
                }
             
            
            let! _= db.SaveExperimentSignals(expt1.id, [|signal1|])
            let! _= db.SaveExperimentSignals(expt1.id, [|signal2|])
            
            let! dbSignals = db.GetExperimentSignals expt1.id

            
            Expect.equal (dbSignals |> Array.map(fun s -> s.id) |> Set.ofSeq) (Set.ofSeq [signal1.id; signal2.id]) ""
        }
        ]

[<EntryPoint>]
let main(args) = 
  let config =
      { defaultConfig with
          ``parallel`` = false //TODO: if parallel events are supported, tests must be robust to interleavings
          //stress = Some (TimeSpan.FromHours 0.2)
        }
  runTestsWithArgs config args tests  