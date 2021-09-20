using Microsoft.OpenApi.Models;
using SharpYaml.Schemas;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Generator
{
    public static class ServerGenerator
    {
        private static readonly string RouteTableFileHeader = @"
//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Server.RouteTable

open Microsoft.AspNetCore.Http
open Giraffe

open BCKG.Domain

open BCKG_REST_Server.Shared.ClientPaths
open BCKG_REST_Server.Shared.Codec
open BCKG_REST_Server.Shared.Shared

open HandlerUtils

let routeTable (checkPermitted:HttpHandler) (lookupAPI:HttpContext->IBCKGApi) : HttpHandler =
    choose [
        checkPermitted
";

        private static readonly string RouteTableFileFooter = @"
        RequestErrors.NOT_FOUND ""Not Found""
]";

        private static readonly string StorageFileHeader = @"
//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Server.Storage

open System
open Config
open BCKG.Domain
open BCKG.Events
open BCKG_REST_Server.Shared.Shared
open BCKG_REST_Server.Server.StorageUtils

let getServer (userId:string) (connectionString:string) : IBCKGApi =
    let connectionSettings = 
        connectionString.Split ';'
        |> Array.map (fun x -> x.Trim())
        |> Array.filter ((<>) """")
        |> Array.map(fun opt -> 
            let fields = opt.Split '='
            fields.[0], fields.[1]) 
        |> Map.ofArray    

    printfn ""Using connection settings: %A"" connectionString
    let db = BCKG.API.Instance(BCKG.API.InstanceType.CloudInstance(connectionString), userId)

    let bckgApi = {
";

        private static readonly string StorageFileFooter = @"
    }
    bckgApi

type BCKGApiConfig(connectionString:string) =
    interface IBCKGApiConfig with
        member _.GetApi (userId:string) : IBCKGApi =
            getServer userId connectionString
";

        private static Dictionary<string, string> KnownAPIMappings = new Dictionary<string, string>()
        {

            //GET PART ENDPOINTS
            ["postPartsGuidAPI"] = "fun (guid:Guid) (body:Part) -> processPart db (guid, body) RequestType.POST",
            ["patchPartsGuidAPI"] = "fun (guid:Guid) (body:Part) -> processPart db (guid, body) RequestType.PATCH",
            ["getPartsPromoterGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> PromoterId |> PromoterPartId))",
            ["getPartsRbsGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> RBSId |> RBSPartId))",
            ["getPartsCdsGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> CDSId |> CDSPartId))",
            ["getPartsTerminatorGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e  |> TerminatorId |> TerminatorPartId))",
            ["getPartsOriGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> OriId |> OriPartId))",
            ["getPartsBackboneGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> BackboneId |> BackbonePartId))",
            ["getPartsUserdefinedGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> UserDefinedId |> UserDefinedPartId))",
            ["getPartsScarGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> ScarId |> ScarPartId))",
            ["getPartsLinkerGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> LinkerId |> LinkerPartId))",
            ["getPartsRestrictionsiteGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetPart (e |> RestrictionSiteId |> RestrictionSitePartId))",
            
            //REAGENT ENDPOINTS
            //DNA
            ["getReagentsDnaGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetReagent (e |> DNAId |> DNAReagentId))",
            ["postReagentsDnaGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST",
            ["patchReagentsDnaGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH",
            //RNA
            ["getReagentsRnaGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetReagent (e |> RNAId |> RNAReagentId))",
            ["postReagentsRnaGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST",
            ["patchReagentsRnaGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH",
            //Protein
            ["getReagentsProteinGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetReagent (e |> ProteinId |> ProteinReagentId))",
            ["postReagentsProteinGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST",
            ["patchReagentsProteinGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH",
            //Chemical
            ["getReagentsChemicalGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetReagent (e |> ChemicalId |> ChemicalReagentId))",
            ["postReagentsChemicalGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST",
            ["patchReagentsChemicalGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH",
            //GenericEntity
            ["getReagentsGenericentityGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetReagent (e |> GenericEntityId |> GenericEntityReagentId))",
            ["postReagentsGenericentityGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.POST",
            ["patchReagentsGenericentityGuidAPI"] = "fun (guid:Guid) (body:Reagent) -> processReagent db (guid, body) RequestType.PATCH",

            //CELLS ENDPOINTS
            //Cells
            ["getCellsGuidAPI"] = "unpack1ReturnOption (fun e -> db.TryGetCell (e  |> CellId))",
            ["postCellsGuidAPI"] = "fun (guid:Guid) (body:Cell) -> processCell db (guid, body) RequestType.POST",
            ["patchCellsGuidAPI"] = "fun (guid:Guid) (body:Cell) -> processCell db (guid, body) RequestType.PATCH",
            //Cell Entities
            ["getCellsGuidEntitiesAPI"] = "fun (guid:Guid) -> db.GetCellEntities (guid |> CellId)",
            ["patchCellsGuidAddEntitiesAPI"] = "unpack2ReturnBool(fun (guid:Guid) (body:CellEntity[]) -> db.SaveCellEntities (guid |> CellId) (body))",
            ["patchCellsGuidRemoveEntitiesAPI"] = "unpack2ReturnBool(fun (guid:Guid) (body:CellEntity[]) -> db.RemoveCellEntities (guid |> CellId) (body))",

            //EXPERIMENTS ENDPOINTS
            //Experiment
            ["getExperimentsExperimentIdAPI"] = "unpack1ReturnOption db.TryGetExperiment",
            ["postExperimentsExperimentIdAPI"] = "fun (ExperimentId guid) (body:Experiment) -> processExperiment db (guid, body) RequestType.POST",
            ["patchExperimentsExperimentIdAPI"] = "fun (ExperimentId guid) (body:Experiment) -> processExperiment db (guid, body) RequestType.PATCH",
            //Experiment Samples
            ["postExperimentsExperimentIdSamplesAPI"] = "unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Sample[]) -> db.SaveSamples body)",
            //Experiment Signals
            ["getExperimentsExperimentIdSignalsAPI"] = "fun (experimentId:ExperimentId) -> db.GetExperimentSignals experimentId",
            ["patchExperimentsExperimentIdAddSignalsAPI"] = "unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Signal[]) -> db.SaveExperimentSignals(experimentId,body))",
            ["patchExperimentsExperimentIdRemoveSignalsAPI"] = "unpack2ReturnBool(fun (experimentId:ExperimentId) (body:Signal[]) -> db.RemoveExperimentSignals(experimentId,body))",
            //Experiment Operations
            ["getExperimentsExperimentIdOperationsAPI"] = "fun (experimentId:ExperimentId) -> db.GetExperimentOperations experimentId",
            ["patchExperimentsExperimentIdAddOperationAPI"] = "fun (experimentId:ExperimentId) (body:ExperimentOperation) -> processExperimentOperation db (experimentId, body) (AddRemoveType.ADD)",
            ["patchExperimentsExperimentIdRemoveOperationAPI"] = "fun (experimentId:ExperimentId) (body:ExperimentOperation) -> processExperimentOperation db (experimentId, body) (AddRemoveType.REMOVE)",

            //TAG API ENDPOINTS
            //Parts
            //Promoters
            ["getPartsPromoterGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag)",
            ["patchPartsPromoterGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsPromoterGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> PromoterId |> PromoterPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //RBS
            ["getPartsRbsGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag)",
            ["patchPartsRbsGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsRbsGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RBSId |> RBSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //CDS
            ["getPartsCdsGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag)",
            ["patchPartsCdsGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsCdsGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CDSId |> CDSPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //Terminator
            ["getPartsTerminatorGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag)",
            ["patchPartsTerminatorGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsTerminatorGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> TerminatorId |> TerminatorPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //ORI
            ["getPartsOriGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag)",
            ["patchPartsOriGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsOriGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> OriId |> OriPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //Backbone
            ["getPartsBackboneGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag)",
            ["patchPartsBackboneGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsBackboneGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> BackboneId |> BackbonePartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //UserDefined
            ["getPartsUserdefinedGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag)",
            ["patchPartsUserdefinedGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsUserdefinedGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> UserDefinedId |> UserDefinedPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //Scar
            ["getPartsScarGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag)",
            ["patchPartsScarGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsScarGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ScarId |> ScarPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //Linker
            ["getPartsLinkerGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag)",
            ["patchPartsLinkerGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsLinkerGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> LinkerId |> LinkerPartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            //RestrictionSite
            ["getPartsRestrictionsiteGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag)",
            ["patchPartsRestrictionsiteGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag, body) (AddRemoveType.ADD)",
            ["patchPartsRestrictionsiteGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RestrictionSiteId |> RestrictionSitePartId |> EventsProcessor.PartTag, body) (AddRemoveType.REMOVE)",
            /*Reagents*/
            //DNA
            ["getReagentsDnaGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> DNAId |> DNAReagentId |> EventsProcessor.ReagentTag)",
            ["patchReagentsDnaGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> DNAId |> DNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)",
            ["patchReagentsDnaGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> DNAId |> DNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)",
            //RNA
            ["getReagentsDnaGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag)",
            ["patchReagentsRnaGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)",
            ["patchReagentsRnaGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> RNAId |> RNAReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)",
            //Chemical
            ["getReagentsChemicalGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag)",
            ["patchReagentsChemicalGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)",
            ["patchReagentsChemicalGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ChemicalId |> ChemicalReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)",
            //Protein
            ["getReagentsProteinGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag)",
            ["patchReagentsProteinGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)",
            ["patchReagentsProteinGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> ProteinId |> ProteinReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)",
            //Generic Entity
            ["getReagentsGenericentityGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag)",
            ["patchReagentsGenericentityGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.ADD)",
            ["patchReagentsGenericentityGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> GenericEntityId |> GenericEntityReagentId |> EventsProcessor.ReagentTag, body) (AddRemoveType.REMOVE)",
            //Cells
            ["getCellsGuidTagsAPI"] = "fun (guid:Guid) -> getTags db (guid |> CellId |> EventsProcessor.CellTag)",
            ["patchCellsGuidAddTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CellId |> EventsProcessor.CellTag, body) (AddRemoveType.ADD)",
            ["patchCellsGuidRemoveTagsAPI"] = "fun (guid:Guid) (body:string[]) -> processTagEvent db (guid |> CellId |> EventsProcessor.CellTag, body) (AddRemoveType.REMOVE)",
            //Samples
            ["getSamplesSampleIdTagsAPI"] = "fun (guid:SampleId) -> getTags db (guid |> EventsProcessor.SampleTag)",
            ["patchSamplesSampleIdAddTagsAPI"] = "fun (guid:SampleId) (body:string[]) -> processTagEvent db (guid |> EventsProcessor.SampleTag, body) (AddRemoveType.ADD)",
            ["patchSamplesSampleIdRemoveTagsAPI"] = "fun (guid:SampleId) (body:string[]) -> processTagEvent db (guid |> EventsProcessor.SampleTag, body) (AddRemoveType.REMOVE)",
            //Experiments
            ["getExperimentsExperimentIdTagsAPI"] = "fun (guid:ExperimentId) -> getTags db (guid |> EventsProcessor.ExperimentTag)",
            ["patchExperimentsExperimentIdAddTagsAPI"] = "fun (guid:ExperimentId) (body:string[]) -> processTagEvent db (guid |> EventsProcessor.ExperimentTag, body) (AddRemoveType.ADD)",
            ["patchExperimentsExperimentIdRemoveTagsAPI"] = "fun (guid:ExperimentId) (body:string[]) -> processTagEvent db (guid |> EventsProcessor.ExperimentTag, body) (AddRemoveType.REMOVE)",

            //Sample
            ["getExperimentsExperimentIdSamplesAPI"] = "db.GetExperimentSamples",
            ["getSamplesSampleIdAPI"] = "unpack1ReturnOption(fun e-> db.TryGetSample e )",
            ["postSamplesSampleIdAPI"] = "fun (sampleId: SampleId) (body: Sample) -> processSample db (sampleId, body) RequestType.POST",
            ["patchSamplesSampleIdAPI"] = "fun (sampleId: SampleId) (body: Sample) -> processSample db (sampleId, body) RequestType.PATCH",

            //SampleConditions
            ["getSamplesSampleIdConditionsAPI"] = "db.GetSampleConditions",
            ["patchSamplesSampleIdAddConditionsAPI"] = "fun (sampleId: SampleId) (body:Condition[]) -> processSampleConditionEvent db (sampleId, body) (AddRemoveType.ADD)",
            ["patchSamplesSampleIdRemoveConditionsAPI"] = "fun (sampleId: SampleId) (body:Condition[]) -> processSampleConditionEvent db (sampleId, body) (AddRemoveType.REMOVE)",

            //SampleDevices
            ["getSamplesSampleIdDevicesAPI"] = "fun e-> db.GetSampleDevices e",
            // TODO: fix ["getSamplesSampleIdDeviceCellIdAPI"] = " unpack2ReturnOption(fun (sampleId: SampleId) (cellId: CellId) -> db.TryGetSampleDevice(sampleId, cellId))",
            ["patchSamplesSampleIdAddDevicesAPI"] = "fun (sampleId:SampleId) (body: SampleDevice[]) -> processDeviceEvent db (sampleId , body) (AddRemoveType.ADD)",
            ["patchSamplesSampleIdRemoveDevicesAPI"] = "fun (sampleId:SampleId) (body: SampleDevice[]) -> processDeviceEvent db (sampleId , body) (AddRemoveType.REMOVE)",

            //Observations
            ["getSamplesSampleIdSignalIdObservationsAPI"] = "fun (sampleId: SampleId, signalId: SignalId) -> db.GetObservations (sampleId, signalId)",
            ["getSamplesSampleIdObservationsAPI"] = "db.GetSampleObservations",
            ["getObservationsObservationIdAPI"] = "unpack1ReturnOption ( fun e -> db.TryGetObservation e)",
            ["patchSamplesSampleIdSignalIdAddObservationsAPI"] = "fun (sampleId: SampleId) (signalId: SignalId) (body: SampleDevice[]) -> processObservationEvent db (sampleId, signalId, body) (AddRemoveType.ADD)",
            ["patchSamplesSampleIdSignalIdRemoveObservationsAPI"] = "fun (sampleId: SampleId) (signalId: SignalId) (body: SampleDevice[]) -> processObservationEvent db (sampleId, signalId, body) (AddRemoveType.REMOVE)",
            ["patchSamplesSampleIdAddObservationsAPI"] = "unpack2ReturnBool (fun (sampleId:SampleId) (body:Observation[]) -> db.SaveObservations body)",
            // Other
            ["getReagentsReagentIdFileRefsAPI"] = "db.GetReagentFiles",
            ["getSamplesSampleIdTimeseriesAPI"] = "unpack1ReturnOption db.TryGetTimeSeries",
            ["patchSamplesSampleIdAddTagsAPI"] = "unpackAsyncList (fun e t->db.AddSampleTag(e, (Tag)t))",
            ["patchSamplesSampleIdRemoveTagsAPI"] = "unpackAsyncList (fun e t->db.RemoveSampleTag(e, (Tag)t))",
            ["patchExperimentsExperimentIdAddTagsAPI"] = "unpackAsyncList (fun e t->db.AddExperimentTag(e, (Tag)t))",
            ["patchExperimentsExperimentIdRemoveTagsAPI"] = "unpackAsyncList (fun e t->db.RemoveExperimentTag(e, (Tag)t))",
        };

        public static void WriteRouteTable(OpenApiDocument openApiDocument, string filename)
        {

            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(RouteTableFileHeader.TrimStart());

                    foreach (var path in openApiDocument.Paths)
                    {
                        foreach (var op in path.Value.Operations)
                        {
                            var pathItemTypes = Common.IdentityPathItemTypes(op, path);

                            var f = Common.FormatPathItemTypes(op, path, pathItemTypes);

                            var bodySchemaType = Common.GetSchemaType(pathItemTypes.RequestBodyType);
                            var bodyCodecNames = Common.FormatCodecNames(bodySchemaType);

                            var responseSchemaType = Common.GetSchemaType(pathItemTypes.ResponseType);
                            var responseCodecNames = Common.FormatCodecNames(responseSchemaType);

                            if (!Common.HttpVerbs.ContainsKey(op.Key))
                            {
                                throw new Exception(string.Format("Unhandled operation type: {0}", op.Key.ToString("G")));
                            }

                            var httpVerb = Common.HttpVerbs[op.Key];

                            if (op.Key == OperationType.Get)
                            {
                                var encoderFunction = responseCodecNames.SetFunctionName;
                                if (Common.ReagentTypes.Contains(pathItemTypes.ResponseType))
                                {
                                    encoderFunction = "setReagent";
                                }
                                if (path.Value.Parameters.Count == 0)
                                {
                                    writer.WriteLine(
                                        "        {0} >=> route clientPaths.{1} >=> (get0Handler lookupAPI (fun iApi -> iApi.{2}) Encoders.{3})",
                                        httpVerb,
                                        f.ClientPath,
                                        f.APIPath,
                                        encoderFunction);
                                }
                                else if (path.Value.Parameters.Count == 1)
                                {
                                    writer.WriteLine(
                                        "        {0} >=> routef clientPaths.{1} (get1Handler lookupAPI {2} (fun iApi->iApi.{3}) Encoders.{4})",
                                        httpVerb,
                                        f.ClientPath,
                                        pathItemTypes.ParameterProperTypeDecoders[0],
                                        f.APIPath,
                                        encoderFunction);
                                }
                                else if (path.Value.Parameters.Count == 2)
                                {
                                    writer.WriteLine(
                                        "        {0} >=> routef clientPaths.{1} (get2Handler lookupAPI {2} {3} (fun iApi->iApi.{4}) Encoders.{5})",
                                        httpVerb,
                                        f.ClientPath,
                                        pathItemTypes.ParameterProperTypeDecoders[0],
                                        pathItemTypes.ParameterProperTypeDecoders[1],
                                        f.APIPath,
                                        encoderFunction);
                                }
                                else
                                {
                                    throw new Exception(string.Format("Unhandled operation type: {0} with no parameters", op.Key.ToString("G")));
                                }
                            }
                            else
                            {
                                if (path.Value.Parameters.Count == 0)
                                {
                                    if (responseSchemaType.Name == string.Empty)
                                    {
                                        writer.WriteLine(
                                            "        {0} >=> route clientPaths.{1} >=> (post0Handler lookupAPI Decoders.{2} (fun iApi -> iApi.{3}))",
                                            httpVerb,
                                            f.ClientPath,
                                            bodyCodecNames.GetFunctionName,
                                            f.APIPath);
                                    }
                                    else
                                    {
                                        writer.WriteLine(
                                            "        {0} >=> route clientPaths.{1} >=> (post0GetHandler lookupAPI Decoders.{2} (fun iApi -> iApi.{3}) Encoders.{4})",
                                            httpVerb,
                                            f.ClientPath,
                                            bodyCodecNames.GetFunctionName,
                                            f.APIPath,
                                            responseCodecNames.SetFunctionName);
                                    }
                                }
                                else
                                {
                                    if (responseSchemaType.Name == string.Empty)
                                    {
                                        writer.WriteLine(
                                            "        {0} >=> routef clientPaths.{1} (post1Handler lookupAPI {2} Decoders.{3} (fun iApi -> iApi.{4}))",
                                            httpVerb,
                                            f.ClientPath,
                                            pathItemTypes.ParameterProperTypeDecoders[0],
                                            bodyCodecNames.GetFunctionName,
                                            f.APIPath);
                                    }
                                    else
                                    {
                                        writer.WriteLine(
                                            "        {0} >=> routef clientPaths.{1} (post1GetHandler lookupAPI {2} Decoders.{3} (fun iApi -> iApi.{4}) Encoders.{5})",
                                            httpVerb,
                                            f.ClientPath,
                                            pathItemTypes.ParameterProperTypeDecoders[0],
                                            bodyCodecNames.GetFunctionName,
                                            f.APIPath,
                                            responseCodecNames.SetFunctionName);
                                    }
                                }
                            }
                        }
                    }

                    writer.Write(RouteTableFileFooter);
                }
            }
        }

        public static void WriteStorage(OpenApiDocument openApiDocument, string filename)
        {
            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(StorageFileHeader.TrimStart());

                    foreach (var path in openApiDocument.Paths)
                    {
                        foreach (var op in path.Value.Operations)
                        {
                            var pathItemTypes = Common.IdentityPathItemTypes(op, path);

                            var f = Common.FormatPathItemTypes(op, path, pathItemTypes);

                            var bodySchemaType = Common.GetSchemaType(pathItemTypes.RequestBodyType);
                            var bodyCodecNames = Common.FormatCodecNames(bodySchemaType);

                            var responseSchemaType = Common.GetSchemaType(pathItemTypes.ResponseType);
                            var responseCodecNames = Common.FormatCodecNames(responseSchemaType);

                            if (KnownAPIMappings.ContainsKey(f.APIPath))
                            {
                                writer.WriteLine(@"
        {0} = {1}",
                                    f.APIPath,
                                    KnownAPIMappings[f.APIPath]);
                            }
                            else
                            {
                                if (!Common.HttpVerbs.ContainsKey(op.Key))
                                {
                                    throw new Exception(string.Format("Unhandled operation type: {0}", op.Key.ToString("G")));
                                }

                                var httpVerb = Common.HttpVerbs[op.Key];

                                if (op.Key == OperationType.Get)
                                {
                                    if (path.Value.Parameters.Count == 0)
                                    {
                                        writer.WriteLine(
                                            @"        {0} = fun () -> async {{ return failwith ""Missing API.fs function {0}"" }}",
                                            f.APIPath);
                                    }
                                    else
                                    {
                                        writer.WriteLine(
                                            @"        {0} = fun ({1}:{2}) -> async {{ return failwith ""Missing API.fs function {0}"" }}",
                                            f.APIPath,
                                            f.FormattedParameterNames,
                                            f.FormattedParameterProperTypes);
                                    }
                                }
                                else
                                {
                                    if (path.Value.Parameters.Count == 0)
                                    {
                                        if (responseSchemaType.Name == string.Empty)
                                        {
                                            writer.WriteLine(
                                                @"        {0} = fun ({1}:{2}) -> async {{ failwith ""Missing API.fs function {0}"" }}",
                                                f.APIPath,
                                                f.FormattedRequestBodyName,
                                                f.FormattedRequestBodyType);
                                        }
                                        else
                                        {
                                            writer.WriteLine(
                                                @"        {0} = fun ({1}:{2}) -> async {{ return failwith ""Missing API.fs function {0}"" }}",
                                                f.APIPath,
                                                f.FormattedRequestBodyName,
                                                f.FormattedRequestBodyType);
                                        }
                                    }
                                    else
                                    {
                                        if (responseSchemaType.Name == string.Empty)
                                        {
                                            writer.WriteLine(
                                                @"        {0} = fun ({1}:{2}) ({3}:{4}) -> async {{ failwith ""Missing API.fs function {0}"" }}",
                                                f.APIPath,
                                                f.FormattedParameterNames,
                                                f.FormattedParameterProperTypes,
                                                f.FormattedRequestBodyName,
                                                f.FormattedRequestBodyType);
                                        }
                                        else
                                        {
                                            writer.WriteLine(
                                                @"        {0} = fun ({1}:{2}) ({3}:{4}) -> async {{ return failwith ""Missing API.fs function {0}"" }}",
                                                f.APIPath,
                                                f.FormattedParameterNames,
                                                f.FormattedParameterProperTypes,
                                                f.FormattedRequestBodyName,
                                                f.FormattedRequestBodyType);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    writer.Write(StorageFileFooter);
                }
            }
        }
    }
}
