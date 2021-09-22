// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
using Microsoft.OpenApi.Models;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace Generator
{
    public static class SharedGenerator
    {
        private static readonly string ClientPathsFileHeader = @"
//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Shared.ClientPaths

let apiRoot = ""/api""

";

        private static readonly string SharedFileHeader = @"
//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Shared.Shared

open System
open BCKG.Domain

let guidFromString (x:string) : Option<Guid> =
    try
        Some (System.Guid.Parse x)
    with
        | _  -> None

let guidToString (g:Guid) : string = g.ToString()

let observationIdFromString (x:string) : Option<ObservationId> = failwith ""Unimplemented conversion from string to ObservationId""

let observationIdToString (o:ObservationId) : string = failwith ""Unimplemented conversion from string to ObservationId""

let partIdFromString (x:string) : Option<PartId> = failwith ""Unimplemented conversion from string to PartId""

let partIdToString (p:PartId) : string = failwith ""Unimplemented conversion from string to PartId""

let reagentIdFromString (x:string) : Option<ReagentId> = failwith ""Unimplemented conversion from string to ReagentId""

let reagentIdToString (r:ReagentId) : string = failwith ""Unimplemented conversion from string to ReagentId""

";

        private static readonly string CodecFileHeader = @"
//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Shared.Codec

#if FABLE_COMPILER
open Thoth.Json
#else
open Thoth.Json.Net
#endif

open BCKG.Domain

open Shared

";

        private static readonly string CodecDecoderHeader = @"
module Decoders =
    let decode<'T> (decoder:Decoder<'T>) (name:string) (responseText:string) : 'T =
        let result = Decode.fromString decoder responseText
        match result with
        | Ok t ->
            printfn ""Got %s %A"" name t
            t
        | Error e ->
            failwithf ""Error parsing %s json: %A"" name e

";

        private static readonly string CodecEncoderHeader = @"
module Encoders =
    let encode<'T> (encoder:Encoder<'T>) (name:string) (o:'T) : string =
        let json = Encode.toString 0 (encoder o)
        printfn ""Set %s: %s"" name json
        json

";

        private static readonly Dictionary<string, string> KnownResponseMappings = new Dictionary<string, string>()
        {
            ["getReagentsDnaGuidAPI"] = "Async<Reagent>",
            ["getReagentsRnaGuidAPI"] = "Async<Reagent>",
            ["getReagentsProteinGuidAPI"] = "Async<Reagent>",
            ["getReagentsChemicalGuidAPI"] = "Async<Reagent>",
            ["getReagentsGenericentityGuidAPI"] = "Async<Reagent>",
        };

        private static readonly Dictionary<string, string> KnownCodecMappings = new Dictionary<string, string>()
        {
            ["partDecoder"] = "Part.decode",
            ["partEncoder"] = "Part.encode",
            ["reagentEncoder"] = "Reagent.encode",
            ["reagentDecoder"] = "Reagent.decode",
            ["cellEncoder"] = "Cell.encode",
            ["cellDecoder"] = "Cell.decode",
            ["cellEntityEncoder"] = "CellEntity.encode",
            ["cellEntityDecoder"] = "CellEntity.decode",
            ["sampleDecoder"] = "Sample.decode",
            ["sampleEncoder"] = "Sample.encode",
            ["conditionDecoder"] = "Condition.decode",
            ["conditionEncoder"] = "Condition.encode",
            ["sampleDeviceDecoder"] = "SampleDevice.decode",
            ["sampleDeviceEncoder"] = "SampleDevice.encode",
            ["experimentEncoder"] = "Experiment.encode",
            ["experimentDecoder"] = "Experiment.decode",
            ["experimentOperationEncoder"] = "ExperimentOperation.encode",
            ["experimentOperationDecoder"] = "ExperimentOperation.decode",
            ["signalEncoder"] = "Signal.encode",
            ["signalDecoder"] = "Signal.decode",
            ["observationEncoder"] = "Observation.encode",
            ["observationDecoder"] = "Observation.decode",
        };

        public static void WriteClientPaths(OpenApiDocument openApiDocument, string filename)
        {
            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(ClientPathsFileHeader.TrimStart());
                    writer.WriteLine("let clientPaths = {|");

                    foreach (var path in openApiDocument.Paths)
                    {
                        foreach (var op in path.Value.Operations)
                        {
                            var pathItemTypes = Common.IdentityPathItemTypes(op, path);

                            var f = Common.FormatPathItemTypes(op, path, pathItemTypes);

                            if (path.Value.Parameters.Count == 0)
                            {
                                writer.WriteLine(
                                    string.Format(
                                        "    {0} = apiRoot + \"{1}\"",
                                        f.ClientPath,
                                        f.FormattedClientRoute));
                            }
                            else
                            {
                                writer.WriteLine(
                                    string.Format(
                                        "    {0} = (PrintfFormat<{1}->string,unit,string,string,{1}>)(apiRoot + \"{2}\")",
                                        f.ClientPath,
                                        f.FormattedParameterPathTypes,
                                        f.FormattedClientRoute));
                            }
                        }
                    }

                    writer.WriteLine("|}");
                }
            }
        }

        public static void WriteShared(OpenApiDocument openApiDocument, string filename)
        {
            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(SharedFileHeader.TrimStart());
                    writer.WriteLine("type IBCKGApi = {");

                    foreach (var path in openApiDocument.Paths)
                    {
                        foreach (var op in path.Value.Operations)
                        {
                            var pathItemTypes = Common.IdentityPathItemTypes(op, path);

                            var f = Common.FormatPathItemTypes(op, path, pathItemTypes);
                            var formattedResponseType = f.FormattedResponseType;
                            if (KnownResponseMappings.ContainsKey(f.APIPath))
                            {
                                formattedResponseType = KnownResponseMappings[f.APIPath];

                            }
                            if (op.Key == OperationType.Get || !string.IsNullOrWhiteSpace(f.FormattedParameterNames))
                            {

                                var formattedRequestBodyType = Common.getSchemaName(f.FormattedRequestBodyType);
                                var requestBodyName = string.IsNullOrEmpty(f.FormattedRequestBodyName) ? string.Empty : string.Format("-> {0} ", f.FormattedRequestBodyName);
                                var requestBodyType = string.IsNullOrEmpty(formattedRequestBodyType) ? string.Empty : string.Format("-> {0} ", formattedRequestBodyType);


                                writer.WriteLine(
                                    string.Format("    //{0} : {1} {2}-> {3}",
                                    f.APIPath,
                                    f.FormattedParameterNames,
                                    requestBodyName,
                                    formattedResponseType));

                                writer.WriteLine(
                                    string.Format("    {0} : {1} {2}-> {3}",
                                    f.APIPath,
                                    f.FormattedParameterProperTypes,
                                    requestBodyType,
                                    formattedResponseType));
                            }
                            else
                            {
                                writer.WriteLine(
                                    string.Format("    //{0} : {1} -> {2}",
                                    f.APIPath,
                                    f.FormattedRequestBodyName,
                                    formattedResponseType));

                                writer.WriteLine(
                                    string.Format("    {0} : {1} -> {2}",
                                    f.APIPath,
                                    f.FormattedRequestBodyType,
                                    formattedResponseType));
                            }
                        }
                    }

                    writer.WriteLine("}");
                }
            }
        }

        public static void WriteCodec(OpenApiDocument openApiDocument, string filename)
        {
            var pathItemTypess = openApiDocument.Paths.SelectMany(
                path => path.Value.Operations.Select(
                    op => Common.IdentityPathItemTypes(op, path)));

            var usedTypes = Common.MergePathItemTypes(pathItemTypess);

            var allCodecNames = usedTypes
                .Select(t => new KeyValuePair<Common.SchemaType, Common.CodecNames>(t, Common.FormatCodecNames(t)))
                .ToDictionary(x => x.Key, x => x.Value);

            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(CodecFileHeader.TrimStart());
                    writer.Write(CodecDecoderHeader.TrimStart());

                    var decoderLineMaps = new Dictionary<string, string>();
                    foreach (var schemaType in usedTypes)
                    {
                        var codecNames = allCodecNames[schemaType];
                        var schemaTypeName = Common.getSchemaName(schemaType.Name);
                        if (codecNames.BaseCodecNames == null)
                        {
                            if (KnownCodecMappings.ContainsKey(codecNames.DecoderFunctionName))
                            {
                                var line =
                                    string.Format(
                                        "    let {0} : Decoder<{1}> = {2}",
                                        codecNames.DecoderFunctionName,
                                        schemaTypeName,
                                        KnownCodecMappings[codecNames.DecoderFunctionName]);
                                decoderLineMaps[codecNames.DecoderFunctionName] = line;
                            }
                            else
                            {
                                var line =
                                    string.Format(
                                       "    let {0} : Decoder<{1}> = Decode.Auto.generateDecoderCached<{1}>(CamelCase)",
                                       codecNames.DecoderFunctionName,
                                       schemaTypeName);
                                decoderLineMaps[codecNames.DecoderFunctionName] = line;
                            }
                        }
                        else
                        {
                            var line =
                                string.Format(
                                    "    let {0} : Decoder<{1}> = Decode.array {2}",
                                    codecNames.DecoderFunctionName,
                                    schemaTypeName,
                                    codecNames.BaseCodecNames.DecoderFunctionName);
                            decoderLineMaps[codecNames.DecoderFunctionName] = line;
                        }
                    }

                    var getLineMaps = new Dictionary<string, string>();
                    foreach (var schemaType in usedTypes)
                    {
                        var codecNames = allCodecNames[schemaType];
                        var schemaTypeName = Common.getSchemaName(schemaType.Name);
                        var line =
                            string.Format(
                                "    let {0} : responseText:string -> {2} = decode {1} \"{1}\"",
                                codecNames.GetFunctionName,
                                codecNames.DecoderFunctionName,
                                schemaTypeName);
                        getLineMaps[codecNames.GetFunctionName] = line;
                    }


                    foreach (var key in decoderLineMaps.Keys)
                    {
                        writer.WriteLine(decoderLineMaps[key]);
                        writer.WriteLine();
                    }
                    foreach (var key in getLineMaps.Keys)
                    {
                        writer.WriteLine(getLineMaps[key]);
                        writer.WriteLine();
                    }
                    writer.Write(CodecEncoderHeader.TrimStart());


                    var encoderLineMaps = new Dictionary<string, string>();
                    foreach (var schemaType in usedTypes)
                    {
                        var codecNames = allCodecNames[schemaType];
                        var schemaTypeName = Common.getSchemaName(schemaType.Name);
                        if (codecNames.BaseCodecNames == null)
                        {
                            if (KnownCodecMappings.ContainsKey(codecNames.EncoderFunctionName))
                            {
                                var line = string.Format(
                                    "    let {0} : Encoder<{1}> = {2}",
                                    codecNames.EncoderFunctionName,
                                    schemaTypeName,
                                    KnownCodecMappings[codecNames.EncoderFunctionName]);
                                encoderLineMaps[codecNames.EncoderFunctionName] = line;
                            }
                            else
                            {
                                var line = string.Format(
                                    "    let {0} : Encoder<{1}> = Encode.Auto.generateEncoderCached<{1}>(CamelCase)",
                                    codecNames.EncoderFunctionName,
                                    schemaTypeName);
                                encoderLineMaps[codecNames.EncoderFunctionName] = line;
                            }


                        }
                        else
                        {
                            var line = string.Format(
                                    "    let {0} : Encoder<{1}> = fun ts -> Encode.array (Array.map {2} ts)",
                                    codecNames.EncoderFunctionName,
                                    schemaTypeName,
                                    codecNames.BaseCodecNames.EncoderFunctionName);
                            encoderLineMaps[codecNames.EncoderFunctionName] = line;
                        }
                    }

                    var setLineMaps = new Dictionary<string, string>();
                    foreach (var schemaType in usedTypes)
                    {
                        var codecNames = allCodecNames[schemaType];
                        var schemaTypeName = Common.getSchemaName(schemaType.Name);
                        var line = string.Format(
                               "    let {0} : {3}:{2} -> string = encode {1} \"{1}\"",
                                codecNames.SetFunctionName,
                                codecNames.EncoderFunctionName,
                                schemaTypeName,
                                codecNames.InstanceName);
                        setLineMaps[codecNames.SetFunctionName] = line;

                    }
                    foreach (var key in encoderLineMaps.Keys)
                    {
                        writer.WriteLine(encoderLineMaps[key]);
                        writer.WriteLine();
                    }
                    foreach (var key in setLineMaps.Keys)
                    {
                        writer.WriteLine(setLineMaps[key]);
                        writer.WriteLine();
                    }
                }
            }
        }
    }
}
