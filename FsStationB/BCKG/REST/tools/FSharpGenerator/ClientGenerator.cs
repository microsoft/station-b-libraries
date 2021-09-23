// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
using Microsoft.OpenApi.Models;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace Generator
{
    public static class ClientGenerator
    {
        public static string ApiFileHeader = @"
module BCKG_REST_Server.Client.Api

open BCKG.Domain

open BCKG_REST_Server.Shared.ClientPaths
open BCKG_REST_Server.Shared.Codec
open BCKG_REST_Server.Shared.Shared

open HandlerUtils

let bckgApi : IBCKGApi = {
";

        public static string ApiFileFooter = @"
}
";

        public static void WriteApi(OpenApiDocument openApiDocument, string filename)
        {
            using (var stream = new FileStream(filename, FileMode.Create))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(ApiFileHeader.TrimStart());
                    var apiLines = new Dictionary<string, string>();
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
                                var decoderFunction = responseCodecNames.GetFunctionName;
                                if(Common.ReagentTypes.Contains(pathItemTypes.ResponseType))
                                {
                                    decoderFunction = "getReagent";
                                }
                                if (path.Value.Parameters.Count == 0)
                                {
                                    var line = string.Format(
                                        "        {0} = get0Handler clientPaths.{1} Decoders.{2}",
                                        f.APIPath,
                                        f.ClientPath,
                                        decoderFunction);
                                    apiLines[f.APIPath] = line;
                                }
                                else if (path.Value.Parameters.Count == 1)
                                {
                                    var line = string.Format(
                                        "        {0} = get1Handler {1} clientPaths.{2} Decoders.{3}",
                                        f.APIPath,
                                        pathItemTypes.ParameterProperTypeEncoders[0],
                                        f.ClientPath,
                                        decoderFunction);
                                    apiLines[f.APIPath] = line;
                                }
                                else if (path.Value.Parameters.Count == 2)
                                {
                                    var line = string.Format(
                                       "        {0} = get2Handler {1} {2} clientPaths.{3} Decoders.{4}",
                                        f.APIPath,
                                        pathItemTypes.ParameterProperTypeEncoders[0],
                                        pathItemTypes.ParameterProperTypeEncoders[1],
                                        f.ClientPath,
                                        decoderFunction);
                                    apiLines[f.APIPath] = line;
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
                                        var line = string.Format(
                                            "        {0} = postHandler clientPaths.{1} Encoders.{2}",
                                            f.APIPath,
                                            f.ClientPath,
                                            bodyCodecNames.SetFunctionName);
                                        apiLines[f.APIPath] = line;
                                    }
                                    else
                                    {
                                        var line = string.Format(
                                            "        {0} = postGetHandler clientPaths.{1} Encoders.{2} Decoders.{3}",
                                            f.APIPath,
                                            f.ClientPath,
                                            responseCodecNames.SetFunctionName,
                                            bodyCodecNames.GetFunctionName);
                                        apiLines[f.APIPath] = line;
                                    }
                                }
                                else
                                {
                                    if (responseSchemaType.Name == string.Empty)
                                    {
                                        var line = string.Format(
                                            "        {0} = post1Handler {1} clientPaths.{2} Encoders.{3}",
                                            f.APIPath,
                                            pathItemTypes.ParameterProperTypeEncoders[0],
                                            f.ClientPath,
                                            bodyCodecNames.SetFunctionName);
                                        apiLines[f.APIPath] = line;
                                    }
                                    else
                                    {
                                        var line = string.Format(
                                            "        {0} = post1GetHandler {1} clientPaths.{2} Encoders.{3} Decoders.{4}",
                                            f.APIPath,
                                            pathItemTypes.ParameterProperTypeEncoders[0],
                                            f.ClientPath,
                                            responseCodecNames.SetFunctionName,
                                            bodyCodecNames.GetFunctionName);
                                        apiLines[f.APIPath] = line;
                                    }
                                }
                            }
                        }
                    }
                    foreach (var key in apiLines.Keys)
                    {
                        writer.WriteLine(apiLines[key]);
                    }
                    writer.Write(ApiFileFooter);
                }
            }
        }
    }
}
