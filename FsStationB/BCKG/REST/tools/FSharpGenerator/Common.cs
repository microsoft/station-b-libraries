// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Microsoft.OpenApi.Models;

namespace Generator
{
    public static class Common
    {
        private static readonly char[] delimiterChars = { ' ', '/', '{', '}', '-' };

        private static readonly Dictionary<OperationType, string> ClientPathRoots = new Dictionary<OperationType, string>()
        {
            [OperationType.Get] = "get",
            [OperationType.Post] = "post",
            [OperationType.Patch] = "patch"
        };

        public static readonly Dictionary<OperationType, string> HttpVerbs = new Dictionary<OperationType, string>()
        {
            [OperationType.Get] = "GET",
            [OperationType.Post] = "POST",
            [OperationType.Patch] = "PATCH"
        };

        private static string FirstLetterToUpper(string str)
        {
            if (str.Length > 1)
            {
                return char.ToUpper(str[0]) + str.Substring(1);
            }
            else
            {
                return str.ToUpper();
            }
        }

        private static string FirstLetterToLower(string str)
        {
            if (str.Length > 1)
            {
                return char.ToLower(str[0]) + str.Substring(1);
            }
            else
            {
                return str.ToLower();
            }
        }

        private static string FormatPathKey(string pathKey)
        {
            var pathKeyFragments = pathKey.Split(delimiterChars)
                //.Select(s => s.Trim())
                //.Where(s => !string.IsNullOrWhiteSpace(s))
                .Select(FirstLetterToUpper)
                .ToArray();

            return string.Join("", pathKeyFragments);
        }

        private static string FormatClientPath(OperationType op, string pathKey) =>
            string.Format("{0}{1}", ClientPathRoots[op], FormatPathKey(pathKey));

        private static string FormatClientRoute(
            KeyValuePair<string, OpenApiPathItem> path)
        {
            if (path.Value.Parameters.Count == 0)
            {
                return path.Key;
            }
            else
            {
                var formatString = path.Key;

                foreach (var parameter in path.Value.Parameters)
                {
                    if (parameter.Schema.Type == "string")
                    {
                        formatString = formatString.Replace("{" + parameter.Name + "}", "%s");
                    }
                    else
                    {
                        throw new Exception(string.Format("Unhandled parameter type: {0}", parameter.Schema.Type));
                    }
                }

                return formatString;
            }
        }

        public struct PathItemTypes
        {
            public string[] ParameterNames;
            public string[] ParameterPathTypes;
            public string[] ParameterProperTypes;
            public string[] ParameterProperTypeDecoders;
            public string[] ParameterProperTypeEncoders;
            public string RequestBodyType;
            public string ResponseType;
        }

        public struct FormattedPathItemTypes
        {
            public string ClientPath;
            public string APIPath;
            public string FormattedClientRoute;
            public string FormattedParameterNames;
            public string FormattedParameterPathTypes;
            public string FormattedParameterProperTypes;
            public string FormattedRequestBodyName;
            public string FormattedRequestBodyType;
            public string FormattedResponseType;
        }

        public static FormattedPathItemTypes FormatPathItemTypes(
            KeyValuePair<OperationType, OpenApiOperation> op,
            KeyValuePair<string, OpenApiPathItem> path,
            PathItemTypes pathItemTypes)
        {
            var f = new FormattedPathItemTypes();
            f.ClientPath = FormatClientPath(op.Key, path.Key);
            f.APIPath = FormatClientPath(op.Key, path.Key) + "API";
            f.FormattedClientRoute = FormatClientRoute(path);
            f.FormattedParameterNames = pathItemTypes.ParameterNames.Count() <= 1 ? string.Join("", pathItemTypes.ParameterNames) : string.Format("({0})", string.Join(",", pathItemTypes.ParameterNames));
            f.FormattedParameterPathTypes = !pathItemTypes.ParameterPathTypes.Any() ? "unit" : string.Join("*", pathItemTypes.ParameterPathTypes);
            f.FormattedParameterProperTypes = !pathItemTypes.ParameterProperTypes.Any() ? "unit" : string.Join("*", pathItemTypes.ParameterProperTypes);
            f.FormattedRequestBodyName = string.IsNullOrEmpty(pathItemTypes.RequestBodyType) ? string.Empty : "body"; // FirstLetterToLower(pathItemTypes.RequestBodyType);
            f.FormattedRequestBodyType = pathItemTypes.RequestBodyType;
            f.FormattedResponseType = string.IsNullOrEmpty(pathItemTypes.ResponseType) ? "Async<unit>" : string.Format("Async<{0}>", pathItemTypes.ResponseType);
            return f;
        }

        public static PathItemTypes IdentityPathItemTypes(
            KeyValuePair<OperationType, OpenApiOperation> op,
            KeyValuePair<string, OpenApiPathItem> path)
        {
            var p = new PathItemTypes();
            p.ParameterNames = path.Value.Parameters.Select(p => p.Name).ToArray();
            p.ParameterPathTypes = path.Value.Parameters.Select(p => p.Schema.Type).ToArray();
            p.ParameterProperTypes = path.Value.Parameters.Select(IdentifyParameterType).ToArray();
            p.ParameterProperTypeDecoders = path.Value.Parameters.Select(IdentifyParameterTypeDecoder).ToArray();
            p.ParameterProperTypeEncoders = path.Value.Parameters.Select(IdentifyParameterTypeEncoder).ToArray();
            p.RequestBodyType = op.Value.RequestBody != null ? IdentifyContentType(op.Value.RequestBody.Content, true) : string.Empty;
            p.ResponseType = IdentifyContentType(op.Value.Responses["200"].Content, false);
            return p;
        }

        private static readonly Dictionary<string, string> LookupParameterTypes = new Dictionary<string, string>()
        {
        };

        private static string IdentifyParameterType(
            OpenApiParameter p)
        {
            if (LookupParameterTypes.ContainsKey(p.Name))
            {
                return LookupParameterTypes[p.Name];
            }

            return FirstLetterToUpper(p.Name);
        }

        private static readonly Dictionary<string, string> LookupParameterTypeDecoders = new Dictionary<string, string>()
        {
            ["ObservationId"] = "observationIdFromString",
            ["PartId"] = "partIdFromString",
            ["ReagentId"] = "reagentIdFromString",
            ["Guid"] = "guidFromString"
        };

        public static string IdentifyParameterTypeDecoder(
            OpenApiParameter p)
        {
            var parameterType = IdentifyParameterType(p);

            if (LookupParameterTypeDecoders.ContainsKey(parameterType))
            {
                return LookupParameterTypeDecoders[parameterType];
            }

            return string.Format("{0}.fromString", parameterType);
        }

        private static readonly Dictionary<string, string> LookupParameterTypeEncoders = new Dictionary<string, string>()
        {
            ["ObservationId"] = "observationIdToString",
            ["PartId"] = "partIdToString",
            ["ReagentId"] = "reagentIdToString",
            ["Guid"] = "guidToString"
        };

        private static string IdentifyParameterTypeEncoder(
            OpenApiParameter p)
        {
            var parameterType = IdentifyParameterType(p);

            if (LookupParameterTypeEncoders.ContainsKey(parameterType))
            {
                return LookupParameterTypeEncoders[parameterType];
            }

            return string.Format("{0}.toString", parameterType);
        }

        private static string IdentifyContentType(
            IDictionary<string, OpenApiMediaType> content,
            bool throwIfUnhandled)
        {
            if (content.ContainsKey("application/json"))
            {
                var jsonMediaType = content["application/json"];

                return IdentifyJsonMediaType(jsonMediaType.Schema);
            }
            else if (content.ContainsKey("text/plain"))
            {
                var textMediaType = content["text/plain"];

                return textMediaType.Schema.Type;
            }
            else if (content.ContainsKey("plain/text"))
            {
                // TODO Shouldn't this be text/plain?
                var textMediaType = content["plain/text"];

                return textMediaType.Schema.Type;
            }

            if (!throwIfUnhandled)
            {
                return string.Empty;
            }
            else
            {
                throw new Exception("Unhandled content type");
            }
        }

        private static string IdentifyJsonMediaType(OpenApiSchema schema)
        {
            if (schema.Type == "object")
            {
                if (schema.Reference != null)
                {
                    return schema.Reference.Id;
                }
                else
                {
                    throw new System.Exception("Unhandled object response type");
                }
            }
            else if (schema.OneOf.Count > 0)
            {
                return IdentifyUnionType(schema.OneOf);
            }
            else if (schema.Type == "array")
            {
                var arrayItemType = string.Empty;

                if (schema.Items.Reference != null)
                {
                    arrayItemType = schema.Items.Reference.Id;
                }
                else if (schema.Items.Type != null)
                {
                    arrayItemType = schema.Items.Type;
                }
                else if (schema.Items.OneOf.Count > 0)
                {
                    arrayItemType = IdentifyUnionType(schema.Items.OneOf);
                }
                else
                {
                    throw new System.Exception("Unhandled array response type");
                }

                return string.Format("{0}[]", arrayItemType);
            }
            else
            {
                throw new System.Exception("Unhandled response type");
            }
        }

        private static string IdentifyUnionType(IList<OpenApiSchema> schemas)
        {
            if (schemas.Any(f => f.Reference.Id == "BuildExperiment"))
            {
                return "Experiment";
            }
            else if (schemas.Any(f => f.Reference.Id == "DNA"))
            {
                return "Reagent";
            }
            else if (schemas.Any(f => f.Reference.Id == "PlateReaderFluorescence"))
            {
                return "Signal";
            }
            else if (schemas.Any(f => f.Reference.Id == "Prokaryote"))
            {
                return "Cell";
            }
            else
            {
                throw new System.Exception("Unhandled union type");
            }
        }

        public class SchemaType : IEquatable<SchemaType>
        {
            public string Name;
            public SchemaType BaseSchemaType;   // for arrays

            public override bool Equals(object obj)
            {
                if (obj == null) return false;
                var obj2 = obj as SchemaType;
                if (obj2 == null) return false;
                else return Equals(obj2);
            }
            public override int GetHashCode()
            {
                return Name.GetHashCode();
            }
            public bool Equals(SchemaType other)
            {
                if (other == null) return false;
                return
                    (Name.Equals(other.Name)) &&
                    ((BaseSchemaType == null && other.BaseSchemaType == null) ||
                     (BaseSchemaType != null && BaseSchemaType.Equals(other.BaseSchemaType)));
            }
        }

        public static SchemaType GetSchemaType(string schemaName)
        {
            if (schemaName.EndsWith("[]"))
            {
                var baseSchemaName = schemaName.Substring(0, schemaName.Length - 2);

                var schemaType = new SchemaType();
                schemaType.Name = schemaName;
                schemaType.BaseSchemaType = GetSchemaType(baseSchemaName);
                return schemaType;
            }
            else
            {
                var schemaType = new SchemaType();
                schemaType.Name = schemaName;
                schemaType.BaseSchemaType = null;
                return schemaType;
            }
        }

        public static SchemaType[] MergePathItemTypes(IEnumerable<PathItemTypes> pathItemTypess)
        {
            var usedTypes = new List<SchemaType>();

            void AddUsedType(string newType)
            {
                var schemaType = GetSchemaType(newType);

                if (schemaType.BaseSchemaType != null)
                {
                    if (!usedTypes.Contains(schemaType.BaseSchemaType))
                    {
                        usedTypes.Add(schemaType.BaseSchemaType);
                    }
                }

                if (!usedTypes.Contains(schemaType))
                {
                    usedTypes.Add(schemaType);
                }
            }

            foreach (var pathItemTypes in pathItemTypess)
            {
                if (!string.IsNullOrEmpty(pathItemTypes.RequestBodyType))
                {
                    AddUsedType(pathItemTypes.RequestBodyType);
                }

                if (!string.IsNullOrEmpty(pathItemTypes.ResponseType))
                {
                    AddUsedType(pathItemTypes.ResponseType);
                }
            }

            return usedTypes.ToArray();
        }

        public static string[] ReagentTypes = {"RNAReagent", "DNAReagent", "ChemicalReagent", "ProteinReagent", "GenericEntityReagent" };

        public static string getSchemaName(string schemaName)
        {
            if(Common.ReagentTypes.Contains(schemaName)){
                return "Reagent";
            }
            return schemaName;
        }
        public class CodecNames
        {
            public string DecoderFunctionName;
            public string GetFunctionName;
            public string EncoderFunctionName;
            public string SetFunctionName;
            public string InstanceName;
            public CodecNames BaseCodecNames; // For arrays.
        }

        public static CodecNames FormatCodecNames(SchemaType schemaType)
        {

            if (schemaType.BaseSchemaType == null)
            {
                var lowerSchemaName = FirstLetterToLower(schemaType.Name);
                var schemaName = schemaType.Name;
                if (ReagentTypes.Contains(schemaType.Name))
                {
                    lowerSchemaName = "reagent";
                    schemaName = "Reagent";
                }
                var c = new CodecNames();
                c.DecoderFunctionName = string.Format("{0}Decoder", lowerSchemaName);
                c.GetFunctionName = string.Format("get{0}", schemaName);
                c.EncoderFunctionName = string.Format("{0}Encoder", lowerSchemaName);
                c.SetFunctionName = string.Format("set{0}", schemaName);
                c.InstanceName = lowerSchemaName;
                return c;
            }
            else
            {
                var lowerSchemaName = FirstLetterToLower(schemaType.BaseSchemaType.Name);

                var c = new CodecNames();
                c.DecoderFunctionName = string.Format("{0}ArrayDecoder", lowerSchemaName);
                c.GetFunctionName = string.Format("get{0}Array", schemaType.BaseSchemaType.Name);
                c.EncoderFunctionName = string.Format("{0}ArrayEncoder", lowerSchemaName);
                c.SetFunctionName = string.Format("set{0}Array", schemaType.BaseSchemaType.Name);
                c.InstanceName = lowerSchemaName + "Array";
                c.BaseCodecNames = FormatCodecNames(schemaType.BaseSchemaType);
                return c;
            }
        }
    }
}
