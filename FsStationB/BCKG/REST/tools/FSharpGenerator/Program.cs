// -------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
// -------------------------------------------------------------------------------------------
using System.IO;
using Microsoft.OpenApi.Models;
using Microsoft.OpenApi.Readers;

namespace Generator
{
    class Program
    {
        static void Main(string[] args)
        {
            var openApiDocument = (OpenApiDocument)null;

            using (var stream = new FileStream("bckg_api.yml", FileMode.Open, FileAccess.Read))
            {
                openApiDocument = new OpenApiStreamReader().Read(stream, out var diagnostic);
            }

            var relativeDirectory = "../../../../../src";

            var relativeClientDirectory = Path.Combine(relativeDirectory, "Client");

            ClientGenerator.WriteApi(openApiDocument, Path.Combine(relativeClientDirectory, "Api.fs"));

            var relativeSharedDirectory = Path.Combine(relativeDirectory, "Shared");

            SharedGenerator.WriteClientPaths(openApiDocument, Path.Combine(relativeSharedDirectory, "ClientPaths.fs"));
            SharedGenerator.WriteShared(openApiDocument, Path.Combine(relativeSharedDirectory, "Shared.fs"));
            SharedGenerator.WriteCodec(openApiDocument, Path.Combine(relativeSharedDirectory, "Codec.fs"));

            var relativeServerDirectory = Path.Combine(relativeDirectory, "Server");

            ServerGenerator.WriteRouteTable(openApiDocument, Path.Combine(relativeServerDirectory, "RouteTable.fs"));
            ServerGenerator.WriteStorage(openApiDocument, Path.Combine(relativeServerDirectory, "Storage.fs"));
        }
    }
}
