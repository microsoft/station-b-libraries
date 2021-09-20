@echo off

REM
REM From:
REM https://safe-stack.github.io/docs/quickstart/#create-your-first-safe-app
REM
REM Install: .NET Core 3.1 SDK (v3.1.404)
REM Install: npm (via NodeJS)
REM
REM The RPC server will listen on localhost:8085
REM A test web client will listen on localhost:8080

REM For the first time:
REM dotnet new -i SAFE.Template

dotnet new SAFE --name BCKG_REST_Server

dotnet tool restore

dotnet fake build --target run

REM
REM https://safe-stack.github.io/docs/recipes/developing-and-testing/testing-the-server/
REM

dotnet paket add Thoth.Json -p Shared
dotnet paket add Thoth.Json.Net -p Shared

dotnet paket add Fable.SimpleHttp -p Client

dotnet paket add Microsoft.NET.Test.Sdk -p Server.Tests

dotnet paket add YoloDev.Expecto.TestSdk -p Server.Tests

dotnet paket add Argu -p Server
dotnet paket add WindowsAzure.Storage --version 9.3.3 -p Server

dotnet new console --language C# --name Generator --output tools\FSharpGenerator
dotnet sln add --solution-folder tools\FSharpGenerator .\tools\FSharpGenerator\Generator.csproj

dotnet add tools\FSharpGenerator\Generator.csproj package Microsoft.OpenApi
dotnet add tools\FSharpGenerator\Generator.csproj package Microsoft.OpenApi.Readers
