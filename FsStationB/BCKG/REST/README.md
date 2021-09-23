# BCKG REST API using SAFE Template

This template can be used to generate a full-stack web application using the [SAFE Stack](https://safe-stack.github.io/). It was created using the dotnet [SAFE Template](https://safe-stack.github.io/docs/template-overview/). If you want to learn more about the template why not start with the [quick start](https://safe-stack.github.io/docs/quickstart/) guide?

## Install pre-requisites
You'll need to install the following pre-requisites in order to build SAFE applications

* The [.NET Core SDK](https://www.microsoft.com/net/download) 3.1 or higher.
* [npm](https://nodejs.org/en/download/) package manager.
* [Node LTS](https://nodejs.org/en/download/).

## Starting the application
Before you run the project **for the first time only** you must install dotnet "local tools" with this command:

```bash
dotnet tool restore
```

To concurrently run the server and the client components in watch mode use the following command:

```bash
dotnet fake build -t run
```

Then open `http://localhost:8080` in your browser.

To run concurrently server and client tests in watch mode (run in a new terminal):

```bash
dotnet fake build -t runtests
```

Client tests are available under `http://localhost:8081` in your browser and server tests are running in watch mode in console.

## SAFE Stack Documentation
If you want to know more about the full Azure Stack and all of it's components (including Azure) visit the official [SAFE documentation](https://safe-stack.github.io/docs/).

You will find more documentation about the used F# components at the following places:

* [Saturn](https://saturnframework.org/docs/)
* [Fable](https://fable.io/docs/)
* [Elmish](https://elmish.github.io/elmish/)

Test links:

* [Cell](http://localhost:8085/api/cells/CELL_ID)
* [Sample Conditions](http://localhost:8085/api/samples/SAMPLE_ID/conditions)

To run the server against a local storage edit build.fsx:

```
Target.create "Run" (fun _ ->
    dotnet "build" sharedPath ""
    [ async { dotnet "watch run" serverPath "--local --port 8085" }
      async { npm "run start" "." } ]
    |> Async.Parallel
    |> Async.RunSynchronously
    |> ignore
)
```

To run the server against a different end point:

```
Target.create "Run" (fun _ ->
    dotnet "build" sharedPath ""
    [ async { dotnet "watch run" serverPath "--connectionstring <Storage connection string> --port 8085" }
      async { npm "run start" "." } ]
    |> Async.Parallel
    |> Async.RunSynchronously
    |> ignore
)
```
