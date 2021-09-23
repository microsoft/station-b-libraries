# F# Libraries

## Getting Started

The main tools are located in the folders [BCKG](BCKG), [BCKG REST Server](BCKG/REST), [FSTL](FSTL), and [PlateReaderLoader](PlateReaderLoader).  To build these solutions, you will need to configure Visual Studio with the necessary add-ins, and also install a few more dependencies.

The instructions for building on Windows are as follows:

1. **Install Visual Studio 2019**

    The components required are:
    - Workloads tab:
        - .NET desktop development
        - Desktop development with C++
        - Azure development
    - Individual Components tab, ".NET" section:
        - .NET Framework 4.7.2 SDK
        - .NET Framework 4.7.2 targeting pack
    - Individual Components tab, "Compilers, build tools, and runtimes" section:
        - MSVC (XYZ) - VS 2019; Libs for Spectre (x86 and x64) [where XYZ is the highest version corresponding to the VC++ 2019 version (XYZ) latest v142 tools, which might have been selected already]
    - Individual Components tab, "Development activities" section:
        - F# desktop language support
        - F# language support
        - F# language support for web projects
    - Individual Components tab, "SDKs, libraries, and frameworks" section:
        - Windows 10 SDK (10.0.17763.0)

2. **Install .NET Core SDKs**. These are available [here](https://dotnet.microsoft.com/download/visual-studio-sdks). The required versions are 2.1 and 3.1.

3. **Install NodeJS**. N.B. This may involve restarting your computer several times.

4. **Install Yarn**. The instructions (for Windows) are [here](https://classic.yarnpkg.com/en/docs/install/#windows-stable).
