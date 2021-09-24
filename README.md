# Station B Libraries for Biological Computation

## Overview

This open source repository contains various libraries and wrappers that can enable a variety of computational workflows in the field of synthetic biology, genetic engineering, computational biology, and lab automation.

Station B Libraries is divided into two main implementations:

- [PyStationB](PyStationB): Libraries and Projects written in Python
- [FsStationB](FsStationB): Libraries and Projects written in F#

The workflows and libraries in repository can be broadly categorized into the following projects:

- **Biological Knowledge Graph (BCKG)**: A framework to ingest, structure, store, and retrieve biological knowledge and experimental data. The BCKG project comprises of the following:
  - [BCKG](FsStationB/BCKG): An F# library structure biological data and store and retrieve biological and experimental data using Azure [Table](https://azure.microsoft.com/en-gb/services/storage/tables/) and [Blob](https://azure.microsoft.com/en-gb/services/storage/blobs/) Storage.
  - [BCKG REST API Server](FsStationB/BCKG/REST): A [SAFE Stack](https://safe-stack.github.io/) implementation to host a REST API server for BCKG.
  - [PyBCKG](PyStationB/libraries/PyBCKG): An extensive Python library for BCKG, to represent biological data as structured python objects, and to retrieve data via wrappers for the Azure Storage Rest API. The library also contains functions to query the knowledge graph and plot timeseries data and observations.
- **Automated Biological Experimentation (ABEX)**: A framework to enable Gaussian process modelling and Bayesian optimization. The ABEX project seeks to reduce the number of experiments required to optimize a biological protocol. The goal of this project is to explore and compare the usage of frameworks like Bayesian optimization to effectively traverse a solution space against traditional block-design strategies. This project includes:
  - [Global Penalisation](PyStationB/libraries/GlobalPenalisation): a library that provides an implementation of the moment-matched approximation of qEI as well as benchmarks against other Bayesian Optimization acquisition functions.
  - [ABEX](PyStationB/libraries/ABEX): a wrapper around the publicly available [Emukit library](https://github.com/EmuKit/emukit)
  - [Cell Signalling](PyStationB/projects/CellSignalling): a project to demonstrate an end-to-end semi-automated pipeline that used Bayesian optimization, static characterization, and lab automation to optimize biological protocols for a desired objective.
- **Time-series modelling and characterization**: This includes the following:
  - [StaticCharacterization](PystationB/libraries/StaticCharacterization): A Python library to enable the static characterization of gene expression. To *characterize* how circuit activity depends on different conditions, the static characterization library seeks to **summarize the whole time series into a single number**.
  - [FSTL](FsStationB/FSTL): An F# library to specify properties of time-series traces using Signal Temporal Logic.
- **Lab Automation**: Libraries and scripts to enable end-to-end lab automation. This includes:
  - [Barcoder](PyStationB/projects/Barcoder): A Python library to annotate a Barcoded plate with reagent information based on the barcodes stored in the BCKG database.
  - [Plate Reader Loader](FsStationB/PlateReaderLoader): An F# library to parse Plate Reader data into time-series files and automate the process of uploading raw and processed plate-reader files to an Azure Blob storage instance.
- **Miscellaneous Utilities**:
  - [UniProt API Wrapper](PyStationB/libraries/UniProt): This API is a Python wrapper around the publicly available [UniProt Rest API](https://www.uniprot.org/help/api) retrieve protein information from the [UniProt database](https://www.uniprot.org/).
  - [Python Utilities](PyStationB/libraries/Utilities): A collection of scripts and utility code for the Python libraries.

Various projects and libraries in this repository also contain wrappers around plotting libraries to visualize and generate images for workflows in computational biology.

## Contributions

There have been several contributors to this codebase, prior to its migration to this location on GitHub. Significant contributions have come from the following (listed in alphabetical order):

- Boyan Yordanov ([byoyo](https://github.com/byoyo))
- Bruno Mlodozeniec ([BrunoKM](https://github.com/BrunoKM))
- Colin Gravill ([cgravill](https://github.com/cgravill))
- David Carter ([sennendoko](https://github.com/sennendoko))
- Javier González ([javiergonzalezh](https://github.com/javiergonzalezh))
- Jonathan Tripp ([JonathanTripp](https://github.com/JonathanTripp))
- Melissa Bristow ([MG92](https://github.com/MG92))
- Neil Dalchau ([ndalchau](https://github.com/ndalchau))
- Pawel Czyz ([pawel-czyz](https://github.com/pawel-czyz))
- Prashant Vaidyanathan ([PrashantVaidyanathan](https://github.com/PrashantVaidyanathan))

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
