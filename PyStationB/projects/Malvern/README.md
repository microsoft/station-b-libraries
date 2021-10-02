## Getting started

1. Install [Node.js](https://nodejs.org/en/). This will automatically install Node Package Manager (npm) too.

2. To install the dependencies for our app (specified in package.json):

`npm install`

## Setting up the conda envrionment
Run the following command from the Malvern directory:

`conda env create -f environment.yml`

## To Activate the conda env:

`conda activate malvern`


## Running the app

### To start the React app

`npm run start`

### To start the Flask API
`npm run start-api`

## Connecting to Azure 
When you first start the app you will be prompted to login. To do this, enter a connection string in the login tab. Follow the [instructions here](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&tabs=azure-portal#view-account-access-keys) to get the connection string for your storage account