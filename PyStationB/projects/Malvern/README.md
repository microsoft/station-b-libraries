## Getting started
1. Install [Node.js](https://nodejs.org/en/). This will automatically install Node Package Manager (npm) too.
2. To install the dependencies for our app (specified in package.json):
```cd app```
```npm install```

## Setting up the backend (Flask api)
To create the virtual environment:

```pyStationB/app/api$ python -m venv MalvernEnv```

## To install the packages:
```pyStationB/app/api$ python -m pip install -r ../requirements_dev.txt```
```pyStationB/app/api$ python -m pip install ../../libraries/ABEX```
```pyStationB/app/api$ python -m pip install ../../libraries/Emukit```

## Install pyBCKG git submodule:
```pyStationB/app$ cd ../ ```
```pyStationB$ git submodule update --init --recursive``

## To Activate the virtual env:

```pyStationB/app$ api\MalvernEnv\Scripts\activate Malvern``

## Running the app

### To start the React app
`app$ npm run start`

### To start the Flask API
`app$ npm run start-api`
