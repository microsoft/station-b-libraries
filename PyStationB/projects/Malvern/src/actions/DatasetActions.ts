import axios from "axios";
import {  Dispatch } from "redux";
import {  IDataset } from "../components/Interfaces";
import { IAppState } from "../reducers/RootReducer";
import { getFileFromInput } from "./actionUtils";
import { GET_DATASETS_FAIL, GET_DATASETS_SUCCESS, PARSE_AML_SECRETS_FAIL, PARSE_AML_SECRETS_SUCCESS, UPLOAD_DATASET_FAIL, UPLOAD_DATASET_SUCCESS } from "./DatasetActionTypes";

/// GET DATASETS /// 

function getDatasetOptions(api_url: string, connectionString: string) {
    return axios.get<IDataset[]>(
        api_url + '/get-dataset-options',
        { headers: { storageConnectionString: connectionString } }
    )
}

export function GetDatasetOptionsActionCreator(apiUrl: string, connectionString: string) {
    /// Unlike a plain action, within an actioncreator we return a function - the "thunk".
    // When this function is passed to `dispatch`, the thunk middleware will intercept it,
    // and call it with `dispatch` and `getState` as arguments.
    return function (dispatch: Dispatch) {
        return getDatasetOptions(apiUrl, connectionString).then(
            (res) => ( res.data ))
            .then((data_options) =>
               dispatch({
                   type: GET_DATASETS_SUCCESS,
                   payload: {
                       dataset_options: data_options
                   }
               }),
                (error) => dispatch({
                    type: GET_DATASETS_FAIL,
                    
                }),
        );
    };
}


/// UPLOAD OBSERVATIONS ///

function uploadObservations(apiUrl: string, connectionString: string, uploadedObservations: File, binary: string) {
    const data = new FormData()
    data.append('file', uploadedObservations)
    const response = axios.post(
        apiUrl + '/upload-observation-data',
        data,
        {
            headers: {
                storageConnectionString: connectionString,
                fileName: uploadedObservations.name || "",
                // observations: binary
            }
        },
    )
    console.log('response from upload observations: ', response)
    return response
}


export function UploadObservationsActionCreator(apiUrl: string, formData: FormData) {
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        const uploadedObservations = formData.get('uploadObservations') as File
        console.log('observations in action: ', uploadedObservations)

        // load data from file as binary
        getFileFromInput(uploadedObservations)
            .then(binary => {
                console.log('returning binary: ', binary)
                console.log('uploaded observations: ', uploadedObservations)
                return uploadObservations(apiUrl, connectionString, uploadedObservations, binary)
                    .then((response) =>
                        dispatch({
                            type: UPLOAD_DATASET_SUCCESS,
                            payload: response.data
                        }),
                        (error) => dispatch({
                            type: UPLOAD_DATASET_FAIL,
                            error: error
                        }),
                    );
            })


    };
}

// TODO: move this
// PARSE AML FILE


function parseAMLFile(apiUrl: string, connectionString: string, uploadedAMLSecrets: File, binary: string) {
    const data = new FormData()
    data.append('file', uploadedAMLSecrets)
    const response = axios.post(
        apiUrl + '/parse-aml-secrets',
        data,
        {
            headers: {
                storageConnectionString: connectionString,
                
                // observations: binary
            }
        },
    )
    console.log('response from parse AML secrets: ', response)
    return response

}



export function parseAMLFileActionCreator(apiUrl: string, formData: FormData){
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        const uploadedAMLFile = formData.get('uploadAMLSecrets') as File
        console.log('Uploaded AML secrets in action: ', uploadedAMLFile)

        // load data from file as binary
        getFileFromInput(uploadedAMLFile)
            .then(binary => {
                console.log('returning binary: ', binary)
                console.log('uploaded AML File: ', uploadedAMLFile)
                return parseAMLFile(apiUrl, connectionString, uploadedAMLFile, binary)
                    .then((aml_config) =>
                        
                        dispatch({
                            type: PARSE_AML_SECRETS_SUCCESS,
                            payload: {
                                aml_config: aml_config
                            }
                        }),
                        (error) => dispatch({
                            type: PARSE_AML_SECRETS_FAIL,
                            error: error
                        }),
                    );
            })


    };
}