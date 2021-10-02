import axios from "axios";
import { Dispatch } from "redux";
import { IAbexConfig } from "../components/Interfaces";
import { IAppState } from "../reducers/RootReducer";
import { getFileFromInput } from "./actionUtils";
import { GET_CONFIGS_FAIL, GET_CONFIGS_SUCCESS, UPLOAD_CONFIG_FAIL, UPLOAD_CONFIG_SUCCESS } from "./ConfigActionTypes";

/// GET CONFIG OPTIONS ///
function getConfigOptions(apiUrl: string, connectionString: string) {
    return axios.get<IAbexConfig[]>(
        apiUrl + '/get-config-options',
        { headers: { storageConnectionString: connectionString } }
    )
}

export function GetConfigOptionsActionCreator(apiUrl: string) {

    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        return getConfigOptions(apiUrl, connectionString).then(
            (res) => (res.data))
            .then((config_options) =>
                dispatch({
                    type: GET_CONFIGS_SUCCESS,
                    payload: {
                        config_options: config_options
                    }
                }),
                (error) =>
                    dispatch({
                    type: GET_CONFIGS_FAIL,

                }),
            );
    };
}

/// UPLOAD CONFIG ///
function uploadConfig(apiUrl: string, connectionString: string, uploadedConfig: File, binary: string) {
    const data = new FormData()
    data.append('file', uploadedConfig)

    const response = axios.post(
        apiUrl + '/upload-config-data',
        data,
        {
            headers: {
                storageConnectionString: connectionString,
                fileName: uploadedConfig.name || "",
                // config: binary
            }
        }
    )
    console.log('response from upoad config: ', response)
    return response

}

export function UploadConfigActionCreator(apiUrl: string, formData: FormData) {
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        const uploadedConfig = formData.get('uploadConfig') as File
        console.log('config in action: ', uploadedConfig)

        getFileFromInput(uploadedConfig)
            .then(binary => {
                console.log('returning binary', binary)
                console.log('returning config', uploadedConfig )
                return uploadConfig(apiUrl, connectionString, uploadedConfig, binary)
                    .then((response) =>
                        dispatch({
                            type: UPLOAD_CONFIG_SUCCESS,
                            payload: response.data
                        }),
                        (error) => dispatch({
                            type: UPLOAD_CONFIG_FAIL,
                            error: error
                        }),
                    );
            })

        
    };
}