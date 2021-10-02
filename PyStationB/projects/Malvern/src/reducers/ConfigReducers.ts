import { Reducer } from "redux"
import { GetConfigsDispatchType, GETTING_CONFIGS, GET_CONFIGS_FAIL, GET_CONFIGS_SUCCESS, UploadConfigDispatchType, UPLOAD_CONFIG_FAIL, UPLOADING_CONFIG, UPLOAD_CONFIG_SUCCESS } from "../actions/ConfigActionTypes"
import { defaultGetConfigState, defaultUploadState, IGetConfigOptionsState, IUploadState } from "./reducerInterfaces"

/// GET CONFIG OPTIONS 

export const getConfigOptionsReducer: Reducer<IGetConfigOptionsState, GetConfigsDispatchType> = (state = defaultGetConfigState, action) => {
    switch (action.type) {
        case GET_CONFIGS_FAIL:
            return state
        case GETTING_CONFIGS:
            return {
                ...state,
                getting: true
            }
        case GET_CONFIGS_SUCCESS:
            return {
                ...state,
                getting: false,
                getConfigsResult: action.payload
            }
        default:
            return state
    }
}

/// UPLOAD CONFIGS

export const uploadConfigReducer: Reducer<IUploadState, UploadConfigDispatchType> = (state = defaultUploadState, action) => {
    switch (action.type) {
        case UPLOAD_CONFIG_FAIL:
            console.log('upload config failed')
            return {
                ...state,
                error: action.payload 
                }
        case UPLOADING_CONFIG:
            return {
                ...state,
                uploading: true
            }
        case UPLOAD_CONFIG_SUCCESS:
            return {
                ...state, 
                uploading: false,
                error: null,
                filePath: action.payload // this is the request response at the moment
            }
        default:
            return state
    }
}