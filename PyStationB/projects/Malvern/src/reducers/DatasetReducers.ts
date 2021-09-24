import { Reducer } from "redux"
import { GetDatasetsDispatchType,  GETTING_DATASETS, GET_DATASETS_FAIL, GET_DATASETS_SUCCESS, IGetDataResult, UploadDatasetDispatchType, UPLOADING_DATASET, UPLOAD_DATASET_FAIL, UPLOAD_DATASET_SUCCESS } from "../actions/DatasetActionTypes"
import { defaultUploadState, IUploadState } from "./reducerInterfaces"


/// GET STORED DATASET OPTIONS ///
export interface IGetDatasetsState {
    getting: boolean,
    getDatasetResult?: IGetDataResult
}

export const defaultDatasetsState: IGetDatasetsState = {
    getting: false
}

export const getDatasetReducer: Reducer<IGetDatasetsState, GetDatasetsDispatchType> = (state = defaultDatasetsState, action) => {
    switch (action.type) {
        case GET_DATASETS_FAIL:
            return state
        case GETTING_DATASETS:
            return {
                ...state,
                getting: true
            }
        case GET_DATASETS_SUCCESS:
            return {
                ...state,
                getting: false,
                getDatasetResult: action.payload
            }
        default:
            return state
    }
}


/// UPLOAD OBSERVATIONS ///

export const uploadDatasetReducer: Reducer<IUploadState, UploadDatasetDispatchType> = (state = defaultUploadState, action) => {
    switch (action.type) {
        case UPLOAD_DATASET_FAIL:
            console.log('upload dataset failed')
            return {
                ...state,
                error: action.payload
            }
        case UPLOADING_DATASET:
            return {
                ...state,
                uploading: true
            }
        case UPLOAD_DATASET_SUCCESS:
            return {
                ...state,
                uploading: false,
                error: null,
                filePath: action.payload
            }
        default:
            return state
    }
}
