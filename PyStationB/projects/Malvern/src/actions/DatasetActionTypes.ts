import { IDataset } from "../components/Interfaces"

/// DATASETS ///

export const GETTING_DATASETS = "GETTING_DATASETS"
export const GET_DATASETS_FAIL = "GET_DATASETS_FAIL"
export const GET_DATASETS_SUCCESS = "GET_DATASETS_SUCCESS"

export interface IGetDataResult {
    dataset_options?: IDataset[]
}


export interface GettingDatasets {
    type: typeof GETTING_DATASETS
}

export interface GetDatasetsFail {
    type: typeof GET_DATASETS_FAIL
}

export interface GetDatasetsSuccess {
    type: typeof GET_DATASETS_SUCCESS,
    payload: {
        dataset_options: IDataset[]
    }
}

export type GetDatasetsDispatchType = GettingDatasets | GetDatasetsFail | GetDatasetsSuccess


/// UPLOAD OBSERVATIONS ///
export const UPLOADING_DATASET = "UPLOADING_DATASET"
export const UPLOAD_DATASET_FAIL = "UPLOAD_DATASET_FAIL"
export const UPLOAD_DATASET_SUCCESS = "UPLOAD_DATASET_SUCCESS"


export interface IUploadingDataset {
    type: typeof UPLOADING_DATASET
}

export interface IUploadDatasetSuccess {
    type: typeof UPLOAD_DATASET_SUCCESS
    payload: {
        response: any
    }
}

export interface IUploadDatasetFail {
    type: typeof UPLOAD_DATASET_FAIL
    payload: {
        response: any
    }
}

export type UploadDatasetDispatchType = IUploadingDataset | IUploadDatasetSuccess | IUploadDatasetFail