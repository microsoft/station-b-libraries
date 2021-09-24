import { IAbexConfig } from "../components/Interfaces"


/// GET CONFIG OPTIONS /// 
export const GETTING_CONFIGS = "GETTING_CONFIGS"
export const GET_CONFIGS_SUCCESS = "GET_CONFIGS_SUCCESS"
export const GET_CONFIGS_FAIL = "GET_CONFIGS_FAIL"

export interface IGetConfigsResult {
    config_options?: IAbexConfig[]
}

export interface IGettingConfigs {
    type: typeof GETTING_CONFIGS
}

export interface IGetConfigsSuccess {
    type: typeof GET_CONFIGS_SUCCESS
    payload: {
        config_options: IAbexConfig[]
    }
}

export interface IGetConfigsFail {
    type: typeof GET_CONFIGS_FAIL
}

export type GetConfigsDispatchType = IGettingConfigs | IGetConfigsSuccess | IGetConfigsFail


/// UPLOAD CONFIG /// 
export const UPLOADING_CONFIG = "UPLOADING_CONFIG"
export const UPLOAD_CONFIG_FAIL = "UPLOAD_CONFIG_FAIL"
export const UPLOAD_CONFIG_SUCCESS = "UPLOAD_CONFIG_SUCCESS"

export interface IUploadingConfig {
    type: typeof UPLOADING_CONFIG
}

export interface IUploadConfigSuccess {
    type: typeof UPLOAD_CONFIG_SUCCESS
    payload: {
        response: any
    }
}

export interface IUploadConfigFail {
    type: typeof UPLOAD_CONFIG_FAIL
    payload: {
        response: any
    }
}

export type UploadConfigDispatchType = IUploadingConfig | IUploadConfigSuccess | IUploadConfigFail