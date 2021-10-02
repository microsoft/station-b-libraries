import { IGetConfigsResult } from "actions/ConfigActionTypes"
import { IAMLConnectionResult, IGetDataResult } from "actions/DatasetActionTypes"
import { IGetAMLRunIdsResult, IGetExperimentOptionsResult, IGetExperimentResultResult, ISubmitExperimentResult } from "actions/ExperimentActionTypes"


// GET EXISTING CONFIG OPTIONS
export interface IGetConfigOptionsState {
    getting: boolean,
    getConfigsResult?: IGetConfigsResult
}

export const defaultGetConfigState: IGetConfigOptionsState = {
    getting: false
}

// GET EXISTING DATASET OPTIONS
export interface IGetDatasetsState {
    getting: boolean,
    getDatasetResult?: IGetDataResult
}

export const defaultDatasetsState: IGetDatasetsState = {
    getting: false
}

// UPLOAD FILE 
export interface IUploadState {
    uploading: boolean,
    error?: any,
    filePath?: any
}

export const defaultUploadState: IUploadState = {
    uploading: false,
}

// PARSE AML SECRES FILE
export interface IParseAMLFileState {
    parsing: boolean,
    error?: any,
    amlConfigResult?: IAMLConnectionResult
}

export const defaultParseAMLFileState: IParseAMLFileState = {
    parsing: false
}


/// GET EXPERIMENT OPTIONS ///
export interface IGetExperimentOptionsState {
    getting: boolean,
    getExperimentOptionsResult?: IGetExperimentOptionsResult
}

export const defaultExperimentOptionsState: IGetExperimentOptionsState = {
    getting: false
}


/// GET AML RUN IDS ///
export interface IGetAMLRunIdsState {
    getting: boolean,
    getAMLRunIdsResult?: IGetAMLRunIdsResult
}

export const defaultAMLRunIdsState: IGetAMLRunIdsState =
{
    getting: false
}

/// GET EXPERIMENT RESULTS ///
export interface IGetExperimentResultState {
    getting: boolean,
    getExperimentResultResult?: IGetExperimentResultResult
}

export const defaultExperimentResultState: IGetExperimentResultState = {
    getting: false
}

// SUBMIT NEW EXPERIMENT
export interface ISubmitExperimentState {
    submitting: boolean,
    submitExperimentResponse?: ISubmitExperimentResult
}

export const defaultSubmitExperimentState: ISubmitExperimentState = {
    submitting: false
}