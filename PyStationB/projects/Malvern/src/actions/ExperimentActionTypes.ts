import { IExperimentResult, IPyBCKGExperiment, IAMLRun } from "../components/Interfaces"

/// GET EXPERIMENT OPTIONS ///

export const GETTING_EXPERIMENT_OPTIONS = "GETTING_EXPERIMENTS"
export const GET_EXPERIMENT_OPTIONS_FAIL = "GET_EXPERIMENTS_FAIL"
export const GET_EXPERIMENT_OPTIONS_SUCCESS = "GET_EXPERIMENTS_SUCCESS"

export interface IGetExperimentOptionsResult {
    experiment_options?: IPyBCKGExperiment[]
}

export interface GettingExperiments {
    type: typeof GETTING_EXPERIMENT_OPTIONS
}

export interface GetExperimentsFail {
    type: typeof GET_EXPERIMENT_OPTIONS_FAIL
}

export interface GetExperimentsSuccess {
    type: typeof GET_EXPERIMENT_OPTIONS_SUCCESS,
    payload: {
        experiment_options: IPyBCKGExperiment[]
    }
}

export type GetExperimentOptionsDispatchType = GettingExperiments | GetExperimentsFail | GetExperimentsSuccess;


/// GET AML RUN IDS ///

export const GETTING_AML_RUNIDS = "GETTING_AML_RUNIDS"
export const GET_AML_RUNIDS_SUCCESS = "GET_AML_RUNIDS_SUCCESS"
export const GET_AML_RUNIDS_FAIL = "GET_AML_RUNIDS_FAIL"

export interface IGetAMLRunIdsResult {
    aml_run_ids?: IAMLRun[]
}

export interface GettingAMLRunIds {
    type: typeof GETTING_AML_RUNIDS
}

export interface GetAMLRunIdsSuccess {
    type: typeof GET_AML_RUNIDS_SUCCESS,
    payload: {
        aml_run_ids: IAMLRun[]
    }
}

export interface GetAMLRunIdsFail {
    type: typeof GET_AML_RUNIDS_FAIL
}

export type GetAMLRunIdsDispatchType = GettingAMLRunIds | GetAMLRunIdsSuccess | GetAMLRunIdsFail;


/// GET EXPERIMENT RESULTS ///

export const GETTING_EXPERIMENT_RESULTS = "GETTING_EXPERIMENT_RESULTS"
export const GET_EXPERIMENT_RESULTS_FAIL = "GET_EXPERIMENT_RESULTS_FAIL"
export const GET_EXPERIMENT_RESULTS_SUCCESS = "GET_EXPERIMENT_RESULTS_SUCCESS"

export interface IGetExperimentResultResult {
    experiment_result?: IExperimentResult
}

export interface IGettingExperimentResults {
    type: typeof GETTING_EXPERIMENT_RESULTS
}

export interface IGetExperimentResultSuccess {
    type: typeof GET_EXPERIMENT_RESULTS_SUCCESS,
    payload: {
        experiment_result: IExperimentResult
    }
}

export interface IGetExperimentResultFail {
    type: typeof GET_EXPERIMENT_RESULTS_FAIL
}

export type GetExperimentResultDispatchType = IGettingExperimentResults | IGetExperimentResultSuccess | IGetExperimentResultFail;

/// SUBMIT NEW EXPERIMENT ///

export const SUBMITTING_EXPERIMENT = "SUBMITTING_EXPERIMENT"
export const SUBMIT_EXPERIMENT_FAIL = "SUBMIT_EXPERIMENT_FAIL"
export const SUBMIT_EXPERIMENT_SUCCESS = "SUBMIT_EXPERIMENT_SUCCESS"

export interface ISubmitExperimentResult {
    configPath: string,
    observationsPath: string
}

export interface ISubmittingExperiment {
    type: typeof SUBMITTING_EXPERIMENT
}

export interface ISubmitExperimentSuccess {
    type: typeof SUBMIT_EXPERIMENT_SUCCESS,
}

export interface ISubmitExperimentFail {
    type: typeof SUBMIT_EXPERIMENT_FAIL,
    payload: {
        response: any
    }
}

export type SubmitExperimentDispatchType = ISubmittingExperiment | ISubmitExperimentSuccess | ISubmitExperimentFail