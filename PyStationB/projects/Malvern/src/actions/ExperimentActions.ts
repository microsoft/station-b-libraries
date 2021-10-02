import axios from "axios";
import { Dispatch } from "redux";
import { IExperimentResult, IPyBCKGExperiment, IAMLRun, IAMLConfig } from "../components/Interfaces";
import {GET_AML_RUNIDS_FAIL,GET_AML_RUNIDS_SUCCESS, GET_EXPERIMENT_OPTIONS_FAIL, GET_EXPERIMENT_OPTIONS_SUCCESS, GET_EXPERIMENT_RESULTS_FAIL,
     GET_EXPERIMENT_RESULTS_SUCCESS, SUBMIT_EXPERIMENT_FAIL, SUBMIT_EXPERIMENT_SUCCESS} from "./ExperimentActionTypes";
import { IAppState } from "../reducers/RootReducer";


/// GET EXPERIMENT OPTIONS ///
function getExperimentOptions(api_url: string, connectionString: string) {
    return axios.get<IPyBCKGExperiment[]>(
        api_url + '/get-experiment-options',
        { headers: {storageConnectionString: connectionString} }
    )
}

export function GetExperimentOptionsActionCreator(api_url: string) {
    /// Unlike a plain action, within an actioncreator we return a function - the "thunk".
    // When this function is passed to `dispatch`, the thunk middleware will intercept it,
    // and call it with `dispatch` and `getState` as arguments.
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        return getExperimentOptions(api_url, connectionString).then(
            (res) => (res.data))
            .then((experiments_options) =>
                dispatch({
                    type: GET_EXPERIMENT_OPTIONS_SUCCESS,
                    payload: {
                        experiment_options: experiments_options
                    }
                }),
                (error) => dispatch({
                    type: GET_EXPERIMENT_OPTIONS_FAIL,

                }),
            );
    };
}

/// GET AML RUN ID OPTIONS
export function getAMLRunIds(api_url: string, connectionString: string) {
    return axios.get<IAMLRun[]>(
        api_url + 'get-aml-runs',
        { headers: { storageConnectionString: connectionString } }
    )
}

export function GetAMLRunIdsActionCreator(api_url: string) {
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""
        return getAMLRunIds(api_url, connectionString).then(
            (res) => (res.data))
            .then((aml_run_ids) =>
                dispatch({
                    type: GET_AML_RUNIDS_SUCCESS,
                    payload: {
                        aml_run_ids: aml_run_ids
                    }
                }),
                (error) => dispatch({
                    type: GET_AML_RUNIDS_FAIL
                }),
            );
    }
}

/// GET EXPERIMENT RESULTS ///

export function getExperimentResultFromAPI(api_url: string, connectionString: string, selectedExperiment: IPyBCKGExperiment) {
    return axios.get<IExperimentResult>(
        api_url + '/get-experiment-result',
        {
            headers: { storageConnectionString: connectionString },
            params: { experimentName: selectedExperiment.Name }
        }
    )
}


export function GetExperimentResultActionCreator(api_url: string, selectedExperiment: IPyBCKGExperiment) {
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || ""

        return getExperimentResultFromAPI(api_url, connectionString, selectedExperiment).then(
            (res) => (res.data))
            .then((experiment_result) =>
                dispatch({
                    type: GET_EXPERIMENT_RESULTS_SUCCESS,
                    payload: {
                        experiment_result: experiment_result
                    }
                }),
                (error) => dispatch({
                    type: GET_EXPERIMENT_RESULTS_FAIL,
                }),
            );
    };
}

// SUBMIT NEW EXPERIMENT //
export function submitNewExperiment(api_url: string, connectionString: string, configPath:string,
    observationsPath:string, amlConfig: IAMLConfig) {
    
        const response = axios.post<IExperimentResult>(
            api_url + '/submit-new-experiment',
            {
                headers: {
                    storageConnectionString: connectionString,
                    configPath: configPath,
                    observationsPath: observationsPath,
                    amlConfig: amlConfig
                }
            }
        );
        console.log("Response from submit new experiment: ", response);
        return response;
}


export function SubmitExperimentActionCreator(api_url: string, formData: FormData) {
    return function (dispatch: Dispatch, getState: () => IAppState) {
        const connectionString = getState().connectionState.connection?.connectionString || "";

        const configPath = formData.get("configPath") as string
        const observationsPath = formData.get("observationsPath") as string
        const emptyAMLConfig: IAMLConfig = {SubscriptionId: "", ResourceGroup: "",  WorkspaceName: "", ComputeTarget: ""}
        const amlConfig = getState().amlConfigState.amlConfigResult?.aml_config || emptyAMLConfig

        return submitNewExperiment(api_url, connectionString, configPath, observationsPath, amlConfig).then(
            (res) => (res.data))
            .then((newExperimentResponse) =>
                dispatch({
                    type: SUBMIT_EXPERIMENT_SUCCESS,
                    payload: {
                        amlRun: newExperimentResponse
                    }
                }),
                (error) => dispatch({
                    type: SUBMIT_EXPERIMENT_FAIL,
                    payload:
                    {
                        error: error

                    }
                }),
            );
    };
}