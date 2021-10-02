import { Reducer } from "redux"
import {
    GetAMLRunIdsDispatchType,
    GetExperimentOptionsDispatchType, GetExperimentResultDispatchType, GETTING_AML_RUNIDS, GETTING_EXPERIMENT_OPTIONS, GETTING_EXPERIMENT_RESULTS,
    GET_AML_RUNIDS_FAIL,
    GET_AML_RUNIDS_SUCCESS,
    GET_EXPERIMENT_OPTIONS_FAIL, GET_EXPERIMENT_OPTIONS_SUCCESS, GET_EXPERIMENT_RESULTS_FAIL, GET_EXPERIMENT_RESULTS_SUCCESS,
    SubmitExperimentDispatchType, SUBMITTING_EXPERIMENT, SUBMIT_EXPERIMENT_FAIL, SUBMIT_EXPERIMENT_SUCCESS
} from "../actions/ExperimentActionTypes"
import { defaultAMLRunIdsState, defaultExperimentOptionsState, defaultExperimentResultState, defaultSubmitExperimentState, IGetAMLRunIdsState, IGetExperimentOptionsState, IGetExperimentResultState, ISubmitExperimentState } from "./reducerInterfaces"


/// GET EXPERIMENT OPTIONS ///

export const getExperimentOptionsReducer: Reducer<IGetExperimentOptionsState, GetExperimentOptionsDispatchType> = (state = defaultExperimentOptionsState, action) => {
    switch (action.type) {
        case GET_EXPERIMENT_OPTIONS_FAIL:
            return state
        case GETTING_EXPERIMENT_OPTIONS:
            return {
                ...state,
                getting: true
            }
        case GET_EXPERIMENT_OPTIONS_SUCCESS:
            return {
                ...state,
                getExperimentOptionsResult: action.payload
            }
        default:
            return state
    }
}

/// GET AML RUN IDS ///

export const getAMLRunIdsReducer: Reducer<IGetAMLRunIdsState, GetAMLRunIdsDispatchType> = (state = defaultAMLRunIdsState, action) => {
    switch (action.type) {
        case GET_AML_RUNIDS_FAIL:
            return state
        case GETTING_AML_RUNIDS:
            return {
                ...state,
                getting: true
            }
        case GET_AML_RUNIDS_SUCCESS:
            return {
                ...state,
                getAMLRunIdsResult: action.payload
            }
        default:
            return state

    }
}


/// GET EXPERIMENT RESULTS ///

export const getExperimentResultReducer: Reducer<IGetExperimentResultState, GetExperimentResultDispatchType> = (state = defaultExperimentResultState, action) => {
    switch (action.type) {
        case GET_EXPERIMENT_RESULTS_FAIL:
            return state
        case GETTING_EXPERIMENT_RESULTS:
            return {
                ...state,
                getting: true
            }
        case GET_EXPERIMENT_RESULTS_SUCCESS:
            return {
                ...state,
                getExperimentResultResult: action.payload
            }
        default:
            return state
    }
}

// SUBMIT NEW EXPERIMENT //

export const submitExperimentReducer: Reducer<ISubmitExperimentState, SubmitExperimentDispatchType> = (state = defaultSubmitExperimentState, action) => {
    switch (action.type) {
        case SUBMIT_EXPERIMENT_FAIL:
            return {
                ...state,
                error: action.payload
            }
        case SUBMITTING_EXPERIMENT:
            return {
                ...state,
                submitting: true
            }
        case SUBMIT_EXPERIMENT_SUCCESS:
            return {
                ...state,
                submitting: false,
                submitExperimentResponse: action.payload
            }
        default:
            return state
    }
}