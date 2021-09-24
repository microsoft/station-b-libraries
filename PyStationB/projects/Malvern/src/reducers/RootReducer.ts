import { combineReducers } from "redux";
import { getConfigOptionsReducer, IGetConfigOptionsState, uploadConfigReducer } from "./ConfigReducers";
import { connectionReducer, IConnectionState } from "./ConnectionReducer";
import { errorReducer, IErrorState } from "./ErrorReducer";
import { getDatasetReducer, IGetDatasetsState, uploadDatasetReducer } from "./DatasetReducers";
import { getAMLRunIdsReducer, getExperimentOptionsReducer, getExperimentResultReducer, IGetAMLRunIdsState, IGetExperimentOptionsState, IGetExperimentResultState, ISubmitExperimentState, submitExperimentReducer } from "./ExperimentsReducers";
import { IUploadState } from "./reducerInterfaces";

export interface IAppState {
    readonly connectionState: IConnectionState
    readonly getDatasetState: IGetDatasetsState
    readonly getConfigOptionsState: IGetConfigOptionsState
    readonly getExperimentOptionsState: IGetExperimentOptionsState
    readonly getAMLRunIdsState: IGetAMLRunIdsState
    readonly getExperimentResultState: IGetExperimentResultState
    readonly uploadConfigState: IUploadState
    readonly uploadObservationsState: IUploadState
    readonly errorState: IErrorState
    readonly submitExperimentState: ISubmitExperimentState
}

// combine all reducers into single object. Outputs state object with given keys
export const rootReducer = combineReducers<IAppState>({
    connectionState: connectionReducer,
    getDatasetState: getDatasetReducer,
    getConfigOptionsState: getConfigOptionsReducer,
    getExperimentOptionsState: getExperimentOptionsReducer,
    getAMLRunIdsState: getAMLRunIdsReducer,
    getExperimentResultState: getExperimentResultReducer,
    uploadConfigState: uploadConfigReducer,
    uploadObservationsState: uploadDatasetReducer,
    errorState: errorReducer,
    submitExperimentState: submitExperimentReducer
})
