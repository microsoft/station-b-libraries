import { connect, ConnectedProps } from "react-redux"
import { AnyAction, bindActionCreators, Dispatch } from "redux"
import { ThunkDispatch } from "redux-thunk"
import { GetConfigOptionsActionCreator, UploadConfigActionCreator } from "../actions/ConfigActions"
import { ConnectToAzureStorage, DisconnectFromAzureStorage } from "../actions/ConnectionActions"
import { GetDatasetOptionsActionCreator, UploadObservationsActionCreator } from "../actions/DatasetActions"
import { GetAMLRunIdsActionCreator, GetExperimentOptionsActionCreator, GetExperimentResultActionCreator, SubmitExperimentActionCreator } from "../actions/ExperimentActions"

import { IAppState } from "../reducers/RootReducer"


export const mapStateToProps = (state: IAppState) => {
    // Map Redux's state to props
    return {
        connection: state.connectionState.connection,
        loading: state.connectionState.loading,
        getDatasetResult: state.getDatasetState.getDatasetResult,
        getExperimentOptionsResult: state.getExperimentOptionsState.getExperimentOptionsResult,
        getConfigOptionsResult: state.getConfigOptionsState.getConfigsResult,
        getAMLRunIdsResult: state.getAMLRunIdsState.getAMLRunIdsResult,
        getExperimentResultResult: state.getExperimentResultState.getExperimentResultResult,
        submitNewExperimentResult: state.submitExperimentState.submitExperimentResponse,
        loggedIn: state.connectionState.connection?.connected,
        error: state.errorState.error,
        uploadConfigResult: state.uploadConfigState.filePath,
        uploadObservationsResult: state.uploadObservationsState.filePath
    }
}

export const mapDispatchToProps = (dispatch: ThunkDispatch<any, any, AnyAction>) => {
    // Map dispatching of action to props
    return bindActionCreators(
        {
            createStorageConnection: ConnectToAzureStorage,
            dropStorageConnection: DisconnectFromAzureStorage,
            getDatasetOptions: GetDatasetOptionsActionCreator,
            getExperimentOptions: GetExperimentOptionsActionCreator,
            getAMLRunIdOptions: GetAMLRunIdsActionCreator,
            getExperimentResult: GetExperimentResultActionCreator,
            getConfigOptions: GetConfigOptionsActionCreator,
            uploadConfig: UploadConfigActionCreator,
            uploadObservations: UploadObservationsActionCreator,
            submitNewExperiment: SubmitExperimentActionCreator
        },
        dispatch
    )
}

export const connector = connect(mapStateToProps, mapDispatchToProps)
export type PropsFromRedux = ConnectedProps<typeof connector>