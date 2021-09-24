import { Reducer } from "redux"
import { ConnectDispatchType, CONNECTING, CONNECT_FAIL, CONNECT_SUCCESS, DISCONNECT_FAIL, DISCONNECT_SUCCESS, IConnection, IConnectionAction } from "../actions/ConnectionActionTypes"

export interface IConnectionState {
    loading: boolean, 
    connection?: IConnection
}

export const defaultConnectionState: IConnectionState = {
    loading: false,

}

export const connectionReducer: Reducer<IConnectionState, ConnectDispatchType> = (state = defaultConnectionState, action) => {
    console.log('Connection reducer getting called with action', action)
    switch (action.type) {
        case CONNECT_FAIL:
            return state
        case CONNECTING:
            return {
                ...state,
                loading: true
            }
        case CONNECT_SUCCESS:
            console.log(action.payload)
            return {
                ...state,
                loading: false,
                connection: action.payload
            }

        case DISCONNECT_FAIL:
            return state
        case DISCONNECT_SUCCESS:
            console.log(action.payload)
            return {
                ...state,
                loading: false,
                connection: action.payload
            }

        default:
            // note: this is not the full state, only the partial state related to some reducer.
            // The state is never changed
            return state;
    }
}

