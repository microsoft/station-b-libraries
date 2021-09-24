import * as storage from "azure-storage"

// Define action types
export const CREATE_CONNECTION = "CREATE_CONNECTION"
export const DROP_CONNECTION = "DROP_CONNECTION"

export const CONNECTING = "CONNECTING"
export const CONNECT_SUCCESS = "CONNECT_SUCCESS"
export const CONNECT_FAIL = "CONNECT_FAIL"

export const DISCONNECT_SUCCESS = "DISCONNECT_SUCCESS"
export const DISCONNECT_FAIL = "DISCONNECT_FAIL"


export interface IConnection {
    connected: boolean,
    connectionString: string,
    tableService?: storage.TableService
}

export interface IConnectionAction {
    type: string;
    payload: IConnection;
}

export interface IConnecting {
    type: typeof CONNECTING
}

export interface IConnectSuccess {
    type: typeof CONNECT_SUCCESS,
    payload: {
        connectionString: string,
        connected: boolean, 
        tableService: storage.TableService
    }
}

export interface IDisconnectSuccess {
    type: typeof DISCONNECT_SUCCESS,
    payload: {
        connectionString: string, 
        connected: boolean
    }
}

export interface IConnectFail {
    type: typeof CONNECT_FAIL,

}

export interface IDisconnectFail {
    type: typeof DISCONNECT_FAIL,

}

export type ConnectDispatchType = IConnectSuccess | IConnectFail | IConnecting| IDisconnectSuccess | IDisconnectFail