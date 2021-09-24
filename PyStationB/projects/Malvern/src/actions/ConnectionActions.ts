import * as storage from "azure-storage"
import { ConnectDispatchType, CONNECTING, CONNECT_FAIL, CONNECT_SUCCESS, DISCONNECT_FAIL, DISCONNECT_SUCCESS,  IConnection, IConnectionAction } from "./ConnectionActionTypes"
import { Dispatch } from "redux"


export const ConnectToAzureStorage = (connectionString: string) => async (dispatch: Dispatch<ConnectDispatchType>) => {
    try {

        dispatch({
            type: CONNECTING
        })

        const attribs_raw: string[] = connectionString.split(';')
        const attribs: Record<string, string> = {}

        for (const attribute of attribs_raw) {
            const attribute_split = attribute.split(/=(.*)/, 2)  //account key may contain = after first one
            if (attribute_split.length == 2) {
                const [key, value] = attribute_split
                attribs[key] = value
            }
        }
        console.log('attribs')
        console.log(attribs)

        const tableConnection = storage.createTableService(connectionString)
        console.log('table connected')

        dispatch({
            type: CONNECT_SUCCESS,
            payload: {
                connected: true,
                connectionString: connectionString,
                tableService: tableConnection
            }
        })

    } catch (e) {
        console.warn(e)
        dispatch({
            type: CONNECT_FAIL
        })
    }
}

export const DisconnectFromAzureStorage = () => async (dispatch: Dispatch<ConnectDispatchType>) => {
    try {

        console.log('dropping table connection')

        dispatch({
            type: DISCONNECT_SUCCESS,
            payload: {
                connected: false,
                connectionString: "",
                tableService: null
            }
        })

    } catch (e) {
        console.warn(e)
        dispatch({
            type: DISCONNECT_FAIL
        })
    }
}
