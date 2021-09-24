import * as React from "react";
import { IFormValues } from "../components/utils/FormShared";
import { PropsFromRedux, connector } from "../store/connectors";

export interface IProps {
    onSubmit: (values: IFormValues) => any;

}

interface ILoginState {
    errors: string[],
}

class LogoutPage extends React.Component<PropsFromRedux, ILoginState> {

    public render() {
        const connection = this.props.connection
        if (connection && connection?.connected) {
            return (
                <div className="pageContainer">
                    <h3>Are you sure you want to logout?</h3>

                    <div className="button">
                        <button id="logoutButton" onClick={this.handleDisconnect}>Log out</button>
                    </div>

                </div>
            )
        } else {
            return (
                <div>
                    <h3>You are not logged in. Please enter connection string below:</h3>
                    <form onSubmit={this.handleSubmit}>
                        <label>Enter connection string:</label>
                        <input type="text" name="connectionString" onChange={this.handleChange} />
                        <input type="submit" />
                    </ form>
                    <div className="error">
                    </div>
                </div>
            )
        }
    }

    private handleChange = () => {
        console.log('field changing')
    }

    private handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        const connectionString = event.currentTarget.connectionString.value

        console.log('props before createStorageConnection')
        console.log(this.props)

        this.props.createStorageConnection(connectionString)

        console.log('props after createStorageConnection')
        console.log(this.props)

        const connectionResult = this.props.connection
        if (connectionResult && this.props.connection?.connected) {
            console.log('connection succeeded')
            return ({ success: true })
        } else {
            console.log('connected is false')
            return ({ success: false })
        }
    }

    private handleDisconnect = (event: React.MouseEvent<HTMLButtonElement>) => {
        event.preventDefault();

        this.props.dropStorageConnection()

        console.log('disconnection succeeded')
        console.log(this.props)

        const connectionResult = this.props.connection
        if (connectionResult && this.props.connection?.connected) {
            console.log('disconnect failed')
            return ({ success: false })
        } else {
            console.log('connected is false')
            return ({ success: true })
        }

    }

}

export default connector(LogoutPage);
