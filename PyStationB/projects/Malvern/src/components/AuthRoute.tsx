import React from "react";
import { connect } from "react-redux";
import { Redirect, Route, RouteProps } from "react-router-dom";
import { connector } from "../store/connectors";


interface MyRouteProps extends RouteProps {
    component: any;
    authenticated: boolean;
    rest?: any
}

const AuthRoute: React.SFC<MyRouteProps> = ({ component: Component, authenticated, ...rest }: any) => (
    <Route
        {...rest}
        render={
            (props) => authenticated ?
                <Component {...props} /> :
                <Redirect to='/login' />
        }
    />
);


//const mapStateToProps = (state: any) => ({
//    authenticated: state.loggedIn
//});

export default connector(AuthRoute);
