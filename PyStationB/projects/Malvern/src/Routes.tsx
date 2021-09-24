import React from "react"
import {BrowserRouter as Router, Redirect, Route, RouteComponentProps, Switch} from "react-router-dom";

import Header from "./components/utils/Header";
import PageNotFound from "./components/utils/PageNotFound"
//import ExperimentsPage from "./components/ExperimentSelectorPage";


import CloneExperimentPage from "./pages/CloneExperimentPage";

import MasterFormPage from "./pages/MasterFormPage";
import NewExperimentFormPage from "./pages/NewExperimentFormPage";
import ExperimentTypePage from "./components/RunExperiment/ExperimentTypeSelector";
import NewIterationPage from "./pages/NewIterationPage";
import ExperimentSelectorPage from "./pages/ViewExperimentResultsPage";
import Footer from "./components/utils/Footer";
import ExploreDataPage from "./pages/ExploreDataPage";
import LoginPage from "./pages/LoginPage";
import LogoutPage from "./pages/LogoutPage";


interface IState {
    loggedIn: boolean;
}

const RoutesWrap: React.FC = () => {
    return (
        <Router>
            <Route component={Routes} />
        </Router>
    );
}


class Routes extends React.Component<RouteComponentProps,  IState> {
    public constructor(props: RouteComponentProps) {
        super(props);
        this.state = {
            loggedIn:  false
        };
    }
    
    public render() {
        return (
            <div>
                <Router>
                    <Header loggedIn={this.state.loggedIn} />
                    <Switch>
                        <Redirect exact={true} from="/" to="/master-form" />
                        <Redirect exact={true} from="/new-experiment/reload" to="/new-experiment" />
                        <Redirect exact={true} from="/new-iteration/reload" to="/new-iteration" />
                        <Redirect exact={true} from="/clone-experiment/reload" to="/clone-experiment" />
                        <Redirect exact={true} from="/experiments/reload" to="/experiments" />

                        {/*<Route */}
                        {/*    path="/experiments"*/}
                        {/*    key={Date.now()}*/}
                        {/*>*/}
                        {/*    {this.state.loggedIn ? (*/}
                        {/*        <ExperimentSelectorPage />*/}
                        {/*) : (*/}
                        {/*    <Redirect to="/login" />*/}
                        {/*)}*/}
                        {/*</Route>*/}
                        <Route 
                            path="/login"
                        >
                            <LoginPage/> 
                        </Route>
                        <Route path="/logout" component={LogoutPage}/>
                        <Route
                            path="/experimentType"
                            component={ExperimentTypePage}
                        />
                        <Route
                            path="/new-experiment"
                            component={NewExperimentFormPage}
                        />
                        <Route
                            path="/new-iteration"
                            component={NewIterationPage}
                        />
                        <Route
                            path="/clone-experiment"
                            component={CloneExperimentPage}
                        />
                        <Route
                            path="/master-form"
                            component={MasterFormPage}
                        />
                        <Route
                            path="/explore-data"
                            component={ExploreDataPage}
                        />
                        <Route
                            path="/experiments"
                            component={ExperimentSelectorPage}
                        />

                        <Route path='/privacy' component={() => {
                            window.location.href = 'https://privacy.microsoft.com/en-US/data-privacy-notice';
                            return null;
                        }} />
                        <Route path="/404" component={PageNotFound} />
                        <Redirect to="/404" />
                    </Switch>
                    <Footer />
                </Router>
            </div>
        );
    }
}

export default RoutesWrap;


//return <Redirect to="/new-experiment-page-reload" />
//        } else if (experimentType == 'clone') {
//    return <Redirect to="/clone-experiment-reload" />
//} else if (experimentType == 'New experiment') {
//    return <Redirect to="/new-experiment-reload" />
