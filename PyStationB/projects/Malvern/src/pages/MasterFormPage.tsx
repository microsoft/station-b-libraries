import React from "react"
import { RouteComponentProps } from "react-router-dom";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import { Container } from "react-bootstrap";

import "../index.css"

type Props = RouteComponentProps<{ id: string }>;

interface IState {
    loading: boolean;
}

class MasterFormPage extends React.Component<Props, IState> {
    public constructor(props: Props) {
        super(props)
        this.state = {
            loading: true
        }
    }

    private componentUnloaded = false;

    public async componentDidMount() {
        if (!this.componentUnloaded) {
            this.setState({loading: false})
        }
    }

    public componentWillUnmount() {
        this.componentUnloaded = true;
    }

    public render() {
        return (
            <Container fluid>
                <ExperimentHeader />
            </Container >
        )
        
    }
}

export default MasterFormPage;