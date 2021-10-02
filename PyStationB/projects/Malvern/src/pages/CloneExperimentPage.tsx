import axios from "axios";
import React from "react"
import { IExperimentResult, IPyBCKGExperiment } from "../components/Interfaces";
import { IFormState, ISubmitResult } from "../components/utils/Form";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IErrors, IFormValues } from "../components/utils/FormShared";
import { Container, Form } from "react-bootstrap";

interface IProps extends PropsFromRedux{
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;

}

class CloneExperimentPage extends React.Component<IProps, IFormState> {
    public constructor(props: IProps) {
        super(props)
        this.state = {
            values: {
                experimentOptions: [] ,
                selectedExperiment: {} as IPyBCKGExperiment,
            },
            errors: {},
            showButton: true,
            submitted: false,
            submitting: false
        }
    }

    componentDidMount() {
        this.getOptions()
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

    async getOptions() {
        this.props.getExperimentOptions(api_url)

        const experimentOptionsResult = this.props.getExperimentOptionsResult
        const experimentOptions = experimentOptionsResult?.experiment_options

        // TODO: update value of experiment_options in state?
        if (experimentOptions){
            this.setValue("experimentOptions", experimentOptions)
            this.setValue("selectedExperiment", experimentOptions[0])
        }
    }


    private changeExperimentFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedExpName = e.target.value;
        console.log(`Selected experiment name: ${selectedExpName}`)
        const experimentOptions: IPyBCKGExperiment[] = this.state.values.experimentOptions
        const selectedExperiment = experimentOptions.find(exp => exp.Name === selectedExpName)
        if (selectedExperiment) {
            console.log('selected experiment: ')
            console.log(selectedExperiment)
            //this.setState({ selectedExperiment: selectedExperiment })
            this.setValue("selectedExperiment", selectedExperiment)
        }
    }

    private errorsEncountered(errors: IErrors) {
        let errorEncountered = false;
        Object.keys(errors).map((key: string) => {
            if (errors[key].length > 0) {
                errorEncountered = true
            }
        });

        if (!this.state.values.selectedExperiment.name) {
            errorEncountered = true
        }
        return errorEncountered;
    }

    public render() {
        if (!this.props.connection?.connected) {
            return (
                <Container fluid={true}>
                    <h3>You are not connected. Please go to log in to create an experiment</h3>
                </Container>
            )
        } else {
            if (this.state.submitted) {
                return (
                    <Container fluid={true}>
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is ABC123.</p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                    </Container>
                )
            } else {
                const experimentOptionsResult = this.props.getExperimentOptionsResult
                const experimentOptions: IPyBCKGExperiment[] = experimentOptionsResult?.experiment_options || this.state.values.experiment_options || []
                

                return (
                    <Container fluid={true}>
                        <ExperimentHeader />
                        <h1>Repeat a previous experiment</h1>
                        <form onSubmit={this.handleSubmit} >
                            <Form.Label>Select an Experiment:</Form.Label>
                            <select value={this.state.values.selectedExperiment.Name} onChange={
                                (e) => this.changeExperimentFields(e)
                            }>
                                {experimentOptions.map((exp, id) =>
                                    <option key={'exp'+id} value={exp.Name}>
                                        {exp.Name}
                                    </option>)
                                }
                            </select>
                            <br />
                            <input
                                type="submit"
                            >
                            </input>
                        </form>
                    </Container>
                )
            }
        }
    }

    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        // TODO: Create CloneExperiment action and replace this method
        event.preventDefault();

        const response = axios.post<{ string: string }>(
            api_url + '/submit-clone-form',
            {
                'selectedExperiment': this.state.values.selectedExperiment,
            }
        ).then(
            response => {
                console.log('Response data')
                console.log(response.data)
                this.setState({ submitted: true })
                return { success: true }
            }
        )
        return { success: false }
    }
}

export default connector(CloneExperimentPage);