import axios from "axios";
import React from "react"
import { IExperimentResult, IPyBCKGExperiment } from "../components/Interfaces";
import { IFormState, ISubmitResult } from "../components/utils/Form";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IErrors, IFormValues } from "../components/utils/FormShared";

//interface ICloneState {
//    experimentOptions: IPyBCKGExperiment[]
//    selectedExperiment: IPyBCKGExperiment
//}

interface IProps extends PropsFromRedux{
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;

}

class CloneExperimentPage extends React.Component<IProps, IFormState> {
    public constructor(props: IProps) {
        super(props)
        this.state = {
            values: {
                experimentOptions: [{}] as IPyBCKGExperiment[],
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

        const experimentOptions = this.props.getExperimentOptionsResult
        const experiment_options = experimentOptions?.experiment_options || this.state.values.experimentOptions

        // TODO: update value of experiment_options in state?

        this.setValue("experimentOptions", experiment_options)
        this.setValue("selectedExperiment", experiment_options[0])
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
                <div>
                    <h3>You are not connected. Please go to log in to create an experiment</h3>
                </div>
            )
        } else {
            const experimentOptions: IPyBCKGExperiment[] = this.state.values.experimentOptions
            if (this.state.submitted) {
                return (
                    <div className="formContainer">
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is ABC123.</p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                    </div>
                )
            } else {
                return (
                    <div className="pageContainer">
                        <ExperimentHeader />
                        <h1>Repeat a previous experiment</h1>
                        <form onSubmit={this.handleSubmit} >
                            <div className="formField">
                                <label>
                                    <h3>Select an Experiment:</h3>
                                    <div className="formContainer">
                                        <select value={this.state.values.selectedExperiment.name} onChange={
                                            (e) => this.changeExperimentFields(e)
                                        }>
                                            {experimentOptions.map((exp, id) =>
                                                <option key={'exp'+id} value={exp.Name}>
                                                    {exp.Name}
                                                </option>)
                                            }
                                        </select>
                                    </div>
                                </label>
                            </div>
                            <div className="formField">
                                <input
                                    type="submit"
                                    disabled={this.errorsEncountered(this.state.errors)}
                                >
                                </input>
                            </div>
                        </form>
                    </div>
                )
            }
        }
    }

    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
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