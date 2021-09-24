import axios from "axios";
import React from "react"
import { IPyBCKGExperiment } from "../components/Interfaces";
import {  IFormContext, IFormState, ISubmitResult } from "../components/utils/Form";
import UploadBox from "../components/utils/UploadBox";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IValidationProp } from "../components/utils/Validation";
import { IErrors, IFormValues } from "../components/utils/FormShared";

interface IProps extends PropsFromRedux{
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    validationRules: IValidationProp;
    submitted: boolean
}

interface IConfig {
    id: string,
    name: string
}

interface IObservationFile {
    id: string, 
    name: string,
    dateUploaded: string
}


export interface IIterationFormContext {
    /* For passing context to fields */
    errors: IErrors;
    values: IFormValues;
    setValue?: (fieldName: string, value: any) => void;
    validate?: (fieldName: string, value: any) => void;
}

export const IterationFormContext = React.createContext<IFormContext>({
    values: {},
    errors: {}
});

export const IIteraitionFormContext = React.createContext<IFormContext>({
    values: {},
    errors: {}
});

class NewIterationPage extends React.Component<IProps, IFormState> {
    public constructor(props: IProps) {
        super(props)
        this.state = {
            values: {
                selectedIteration: "",
                iterationOptions: [""],
                experimentOptions: [{}] as IPyBCKGExperiment[],
                selectedExperiment: {} as IPyBCKGExperiment,
                selectedConfig: {} as IConfig,
                uploadedCSV: {} as IObservationFile,
                configOptions: []
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

    async getOptions() {
        //const expRes = await axios.get<IPyBCKGExperiment[]>(api_url + '/get-experiment-options')
        //const experiments = expRes.data
        this.props.getExperimentOptions(api_url)

        const experimentOptions = this.props.getExperimentOptionsResult
        const experiment_options = experimentOptions?.experiment_options || this.state.values.experimentOptions

        // TODO: replace with call in  middleware
        const cfgRes = await axios.get<IConfig[]>(api_url + '/get-config-options')
        const configs = cfgRes.data

        this.setValue("experimentOptions", experiment_options)
        this.setValue("selectedExperiment", experiment_options[0])
        this.setValue("configOptions", configs)
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

    private changeConfigFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedConfigName = e.target.value;
        const configOptions: IConfig[] = this.state.values.configOptions
        const selectedConfig = configOptions.find(cfg => cfg.name === selectedConfigName)
        if (selectedConfig) {
            console.log('selected config: ')
            console.log(selectedConfig)
            this.setValue("selectedConfig", selectedConfig)
        }
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

    private errorsEncountered(errors: IErrors) {
        let errorEncountered = false;
        Object.keys(errors).map((key: string) => {
            if (errors[key].length > 0) {
                errorEncountered = true
            }
        });

        if (!this.state.values.selectedConfig.name || !this.state.values.uploadedCSV.name) {
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
            const context: IFormContext = {
                errors: this.state.errors,
                values: this.state.values,
                setValue: this.setValue,
                validate: this.validate,
            };
            if (this.state.submitted) {
                return (
                    <div className="formContainer">
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is ABC123.</p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                    </div>
                )
            } else {
                const experimentOptions: IPyBCKGExperiment[] = this.state.values.experimentOptions
                const configOptions: IConfig[] = this.state.values.configOptions
                return (
                    <div className="pageContainer">
                        <ExperimentHeader />
                        <h1>Start a new iteration for an existing track</h1>
                        <form onSubmit={this.handleSubmit} >
                            <div className="formField">
                                <label>
                                    <h3>Select an Experiment:</h3>
                                    <div className="formContainer">
                                        <select
                                            value={this.state.values.selectedExperiment.name}
                                            onChange={(e) => this.changeExperimentFields(e)}
                                        >
                                            {experimentOptions.map((exp, id) =>
                                                <option key={'exp'+id} value={exp.Name}>
                                                    {exp.Name}
                                                </option>)
                                            }
                                        </select>
                                    </div>
                                </label>
                            </div>
                            <div className="formContainer">
                                <div className="formField">
                                    <h2> Select or upload config </h2>
                                    <label>
                                        <h3>Select an existing config:</h3>
                                        <div className="formContainer">
                                            <select value={this.state.values.selectedConfig.name} onChange={
                                                (e) => this.changeConfigFields(e)
                                            }>
                                                {configOptions.map(cfg =>
                                                    <option key={cfg.id} value={cfg.name}>
                                                        {cfg.name}
                                                    </option>)
                                                }
                                            </select>
                                        </div>
                                    </label>
                                </div>
                                <div className="formField">
                                    <h3> Upload new config (.yml)</h3>
                                    <UploadBox
                                        upload={this.uploadConfigFile}
                                        defaultValues={{}}
                                        expectedFileType=".yml"
                                        validationRules={{}}
                                    />
                                </div>
                                <div className="formField">
                                    <h3> Upload observation file (.csv)</h3>
                                    <UploadBox
                                        upload={this.uploadObservationFile}
                                        defaultValues={{}}
                                        expectedFileType=".csv"
                                        validationRules={{}}
                                    />
                                </div>
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

    private validate = (fieldName: string, value: any): string[] => {
        const rules = this.props.validationRules[fieldName];
        const errors: string[] = [];

        // validate that config is either selected or uploaded
        let error = ""

        if (!this.state.values.selected) {
            console.log('No config selected')
            error = "You must either select config from the dropdown, or upload (remember to click the upload button)"
        }

        if (error ) {
            errors.push(error);
        }

        // if multiple rules for a field
        if (Array.isArray(rules)) {
            rules.forEach(rule => {
                const error = rule.validator(
                    fieldName,
                    this.state.values,
                    rule.args
                );
                if (error) {
                    errors.push(error);
                }
            });
        } else {
            // if single rule
            if (rules) {
                const error = rules.validator(fieldName, this.state.values, rules.args);
                if (error) {
                    errors.push(error);
                }
            }
        }

        // set errors from state
        const newErrors = { ...this.state.errors, [fieldName]: errors };
        this.setState({ errors: newErrors });
        return errors;
    }

    private uploadConfigFile = async (fileSelected: File): Promise<ISubmitResult> => {

        const formData = new FormData();

        if (fileSelected && fileSelected.name) {
            formData.append(
                "selectedConfig",
                fileSelected,
                fileSelected.name
            );
        }

        const response = axios.post(api_url + "/upload-config-data", formData
        ).then(res => {
            // Set value of 'config' in state
            const newConfig = {'name': fileSelected.name}
            this.setValue('selectedConfig', newConfig)
            return { success: true };
        });


        return { success: false };

    };

    private uploadObservationFile = async (fileSelected: File): Promise<ISubmitResult> => {

        const formData = new FormData();

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadedCSV",
                fileSelected,
                fileSelected.name
            );
        }

        const response = axios.post(api_url + "/upload-observation-data", formData
        ).then(res => {
            const newCSV = { 'name': fileSelected.name }
            this.setValue('uploadedCSV', newCSV)
            return { success: true };
        });


        return { success: false };

    };

 
    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const response = axios.post<{ string: string }>(
            api_url + '/submit-iteration-form',
            {
                'selectedExperiment': this.state.values.selectedExperiment,
                'selectedConfig': this.state.values.selectedConfig,
            }
        ).then(
            response => {
                console.log('Response data')
                console.log(response.data)
                this.setState({ submitted: true })
                return { success: true }
            }
        )
        return {success: false}
    }

}

export default connector(NewIterationPage);