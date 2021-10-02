import axios from "axios";
import React from "react"
import { IAbexConfig, IPyBCKGExperiment } from "../components/Interfaces";
import { IFormContext, IFormState, ISubmitResult } from "../components/utils/Form";
import UploadBox from "../components/utils/UploadBox";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IValidationProp } from "../components/utils/Validation";
import { IErrors, IFormValues } from "../components/utils/FormShared";
import { Container, Form } from "react-bootstrap";
import { isYaml } from "components/utils/validators";

interface IProps extends PropsFromRedux{
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    validationRules: IValidationProp;
    submitted: boolean
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
                experimentOptions: [],
                selectedExperiment: {} as IPyBCKGExperiment,
                selectedConfig: {} as IAbexConfig,
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
        this.props.getExperimentOptions(api_url)
        const experimentOptionsResult = this.props.getExperimentOptionsResult
        const experimentOptions = experimentOptionsResult?.experiment_options

        if (experimentOptions){
            this.setValue("experimentOptions", experimentOptions)
            this.setValue("selectedExperiment", experimentOptions[0])
        }

        this.props.getConfigOptions(api_url)
        const cfgRes = this.props.getConfigOptionsResult
        const config_options = cfgRes?.config_options
        console.log("Config options: ", config_options)        


        if (cfgRes){
            console.log("Updating config options")
            this.setValue("configOptions", config_options)

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

    private changeConfigFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedConfigName = e.target.value;
        const configOptions: IAbexConfig[] = this.state.values.configOptions
        const selectedConfig = configOptions.find(cfg => cfg.ConfigName === selectedConfigName)
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
                    <Container fluid={true}>
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is ABC123.</p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                    </Container>
                )
            } else {
                console.log("state: ", this.state)
                const experimentOptionsResult = this.props.getExperimentOptionsResult
                const experimentOptions: IPyBCKGExperiment[] = experimentOptionsResult?.experiment_options || this.state.values.experimentOptions
                
                const configOptionsResult = this.props.getConfigOptionsResult
                const configOptions: IAbexConfig[] = configOptionsResult?.config_options || this.state.values.configOptions
                return (
                    <Container fluid={true}>
                        <ExperimentHeader />
                        <h1>Start a new iteration for an existing track</h1>
                        <form onSubmit={this.handleSubmit} >
                            <Form.Label>Select an Experiment:</Form.Label>
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
                            <br />
                            <Form.Label>Select an existing config:</Form.Label>
                            <select value={this.state.values.selectedConfig.name} onChange={
                                (e) => this.changeConfigFields(e)
                            }>
                                {configOptions.map(cfg =>
                                    <option key={'conf'+cfg.ConfigName} value={cfg.ConfigName}>
                                        {cfg.ConfigName}
                                    </option>)
                                }
                            </select>
                            <br />
                            <Form.Label>OR Upload new config (.yml)</Form.Label>
                            <UploadBox
                            upload={this.uploadConfigFile}
                            defaultValues={{}}
                            expectedFileType='.yml'
                            validationRules={{ selectedFile: { validator: isYaml } }}
                        />
                            <Form.Label> Upload observation file (.csv)</Form.Label>
                            <UploadBox
                                upload={this.uploadObservationFile}
                                defaultValues={{}}
                                expectedFileType=".csv"
                                validationRules={{}}
                            />
                            <br />
                            <input
                                type="submit"
                                disabled={this.errorsEncountered(this.state.errors)}
                            >
                            </input>
                        </form>
                    </Container>
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

        this.props.uploadConfig(api_url, formData)

        if (this.props.error) {
            return {success: false}
        }
        return { success: true };

    };

    private uploadObservationFile = async (fileSelected: File): Promise<ISubmitResult> => {

        const formData = new FormData();

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadObservations",
                fileSelected,
                fileSelected.name
            );
        }

        this.props.uploadObservations(api_url, formData)

        if (this.props.error) {
            return { success: false }
        }
        return { success: true };

    };

 
    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        // TODO: create SubmitNewIterationAction and replace this method
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