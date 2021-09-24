import React from "react"
import { ISubmitResult, IFormState } from "../components/utils/Form"
import axios from "axios";
import UploadBox from "../components/utils/UploadBox";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import { IConfig } from "../components/Interfaces";
//import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IErrors, IFormValues } from "../components/utils/FormShared";
import { IValidationProp } from "../components/utils/Validation";
import { Container, Form, FormGroup } from "react-bootstrap";

interface IProps extends PropsFromRedux {
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    validationRules: IValidationProp;
    submitted: boolean
}

interface IObservationFile {
    id: string,
    name: string,
    dateUploaded: string
}

const isYaml = (fieldName: string, values: IFormValues): string => {
    if (values[fieldName]) {
        const selectedFile = values[fieldName]
        console.log('selected file: ', selectedFile)
        const fileType = selectedFile.type
        if (fileType == 'application/yml') {
            return ""
        } else {
            return "Please ensure file is .yml type"
        }
    }
    console.log('No value for fieldname found in ', values)
    return "Something went wrong - no file found"; 
}

class NewExperimentFormPage extends React.Component<IProps, IFormState> {

    public constructor(props: IProps) {
        super(props);
        this.state = {
            values: {
                selectedIteration: "",
                iterationOptions: [""],
                selectedExperimentType: "BayesOpt",
                selectedConfig: {} as IConfig,
                configOptions: [],
                uploadedCSV: {} as IObservationFile
            },
            errors: {},
            showButton: true,
            submitted: false,
            submitting: false
        };
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

    componentDidMount() {
        this.getOptions()
    }

    async getOptions() {
        this.props.getConfigOptions(api_url)
        const cfgRes = this.props.getConfigOptionsResult //await axios.get<IConfig[]>(api_url + '/get-config-options')
        const configs = cfgRes?.config_options

        this.setValue("configOptions", configs)
    }

    private changeExperimentType = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedExpType = e.target.value;
        if (selectedExpType) {
            console.log('selected experiment type: ')
            console.log(selectedExpType)
            //this.setState({ selectedExperiment: selectedExperiment })
            this.setValue("selectedExperimentType", selectedExpType)
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

            if (this.state.submitted) {
                return (
                    <div className="pageContainer">
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is ABC123.</p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                    </div>
                )
            } else {
                const configOptions: IConfig[] = this.state.values.configOptions
                return (
                    <Container fluid={true}>
                        
                        <ExperimentHeader />
                        <h1 className="header">Start a new iteration for an existing track</h1>
                            <form>
                            <label>
                                <h3>Select an existing config:</h3>
                                    <select
                                        value={this.state.values.selectedConfig.name}
                                        onChange={(e) => this.changeConfigFields(e)}
                                    >
                                        {configOptions.map( cfg =>
                                            <option key={cfg.id} value={cfg.name}>
                                                {cfg.name}
                                            </option> )
                                        }
                                    </select>
                            </label>
                            </form>
                        <h3> or </h3>
                        <Form.Label>Upload new config (.yml)</Form.Label>
                            <UploadBox
                                upload={this.uploadConfigFile}
                                defaultValues={{}}
                                expectedFileType='.yml'
                                validationRules={{ selectedFile: { validator: isYaml } }}
                            />
                    <div className="formField">
                        <h3> Upload observation file (.csv)</h3>
                        <UploadBox
                            upload={this.uploadObservationFile}
                                defaultValues={{}}
                                expectedFileType=".csv"
                            validationRules={{}}
                        />
                    </div>

                    <form onSubmit={this.handleSubmit} >
                        <div className="formField">
                            <label>
                                <h3>Select an Experiment type:</h3>
                                <div className="formContainer">
                                    <select
                                        value={this.state.values.selectedExperimentType.name}
                                        onChange={(e) => this.changeExperimentType(e)
                                        }>
                                        <option key="1" value="BayesOpt">BayesOpt </option>
                                        <option key="2" value="ZoomOpt">ZoomOpt </option>
                                    </select>
                                </div>
                            </label>
                        </div>
                        <div className="formContainer">
                                
                        {this.props.error &&
                                    <p className="error"> Error { this.props.error.response.status }: {this.props.error.response.data.reason} </p>
                        }        
                        </div>
                        <div className="formField">
                            <input
                                type="submit"
                                disabled={this.props.error}
                            >
                            </input>
                        </div>
                    </form>
                </Container>
                )
            }
        }
    }

    private uploadObservationFile = async (fileSelected: File): Promise<ISubmitResult> => {
        // Upload the csv to Azure Storage, and also to local memory, for submitting experiment
        const formData = new FormData();

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadObservations",
                fileSelected,
                fileSelected.name
            );
        }

        this.props.uploadObservations(api_url, formData)

        console.log('props after upload: ', this.props)

        if (this.props.error) {
            return { success: false }
        }
        return { success: true };
    }


    private uploadConfigFile = async (fileSelected: File): Promise<ISubmitResult> => {
         // Upload the .yml to Azure Storage, and also to local memory, for submitting experiment
        const formData = new FormData();
        console.log('file seelcted going into form data: ', fileSelected)

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadConfig",
                fileSelected,
                fileSelected.name
            );
        }

        //for (const key of formData.entries()) {
        //    console.log(key[0] + ', ' + key[1]);
        //}
        this.props.uploadConfig(api_url, formData)

        console.log('props after upload: ', this.props)

        if (this.props.error) {
            return {success: false}
        }
        return { success: true };

    };

    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        // Upload boxes are separate forms, so config and observations wont be in event target
        const filledform = event.currentTarget
        const selectedConfigPath = this.props.uploadConfigResult.filePath
        console.log('selected config Path: ', selectedConfigPath)

        const selectedObservationsPath = this.props.uploadObservationsResult.filePath
        console.log('selected observations Path: ', selectedObservationsPath)
        const formData = new FormData();

        // TODO: properties to get from formData: configname, observation file name, 

        if (selectedConfigPath) {
            formData.append(
                "configPath",
                selectedConfigPath
            );
        }
        if (selectedObservationsPath) {
            formData.append(
                "observationsPath",
                selectedObservationsPath
            )
        }

        this.props.submitNewExperiment(api_url, formData)

        if (this.props.error) {
            return { success: true }
        }
        return { success: false }
    }
}

export default connector(NewExperimentFormPage);