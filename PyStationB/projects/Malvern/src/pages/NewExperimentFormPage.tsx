import React from "react"
import { ISubmitResult, IFormState } from "../components/utils/Form"
import UploadBox from "../components/utils/UploadBox";
import ExperimentHeader from "../components/RunExperiment/ExperimentTypeHeader";
import { IConfig } from "../components/Interfaces";
import { api_url } from "../components/utils/api";
import { isYaml } from "../components/utils/validators";
import { connector, PropsFromRedux } from "../store/connectors";
import { IErrors, IFormValues } from "../components/utils/FormShared";
import { IValidationProp } from "../components/utils/Validation";
import { Container, Form } from "react-bootstrap";

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
        const cfgRes = this.props.getConfigOptionsResult
        const configs = cfgRes?.config_options

        this.setValue("configOptions", configs)
    }

    private changeExperimentType = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedExpType = e.target.value;
        if (selectedExpType) {
            console.log('selected experiment type: ')
            console.log(selectedExpType)
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
            const submitNewExperimentResult = this.props.submitNewExperimentResult
            const amlRun = submitNewExperimentResult?.amlRun
            console.log("AML run: ", amlRun)
            if (amlRun != undefined) {
                const amlRunId = amlRun.RunId || ""
                const amlUrl = amlRun.RunUrl || ""
                return (
                    <Container fluid={true}>
                        <p> Experiment submitted.</p>
                        <p> Your unique experiment id is {amlRunId}.</p>
                        <p> See the status of your Run <a href={amlUrl} target="_blank" rel="noopener noreferrer"> Here</a></p>
                        <p>Please make a note of this and check the Previous Experiments tab in a few hours.</p>
                        <p>Or to start a new experiment, hit refresh.</p>
                    </Container>
                )
            } else {
                const configOptions: IConfig[] = this.state.values.configOptions
                return (
                    <Container fluid={true}>
                        
                        <ExperimentHeader />
                        <h1 className="header">Start a new experiment</h1>
                            {/* <form>
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
                            </form> */}
                        {/* <h3> or </h3> */}
                        <Form.Label>Upload new config (.yml)</Form.Label>
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
                        <Form.Label> Azure ML connection secrets (.yml)</Form.Label>
                        <UploadBox
                            upload={this.parseAMLSecrets}
                                defaultValues={{}}
                                expectedFileType=".yml"
                            validationRules={{ selectedFile: { validator: isYaml } }}
                        />
                    <br />
                    <form onSubmit={this.handleSubmit} >
                       
                        <div className="formContainer">
                                
                        {this.props.error &&
                                    <p className="error"> Error { this.props.error.response.status }: {this.props.error.response.data.reason} </p>
                        }        
                        </div>
                            <input
                                type="submit"
                                // disabled={this.props.error}
                            >
                            </input>
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
        

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadConfig",
                fileSelected,
                fileSelected.name
            );
        }

        console.log('file selected going into form data: ', fileSelected)
        this.props.uploadConfig(api_url, formData)

        if (this.props.error) {
            return {success: false}
        }
        return { success: true };

    };

    private parseAMLSecrets = async (fileSelected: File): Promise<ISubmitResult> => {
        // Parse AML secrets file to connect to AML in order to submit experiments
        const formData = new FormData();
        console.log('file selected going into form data: ', fileSelected)

        if (fileSelected && fileSelected.name) {
            formData.append(
                "uploadAMLSecrets",
                fileSelected,
                fileSelected.name
            );
        }

        this.props.parseAMLFile(api_url, formData)

        console.log('props after upload: ', this.props)

        if (this.props.error) {
            return {success: false}
        }
        return { success: true };

    }

    public handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        console.log("State in handleSubmit", this.state)
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
            return { success: false }
        } else {
            this.setState({'submitted': true})
            return { success: true }
        }
    }
}

export default connector(NewExperimentFormPage);