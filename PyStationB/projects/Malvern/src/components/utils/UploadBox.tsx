import React from "react"
import { mergeStyleSets } from "@uifabric/merge-styles";
import { IterationFormContext } from "../../pages/NewIterationPage";
import { IFormContext, IFormState, ISubmitResult} from "./Form";
import { IErrors, IFormValues } from "./FormShared";
import { IValidationProp } from "./Validation";

const css = mergeStyleSets({
    uploaderContainer:{
      display: "grid",
      width: "40%",
      borderStyle: "ridge",
      backgroundColor: "#dddddddd",
    }
});

interface IUploaderProps {
    defaultValues: IFormValues;
    validationRules: IValidationProp;
    upload: (file: File) => Promise<ISubmitResult>
    expectedFileType: any
}

export class UploadBox extends React.Component<IUploaderProps, IFormState> {
    constructor(props: IUploaderProps) {
        super(props)
        const errors: IErrors = {};
        // Initialise every field with empty array for errors
        Object.keys(props.defaultValues).forEach(fieldName => {
            errors[fieldName] = [];
        });
        this.state = {
            values: props.defaultValues,
            errors,
            submitted: false,
            submitting: false
        };
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

    private onFileChange = (event: React.FormEvent<HTMLInputElement>) => {
        event.preventDefault()
        const targetFiles = event.currentTarget.files;
        console.log('changing target files to ', targetFiles)
        
        if (event.currentTarget.files && event.currentTarget.files[0]) {
            this.setValue("selectedFile", event.currentTarget.files[0]);
        }

    };

    private onFileUpload = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        console.log('target file: ', this.state.values.selectedFile)

        const formData = new FormData();
        const fileSelected = this.state.values.selectedFile;

        if (fileSelected && fileSelected.name) {
            formData.append(
                "file",
                fileSelected
                //fileSelected.name
            );
        }
        this.props.upload(fileSelected)
    };


    private validate = (fieldName: string, value: any): string[] => {
        const rules = this.props.validationRules[fieldName];
        const errors: string[] = [];

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


    private fileData = () => {
        if (this.state.values.selectedFile && this.state.values.selectedFile.name) {
        return (
            <div className={css.uploaderContainer}>
                <div>
                    <h2>File Details:</h2>
                    <p>File Name: {this.state.values.selectedFile.name}</p>
                    <p>File Type: {this.state.values.selectedFile.type}</p>
                </div>

            </div>
        );
        } else {
        return (
            null
        );
        }
    };

    public render() {
        const context: IFormContext = {
            errors: this.state.errors,
            values: this.state.values,
            setValue: this.setValue,
            validate: this.validate,
        };

        return (
            <IterationFormContext.Provider value={context}>
                <div className={css.uploaderContainer}>
                    <div>
                        <form onSubmit={this.onFileUpload}>
                            <input type="file" accept={this.props.expectedFileType} onChange={this.onFileChange} />
                            {/*<button onClick={this.onFileUpload}>*/}
                            {/*    Upload*/}
                            {/*</button>*/}
                            <input type="submit" />
                        </form>
                    </div>
                    {this.fileData()}
                </div>
            </IterationFormContext.Provider>
      );
    }

    }

export default UploadBox;


