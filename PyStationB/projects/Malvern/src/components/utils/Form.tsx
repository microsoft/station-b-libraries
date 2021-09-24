import React from "react"
import {mergeStyleSets} from "@uifabric/merge-styles";
import { IValidationProp, Validator } from "./Validation";
import { IErrors, IFormValues } from "./FormShared";

const css = mergeStyleSets({
    experimentContainer:{
        display: "grid",
        backgroundColor: "#888888",
        },
    submit: {
        float: "left",
        //display: "block",
        margin: "20px",
        //border: "5px",
        //color:"#ffffff",
        backgroundColor: "#009966",
    },
    fields: {
        display: "grid",
        padding: "10px:",
    },
    form: {
        display: "block",
    }
 });

export interface IFormProps {
    /*  Form properties */
    defaultValues: IFormValues;
    validationRules: IValidationProp;
    showButton: boolean;
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
}

export const required: Validator = (fieldName:string, values: IFormValues, args?: any) : string =>
    values[fieldName] === undefined ||
    values[fieldName] === null ||
    values[fieldName] === ""
    ? "Please fill out this field"
    : "";


export interface IFormState {
    /* Submit form properties */
    values: IFormValues;
    success?: boolean;
    showButton?: boolean;
    errors: IErrors;
    submitting: boolean;
    submitted: boolean;
}

export interface IFormContext {
/* For passing context to fields */
    errors: IErrors;
    values: IFormValues;
    setValue?: (fieldName: string, value: any) => void;
    validate?: (fieldName: string, value: any) => void;
}

export interface ISubmitResult {
    /* Submit form result */
    success: boolean;
    errors?: IErrors;
}

export const FormContext = React.createContext<IFormContext>({
    values: {},
    errors: {}
});


export class Form extends React.Component<IFormProps, IFormState> {

    constructor(props: IFormProps) {
        super(props)
        const errors: IErrors = {};
        // Initialise every field with empty array for errors
        Object.keys(props.defaultValues).forEach(fieldName => {
            errors[fieldName] = [];
          });
        this.state = {
            values: props.defaultValues,
            errors,
            showButton: props.showButton,
            submitted: false,
            submitting: false
        };
    }

    private errorsEncountered(errors: IErrors) {
        let errorEncountered = false;
        Object.keys(errors).map((key: string) => {
            if (errors[key].length > 0) {
                errorEncountered = true;
            }
        });
        return errorEncountered; 
    }

    private handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        // This gets called first and does the form validation
        e.preventDefault();

        if (this.validateForm()) {
            this.setState({submitting: true});
            // console.log('point1');
            const result = await this.props.onSubmit(this.state.values);

            this.setState({ 
                //success: result.success,
                errors: result.errors || {},
                submitted: result.success,
                submitting: true 
            });
        }
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

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
                if (error){
                    errors.push(error);
                }
            });
        } else {
            // if single rule
            if (rules) {
                const error = rules.validator(fieldName, this.state.values, rules.args);
                if (error) {
                    errors.push(error);
            }}
        }

        // set errors from state
        const newErrors = {...this.state.errors, [fieldName]: errors};
        this.setState({errors: newErrors});
        return errors;
    }

    private validateForm(): boolean {
        const errors: IErrors = {};
        let haveError = false;
        Object.keys(this.props.defaultValues).forEach(fieldName => {
            errors[fieldName] = this.validate(
                fieldName,
                this.state.values[fieldName]
            );
            if (errors[fieldName].length > 0) {
                haveError = true;
            }
        });
        this.setState({errors});
        return !haveError;
    }

    public render() {
        const context: IFormContext = {
            errors: this.state.errors,
            values: this.state.values,
            setValue: this.setValue,
            validate: this.validate,
        };
        const { success, errors } = this.state;
        // console.log(`submitting: ${this.state.submitting}`)

        return (
            <FormContext.Provider value={context}>
                <form className={css.form} noValidate={true} onSubmit={ this.handleSubmit}>
                    <div className={css.fields}>
                        {this.props.children}
                    </div>
                    { this.state.showButton? 
                        <button 
                            className={css.submit} 
                            type="submit"
                            disabled={this.state.submitting || this.errorsEncountered(errors)}
                        >
                            Submit
                        </button>
                        : null
                    }

                </form>
            </FormContext.Provider>
        );
    }
}