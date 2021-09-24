import React from "react";
import { FormContext, IFormContext } from "./Form";
import "../../index.css"

interface IFieldProps {
    name: string; // the name of the field
    label: string; // text to display in the label
    type?: "Text" | "Select" | "TextArea";  // type of editor to display
    options?: string[]; // only applicable in Select inputs
}

export const ExperimentField: React.SFC<IFieldProps> = (props: IFieldProps) => {
    const { name, label, type, options } = props;
    const handleChange = (
        e:
            | React.ChangeEvent<HTMLInputElement>
            | React.ChangeEvent<HTMLTextAreaElement>
            | React.ChangeEvent<HTMLSelectElement>,
        context: IFormContext
    ) => {
        if (context.setValue) {
            context.setValue(props.name, e.currentTarget.value);
        }
    };

    // use blur event to validate field
    const handleBlur = (
        e: 
            | React.FocusEvent<HTMLInputElement>
            | React.FocusEvent<HTMLTextAreaElement>
            | React.FocusEvent<HTMLSelectElement>,
        context: IFormContext
    ) => {
        if (context.validate) {
            // check that the currentTarget holds a valid value for this field
            context.validate(props.name, e.currentTarget.value)
        }
    };

    return (
        <FormContext.Consumer>
            {context => (
                <div className="formContainer">
                    { label && <label className="fieldHeader" htmlFor={name}>{label}</label>}

                    {(type === "Text") && (
                        <input
                            type={type.toLowerCase()}
                            id={name}
                            value={context.values[name]}
                            onChange={e => handleChange(e, context)}
                            onBlur={e => handleBlur(e, context)}
                        />
                        )}

                    {(type === "Select") && (
                        <div className="formField">
                            <select
                                id={name}
                                name={name}
                                value={context.values[name]}
                                onChange={e => handleChange(e, context)}
                                onBlur={e => handleBlur(e, context)}
                            >
                                {options &&
                                    options.map(option => (
                                        <option key={option} value={option}>
                                            {option}
                                        </option>
                                    ))}
                            </select>
                        </div>
                    )}

                    {type === "TextArea" && (
                        <textarea
                            id={name}
                            value={context.values[name]}
                            onChange={e => handleChange(e, context)}
                            onBlur={e => handleBlur(e, context)}
                        />
                    )}

                    {context.errors[name] &&
                        context.errors[name].length > 0 &&
                        context.errors[name].map(error => (
                            <span key={error} className="formError">
                                {error}
                            </span>
                        ))}

                </div>
            )}
        </FormContext.Consumer>
    );
};

ExperimentField.defaultProps = {
    type: "Text"
};
