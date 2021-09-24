import { IFormValues } from "./FormShared";

//export type RequiredValidator = (
//    fieldName: string,
//    values: IFormValues,
//    args?: any
//) => string;


export type Validator = (
    fieldName: string,
    values: IFormValues,
    args?: any
) => string;

export interface IValidation {
    validator: Validator;
    args?: any;
}

export interface IValidationProp {
    [key: string]: IValidation | IValidation[];
}
