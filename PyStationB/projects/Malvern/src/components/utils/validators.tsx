import { IFormValues } from "./FormShared"

export const isYaml = (fieldName: string, values: IFormValues): string => {
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
