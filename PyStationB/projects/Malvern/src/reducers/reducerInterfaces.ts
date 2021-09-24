export interface IUploadState {
    uploading: boolean,
    error?: any,
    filePath?: any
}

export const defaultUploadState: IUploadState = {
    uploading: false,
}