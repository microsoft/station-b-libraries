export interface ISample{
    id: number,
    experimentId: number
    metaType: string,
    physicalPlateName: string, 
    physicalWellCol: number, 
    physicalWellRow: number,
    virtualWellCol: number,
    virtualWellRow: number
}

export interface ISignal{
    id: number,
    experimentId: number,
    timestamp: string,
    type: string,
    emission: number,
    excitation: number,
    gain: number,
    wavelength: number
}

export interface IPyBCKGExperiment {
    PartitionKey: string,
    RowKey: string, 
    Timestamp: string, 
    Name: string,
    Notes: string, 
    Type: string, 
    Deprecated: boolean
}

export interface IExperimentResult {
    id: number,
    name: string,
    description: string,
    samples: ISample[],
    signals: ISignal[],
    imageFolders: string[],
    imageNames: string[],
    iterations: string[],
    folds: string[]
    type: string,
    timestamp: string,
    deprecated: string,
    suggestedExperiments: Record<string, any>[]
}

export interface IAMLRun {
    PartitionKey?: string, 
    RowKey: string,
    Timestamp: string
}

export interface IConfig {
    id: string,
    name: string
}

export interface IDataset {
    id: string,
    name: string,
    dateCreated: string,
    dataRecords: unknown[],
    dataCols: unknown[]
}

export interface IAbexConfig {
    PartitionKey: string,
    RowKey: string,
    PathToBlob: string,
    ConfigName: string,
    Timestamp: string
}

export interface IAMLConfig {
    SubscriptionId: string,
    ResourceGroup: string,
    WorkspaceName: string,
    ComputeTarget: string
}

export interface IAMLRun {
    ExperimentName: string,
    RunId: string,
    RunUrl: string
}