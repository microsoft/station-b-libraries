import { mergeStyleSets } from "@fluentui/react";
import axios from "axios";
import React from "react"
import { IConfig, IPyBCKGExperiment } from "../Interfaces";
import { api_url } from "../utils/api";
import { ExperimentField } from "../utils/ExperimentFields";
import { Form, IFormContext, ISubmitResult } from "../utils/Form";
import { IFormValues } from "../utils/FormShared";

export interface IProps {
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    options: string[]
}

export interface IExperimentType {
    id: number,
    name: string
}

export interface IExperimentProps {
    onSubmit: (values: IFormValues) => unknown;
    options: IExperimentType[]
}

const css = mergeStyleSets({
    formField: {
        width: "60%",
        margin: "5px",
        padding: "5px",
    },
})

export const ExperimentTypeSelectorComponent: React.FC<IExperimentProps> = (props: IExperimentProps) => {
    const handleSubmit = async (values: IFormValues) => {
        // This gets called second
        // console.log('point 2');
        const result = await props.onSubmit(values);
        return result;
    };
    console.log(`Experiment type options: ${props.options.map( ex => ex.name) }`)
    return (
        <form>
            <select
                name={'experimentType'}
                onSubmit={handleSubmit}
            >
        
            <div className={css.formField}>
                {props.options.map(experimentType => 
                    <option key={experimentType.id} value={experimentType.name}>{experimentType.name}</option>
                )}
            </div>
            </select>  
        </form>


    );
};


export const ExperimentSelectorComponent: React.FC<IProps> = (props: IProps) => {
    const handleSubmit = async (values: IFormValues): Promise<ISubmitResult> => {
        const result = await props.onSubmit(values);
        return result;
    };

    return (
        <Form
            onSubmit={handleSubmit}
            defaultValues={{ experiment: "" }}
            validationRules={{}}
            showButton={false}
        >
            <div className={css.formField}>
                <ExperimentField
                    name="experiment"
                    label="Select an experiment"
                    type="Select"
                    options={props.options}
                />
            </div>
        </Form>
    );
};

export interface INewExperimentProps {
    //onSubmit: (values: IFormValues) => unknown;
    experimentOptions: IPyBCKGExperiment[];
    selectedExperiment: IPyBCKGExperiment;
    //getOptions: () => unknown;
}


interface INewExperimentState {
    experimentOptions: IPyBCKGExperiment[]
    selectedExperiment: IPyBCKGExperiment
}

export class NewExperimentSelectorComponent extends React.Component<unknown, INewExperimentState> {
    public constructor(props: INewExperimentProps) {
        super(props)
        this.state = {
            experimentOptions: [{}] as IPyBCKGExperiment[],
            selectedExperiment: {} as IPyBCKGExperiment
        }
    }

    private changeExperimentFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedExpName = e.target.value;
        console.log(`Selected experiment name: ${selectedExpName}`)
        const selectedExperiment = this.state.experimentOptions.find(exp => exp.Name === selectedExpName)
        if (selectedExperiment) {
            this.setState({ selectedExperiment: selectedExperiment })
        }
    }

    async getExperimentOptions() {
        const res = await axios.get<IPyBCKGExperiment[]>(api_url + '/get-experiments')
        const experiments = res.data

        this.setState({
            experimentOptions: experiments,

        })
    }

    componentDidMount() {
        this.getExperimentOptions()
    }

    public render() {
        return (
        <label>
            <h3>Select an Experiment:</h3>
            <div className="formContainer">
                <select value={this.state.selectedExperiment.Name} onChange={
                    (e) => this.changeExperimentFields(e)
                }>
                    {this.state.experimentOptions.map((exp,id) =>
                        <option key={'exp'+id} value={exp.Name}>
                            {exp.Name}
                        </option>)
                    }
                </select>
            </div>
        </label>
        )
    }

}

export interface INewConfigProps {
    //onSubmit: (values: IFormValues) => unknown;
    configOptions: IConfig[];
    selectedConfig: IConfig;
    //getOptions: () => unknown;
}


interface INewConfigState {
    configOptions: IConfig[]
    selectedConfig: IConfig
}

export class NewConfigSelectorComponent extends React.Component<unknown, INewConfigState> {
    public constructor(props: INewExperimentProps) {
        super(props)
        this.state = {
            configOptions: [{}] as IConfig[],
            selectedConfig: {} as IConfig
    }}
    private changeConfigFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedConfigName = e.target.value;
        const selectedConfig = this.state.configOptions.find(cfg => cfg.name === selectedConfigName)
        if (selectedConfig) {
            this.setState({ selectedConfig: selectedConfig })
    }}

    async getOptions() {
        const res = await axios.get<IConfig[]>(api_url + '/get-configs')
        const configs = res.data
        this.setState({
            configOptions: configs,
        })
    }

    componentDidMount() {
        this.getOptions()
    }

    public render() {
        return (
            <label>
                <h3>Select a config:</h3>
                <div className="formContainer">
                    <select value={this.state.selectedConfig.name} onChange={
                        (event) => this.changeConfigFields(event)
                    }>
                        {this.state.configOptions.map(cfg =>
                            <option key={cfg.id} value={cfg.name}>
                                {cfg.name}
                            </option>)
                        }
                    </select>
                </div>
            </label>
        )
    }
}

export const ConfigSelectorComponent: React.FC<IProps> = (props: IProps) => {
    const handleSubmit = async (values: IFormValues): Promise<ISubmitResult> => {
        const result = await props.onSubmit(values);
        return result;
    };

    return (
        <Form
            onSubmit={handleSubmit}
            defaultValues={{ experiment: "" }}
            validationRules={{}}
            showButton={false}
        >
            <div className={css.formField}>
                <ExperimentField
                    name="config"
                    label="Select a previous config"
                    type="Select"
                    options={props.options}
                />
            </div>
        </Form>
    );
};


export const IterationSelectorComponent: React.FC<IProps> = (props: IProps) => {
    const handleSubmit = async (values: IFormValues): Promise<ISubmitResult> => {
        // This gets called second
        // console.log('point 2');
        const result = await props.onSubmit(values);
        return result;
    };

    return (
        <Form
            onSubmit={handleSubmit}
            defaultValues={{ iteration: "1" }}
            validationRules={{}}
            showButton={true}
        >
            <div className="formField">
                <ExperimentField
                    name="iteration"
                    label="Select an iteration"
                    type="Select"
                    options={props.options}
                />
            </div>
        </Form>
    );
};

export const FoldSelectorComponent: React.FC<IProps> = (props: IProps) => {
    const handleSubmit = async (values: IFormValues): Promise<ISubmitResult> => {
        // This gets called second
        // console.log('point 2');
        const result = await props.onSubmit(values);
        return result;
    };

    return (
        <Form
            onSubmit={handleSubmit}
            defaultValues={{ fold: "1" }}
            validationRules={{}}
            showButton={true}
        >
            <div className="formField">
                <ExperimentField
                    name="fold"
                    label="Select a fold"
                    type="Select"
                    options={props.options}
                />
            </div>
        </Form>
    );
};