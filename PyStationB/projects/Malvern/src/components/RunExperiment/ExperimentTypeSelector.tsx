import { mergeStyleSets } from "@fluentui/react";
import React from "react"
import { Redirect } from "react-router-dom";
import { ExperimentTypeSelectorComponent, IExperimentProps, IExperimentType } from "./ExperimentSelectorComponents";
import { ISubmitResult, Form } from "../utils/Form"
import "../../index.css"
import { IFormValues } from "../utils/FormShared";

interface IProps {
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    options: string[]
}

interface IState {
    experimentType:IExperimentType,
    experimentTypeOptions: IExperimentType[]
}

class ExperimentTypePage extends React.Component<unknown, IState>  {

    public constructor(props: IExperimentProps) {
        super(props);
        this.state = {
            experimentType: {'name':'', 'id':0},
            experimentTypeOptions: [
                {
                    'id': 1, 'name': 'New experiment'
                }, {
                    'id': 2, 'name': 'New iteration'
                }, {
                    'id': 3, 'name': 'Clone previous'
                }]
        };
    }

    // TODO: if samples or signals empty, error will be thrown here. need to default as empty list
    public render() {
        return (
            <div className="formContainer">
                <div >
                    <ExperimentTypeSelectorComponent onSubmit={this.handleSubmit} options={this.state.experimentTypeOptions} />
                </div>
            </div>
        )
    }

    private handleSubmit = async (values: IFormValues) => {
        // Records the experiment type selected
        const data = JSON.stringify(values);

        const experimentType = values.experimentType

        // if experimentType is new experiment, redirect to new experiment page. if clone, go to load existing page
        // TODO: replace string with enum
        if (experimentType == 'New iteration') {
            return <Redirect to="/new-iteration/reload"/>
        } else if (experimentType == 'clone'){
            return <Redirect to="/clone-experiment"/>
        } else if (experimentType == 'New experiment') {
            return <Redirect to="/new-experiment/reload"/>
        } else {
            //raise error
        }

    }
}

export default ExperimentTypePage