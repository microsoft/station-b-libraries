import axios from "axios";
import React from "react"
import { IAMLRun, IExperimentResult, IPyBCKGExperiment } from "../components/Interfaces";
import { ISubmitResult } from "../components/utils/Form";
import Tabs from "../components/utils/Tabs";
import "../index.css"
import { api_url } from "../components/utils/api";
import { connector, PropsFromRedux } from "../store/connectors";
import { IFormValues } from "../components/utils/FormShared";

export interface IExptSelectorProps extends PropsFromRedux{
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>;
    experimentOptions: IPyBCKGExperiment[]
    iterationOptions: string[]
    foldOptions: string[]
}

export interface IState {
    selectedExperiment: IPyBCKGExperiment
    experimentOptions: IPyBCKGExperiment[],
    selectedIteration: string,
    iterationOptions: string[],
    selectedFold: string,
    foldOptions: string[],
    experimentResult: IExperimentResult
    aml_runid_options: IAMLRun[]
}

interface IExperimentProps {
    selectedExperiment: IPyBCKGExperiment
    experimentResult: IExperimentResult
}


class DisplayDataframeComponent extends React.Component<IExperimentProps, {dataColumns: string[]}> {
    public constructor(props: IExperimentProps) {
        super(props)
        this.state = {
            dataColumns:[]
        }
    }

    private displayRecords(recordId: number) {
        const datacolumns = this.state.dataColumns;
        return datacolumns.map((col) =>
            this.displayRecordName(col, recordId)
        )
    }

    public componentDidMount() {
        this.extractColumnNames()
    }

    public componentDidUpdate(prevProps: IExperimentProps) {
        if (this.props.selectedExperiment.Name !== prevProps.selectedExperiment.Name) {
            this.extractColumnNames()
        }
    }

    private extractColumnNames() {
        const dataRecords = this.props.experimentResult.suggestedExperiments

        if (dataRecords) {
            const colNames = Object.keys(dataRecords[0])
            this.setState({ dataColumns: colNames });
        }
    }

    private displayRecordName(colname: string, recordId: number) {
        const record = this.props.experimentResult.suggestedExperiments[recordId];
        return <td>{record[colname]}</td>
    }

    public render() {
        
        const dataRecords = this.props.experimentResult.suggestedExperiments
        const dataColumns = this.state.dataColumns
        console.log('data records')
        console.log(dataRecords)
        console.log('data columns')
        console.log(dataColumns)
        if (dataRecords) {
            return (
                <table className="table" >
                    <thead>
                        {dataColumns && dataColumns.map((col, index) =>
                            <th key={index} className="tableHeader" scope='col'>
                                {col}
                            </th>
                        )}
                    </thead>

                    <tbody>
                        {dataRecords && dataRecords.map((row, index) => (
                            <tr key={index} className="tableRow">
                                {this.displayRecords(index)}
                            </tr>
                        ))}
                    </tbody>
                </table >
            )
        }
        
    }
    
}

class ExperimentSelectorPage extends React.Component<IExptSelectorProps, IState>  {

    public constructor(props: IExptSelectorProps) {
        super(props);
        this.state = {
            selectedExperiment: {} as IPyBCKGExperiment,
            experimentOptions: [{}] as IPyBCKGExperiment[],
            selectedIteration: "",
            iterationOptions: [],
            selectedFold: "",
            foldOptions: [],
            experimentResult: {} as IExperimentResult,
            aml_runid_options: [{}] as IAMLRun[]
        };
    }

    async getOptions() {

        this.props.getExperimentOptions(api_url)
        this.props.getAMLRunIdOptions(api_url)

        console.log('props after calling getAMLRunIdOptions: ', this.props)
        const amlRunOptions = this.props.getAMLRunIdsResult
        const aml_runid_options = amlRunOptions?.aml_run_ids
        if (aml_runid_options) {
            this.setState({ aml_runid_options: aml_runid_options })
        }

        const experimentOptions = this.props.getExperimentOptionsResult
        const experiment_options = experimentOptions?.experiment_options

        if (experiment_options) {
            // To begin with, set a default selected experiment, so that iteration options update properly
            const selectedExperiment: IPyBCKGExperiment = experiment_options[0]

            this.props.getExperimentResult(api_url, selectedExperiment)
            const getExperimentResult = this.props.getExperimentResultResult
            const experiment_result = getExperimentResult?.experiment_result

            const iterationOptions: string[] = experiment_result?.iterations || []
            const selectedIteration: string = iterationOptions[0]
            const foldOptions = experiment_result?.folds || []
            const selectedFold: string = foldOptions[0]

            this.setState({
                experimentOptions: experiment_options,
                selectedExperiment: selectedExperiment,
                iterationOptions: iterationOptions,
                selectedIteration: selectedIteration,
                foldOptions: foldOptions,
                selectedFold: selectedFold
            })
        }
    }

    componentDidMount() {
        this.getOptions()
    }


    private changeExperimentFields = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedExpName = e.target.value;
        //let iterationOptions: string[] = [];
        //let foldOptions: string[] = [];
        //const options: string[] = data.map(d => (d.name))
        const selectedExperiment = this.state.experimentOptions.find(exp => exp.Name === selectedExpName) || this.state.selectedExperiment

        // Get result for this selected experiment 
        this.props.getExperimentResult(api_url, selectedExperiment)
        const getExperimentResult = this.props.getExperimentResultResult
        const experimentResult = getExperimentResult?.experiment_result

        if (selectedExperiment && experimentResult) {
            this.setState({
                selectedExperiment: selectedExperiment,
                experimentResult: experimentResult
            })

            // TODO: does this need to be set separately?
            //iterationOptions = experimentResult.iterations
            //foldOptions = experimentResult.folds

            //this.setState({
            //    iterationOptions: iterationOptions,
            //    foldOptions: foldOptions,
            //    selectedIteration: iterationOptions[0],
            //    selectedFold: foldOptions[0]
            //})
        }
        
    }

    private handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        console.log('Sumitted form values')
        console.log(this.state);

        // This gets called first and does the form validation
        e.preventDefault();
    }

    plotIterationCharts() {
        const experimentResult = this.props.getExperimentResultResult?.experiment_result || this.state.experimentResult
        if (experimentResult.imageFolders) {
            return (
                <div className="imgContainer">
                    {experimentResult.imageFolders.map(imgFolder => (
                        experimentResult.imageNames.map(imgName => (
                            <div key={imgName}>
                                <img
                                    alt='graph1'
                                    src={`${process.env.PUBLIC_URL}${imgFolder}/run_seed00000${this.state.selectedFold}/iter${this.state.selectedIteration}/${imgName}.png`}
                                />
                            </div>
                        ))
                    ))}
                </div>
            )
        } else {
            return null
        }
    }

    plotCombinedIterationsCharts() {
        const experimentResult = this.props.getExperimentResultResult?.experiment_result || this.state.experimentResult
        if (experimentResult.imageFolders) {
            return (
                <div className="imgContainer">
                    {experimentResult.imageFolders.map(imgFolder => (
                        <div key={imgFolder}>
                            <div key={imgFolder+"1"}>
                                <img
                                    alt="convergenceplot"
                                    src={`${process.env.PUBLIC_URL}${imgFolder}/run_seed00000${this.state.selectedFold}/convergence_plot.png`}
                                />
                            </div>
                            <div key={imgFolder+"2"}>
                                <img
                                    alt="simulationSlices"
                                    src={`${process.env.PUBLIC_URL}${imgFolder}/run_seed00000${this.state.selectedFold}/simulation_slices_visualisation.png`}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            )
        } else {
            return null
        }
    }

    public render() {
        const experimentOptions: IPyBCKGExperiment[] = this.props.getExperimentOptionsResult?.experiment_options || this.state.experimentOptions
        const amlRunIdResult = this.props.getAMLRunIdsResult
        const amlRunIdOptions = amlRunIdResult?.aml_run_ids || this.state.aml_runid_options

        if (!this.props.connection?.connected) {
            return (
                <div>
                    <h3>You are not connected. Please go to log in to see experiment results</h3>
                </div>
            )
        } else {
            return (
                <div className="pageContainer">
                    <h2> View experiment results</h2>
                    <form onSubmit={this.handleSubmit}>
                        <label>
                            Select an AML run:
                            <div className="formField">
                                <select 
                                >
                                    {amlRunIdOptions.map((run, id) =>
                                        <option key={'run' + id} value={run.RowKey}>
                                            {run.RowKey}
                                        </option>)
                                    }
                                </select>
                            </div>
                        </label>
                        <label>
                            Select an iteration:
                        <div className="formField">
                                <select value={this.state.selectedIteration} onChange={
                                    (e) => this.setState({ selectedIteration: e.target.value })
                                }>
                                    {this.state.iterationOptions.map(it =>
                                        <option key={'iteration' + it} value={it}>
                                            {it}
                                        </option>)
                                    }
                                </select>
                            </div>
                        </label>
                        <label>
                            Select a fold:
                        <div className="formField">
                                <select value={this.state.selectedFold} onChange={
                                    (e) => this.setState({ selectedFold: e.target.value })
                                }>
                                    {this.state.foldOptions.map(it =>
                                        <option key={'fold' + it} value={it}>
                                            {it}
                                        </option>)
                                    }
                                </select>
                            </div>
                        </label>

                    </form>
                    <Tabs>
                        <Tabs.Tab
                            name="IterationPlots"
                            initialActive={true}
                            heading={() => <b>View plots for this iteration</b>}
                        >
                            {this.plotIterationCharts()}
                        </Tabs.Tab>
                        <Tabs.Tab
                            name="CombinedIterationsPlots"
                            initialActive={true}
                            heading={() => <b>View plots for combined iterations</b>}
                        >
                            {this.plotCombinedIterationsCharts()}
                        </Tabs.Tab>
                        <Tabs.Tab
                            name="dataframe"
                            initialActive={true}
                            heading={() => <b>View suggested experiments</b>}
                        >
                            <div className="table">
                                <DisplayDataframeComponent
                                    selectedExperiment={this.state.selectedExperiment}
                                    experimentResult={this.state.experimentResult}
                                />
                            </div>
                        </Tabs.Tab>
                    </Tabs>

                </div>
            )
        }
    }
}

export default connector(ExperimentSelectorPage);