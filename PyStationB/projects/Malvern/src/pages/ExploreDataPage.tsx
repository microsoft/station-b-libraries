import React from "react"
import ProcessedDataTable from "../components/ProcessedDataTable"
import { IFormState, ISubmitResult } from "../components/utils/Form"
import "../index.css"
import { api_url } from "../components/utils/api"
import {IDataset} from "../components/Interfaces"
import { connector, PropsFromRedux } from "../store/connectors"
import { IFormValues } from "../components/utils/FormShared"


interface IDatasetProps extends PropsFromRedux {
    onSubmit: (values: IFormValues) => Promise<ISubmitResult>
    selectedDataset: IDataset
    datasetOptions: IDataset[]
}

class ExploreDataPage extends React.Component<IDatasetProps, IFormState> {

    public constructor(props: IDatasetProps) {
        super(props)
        this.state = {
            values: {
                selectedDataset: {} as IDataset,
                datasetOptions: [{}] as IDataset[]
            },
            errors: {},
            submitting: false,
            submitted: false
        }
    }

    private setValue = (fieldName: string, value: any) => {
        const newValues = { ...this.state.values, [fieldName]: value };
        this.setState({ values: newValues });
    }

    public displayOptions() {

        //const tableService = this.props.connection?.tableService
        //this.props.getDatasetOptions(api_url)
        //console.log('props in explore data page: ', this.props)

        const connection = this.props.connection
        const connected = connection?.connected || false
        const connectionString = connection?.connectionString || ""
        this.props.getDatasetOptions( api_url, connectionString)

        const getDataResult = this.props.getDatasetResult
        const datasets = getDataResult?.dataset_options
        console.log('does datasets exist? ')
        console.log(getDataResult)

        if (datasets) {
            this.setValue("datasetOptions", datasets)
            this.setValue("selectedDataset", datasets[0])
        }
    }

    componentDidMount() {

        this.displayOptions()
        
    }

    //shouldComponentUpdate(nextProps: IDatasetProps) {
    //    console.log('current props getdataset: ', this.props.getDatasetResult)
    //    return (this.props.getDatasetResult != nextProps.getDatasetResult)
    //}

    //componentDidUpdate(prevProps: IDatasetProps) {
    //    if (prevProps.getDatasetResult !== this.props.getDatasetResult) {
    //        console.log('updating component')
    //        this.displayOptions()
    //    }
    //}

    private changeDataset = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedDatasetName = e.target.value;
        const datasetOptions: IDataset[] = this.state.values.datasetOptions
        const selectedDataset = datasetOptions.find(data => data.name === selectedDatasetName)

        if (selectedDataset) {
            this.setValue("selectedDataset", selectedDataset)           
        }
    }

    public render() {
        const dataset = this.props.getDatasetResult || { dataset_options: [{} as IDataset]}
        const datasetOptions: IDataset[] = dataset.dataset_options || [{} as IDataset]
        console.log('props in render: ', this.props)
        console.log('dataset: ', dataset)
        console.log('dataset options: ', datasetOptions)
        const selectedDataset = this.state.values.selectedDataset

        //console.log('Explore data page connection')
        //console.log(this.props.connection)

        if (!this.props.connection?.connected) {
            return (
                <div>
                    <h3>You are not connected. Please go to log in to see data</h3>
                </div>
            )
        } else {
            if (selectedDataset.name) {
                return (
                    <div className="pageContainer">
                        <div>
                            <form >
                                <div className="formField">
                                    <label>
                                        <h3>Select a dataset:</h3>
                                        <div className="formContainer">
                                            <select
                                                value={selectedDataset.name}
                                                onChange={(e) => this.changeDataset(e)
                                                }>
                                                {datasetOptions.map((dt, index) =>
                                                    <option key={'dataset'+index} value={dt.name}>
                                                        {dt.name}
                                                    </option>)
                                                }
                                            </select>
                                        </div>
                                    </label>
                                </div>
                            </form>
                        </div>
                        <div className="tableContainer">
                            <div className="table">
                                <ProcessedDataTable dataset={selectedDataset} />
                            </div>
                        </div>

                    </div>
                )
            } else {
                return (
                    <div className="pageContainer">
                        <div>
                            <form >
                                <div className="formField">
                                    <label>
                                        <h3>Select a dataset:</h3>
                                        <div className="formContainer">
                                            <select
                                                value={selectedDataset.name}
                                                onChange={(e) => this.changeDataset(e)
                                                }>
                                                {datasetOptions.map((dt, index) =>
                                                    <option key={'dataset' + index} value={dt.name}>
                                                        {dt.name}
                                                    </option>)
                                                }
                                            </select>
                                        </div>
                                    </label>
                                </div>
                            </form>
                        </div>
                    </div>
                )
            }
        }
    }
      
}

export default connector(ExploreDataPage);