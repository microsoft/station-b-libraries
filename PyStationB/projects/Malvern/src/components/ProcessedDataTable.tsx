import React from "react"

interface IProcessedProps {
    dataset: IData
}

interface IProcessedState {
    dataColumns: string[]
}

interface IData {
    id: string,
    name: string,
    dateCreated: string,
    dataRecords: Record<string, any>[],
    dataCols: []
}

class ProcessedDataTable extends React.Component<IProcessedProps, IProcessedState> {
    public constructor(props: IProcessedProps) {
        super(props)
        this.state = {
            dataColumns: [""]
        }
    }

    public componentDidMount() {
        this.extractColumnNames()
    }

    public componentDidUpdate(prevProps: IProcessedProps) {
        if (this.props.dataset.name !== prevProps.dataset.name){
            this.extractColumnNames()
        }
    }

    private extractColumnNames() {
        const dataRecords = this.props.dataset.dataRecords
        
        if (dataRecords) {
            const colNames = Object.keys(dataRecords[0])
            this.setState({ dataColumns: colNames });
        }
    }

    private displayRecords(recordId: number) {
        const datacolumns = this.state.dataColumns;
        return datacolumns.map((col) =>
            this.displayRecordName(col, recordId)
        )
    }

    private displayRecordName(colname: string, recordId: number) {
        const record = this.props.dataset.dataRecords[recordId];
        return <td>{record[colname]}</td>
    }

    public render() {
        const dataset = this.props.dataset
        const dataRecords = dataset.dataRecords
        const dataColumns = this.state.dataColumns
        if (dataRecords.length > 0) {
            return (
                <table className="table">
                    <thead>
                        {dataColumns && dataColumns.map((col, index) =>
                            <th key={index+'header'+Date.now()} className="tableHeader" scope='col'>
                                {col}
                            </th>
                        )}
                    </thead>

                    <tbody>
                        {dataRecords && dataRecords.map((row, index) => (
                            <tr key={index+'body'+Date.now()} className="tableRow">
                                {this.displayRecords(index)}
                            </tr>
                        ))}
                    </tbody>
                </table>
            )
        } else {
            return (
                <p> No data for this table</p>    
            )
        }
    }

}

export default ProcessedDataTable;