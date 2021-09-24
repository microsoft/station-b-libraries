import React from "react"
import { IExperimentResult } from "./Interfaces"
import "../../index.css"

interface TableProps {
    experiments: IExperimentResult[]
}


export class SampleTable extends React.Component<TableProps, { experiments: IExperimentResult[] }> {
    public constructor(props: TableProps) {
        super(props)
        this.state = {
            experiments: [{}] as IExperimentResult[]
        }
    }
    public render() {
        const experiments = this.state.experiments
        return (
            <table className="table">
                <thead>
                    <tr >
                        <th >SampleId</th>
                        <th >ExperimentId</th>
                        <th >Meta type</th>
                        <th >Physical plate name</th>
                        <th >Physical well col</th>
                        <th >Physical well row</th>
                        <th >Virtual well col</th>
                        <th >Virtual well row</th>
                    </tr>
                </thead>
                {experiments.map(experiment => (
                    <div key={experiment.id}>
                        <tbody>
                            {experiment.samples.map(
                                sample =>
                                    <div key={sample.id}>
                                        <tr>
                                            <td >{sample.id}</td>
                                            <td >{sample.experimentId}</td>
                                            <td >{sample.experimentId}</td>
                                            <td >{sample.metaType}</td>
                                            <td >{sample.physicalPlateName}</td>
                                            <td >{sample.physicalWellCol}</td>
                                            <td >{sample.physicalWellRow}</td>
                                            <td >{sample.virtualWellCol}</td>
                                            <td >{sample.virtualWellRow}</td>
                                        </tr>
                                    </div>
                            )}
                        </tbody>
                    </div>
                ))}

            </table>
        )
    }
}


export class SignalTable extends React.Component<TableProps, { experiments: IExperimentResult[] }> {
    public constructor(props: TableProps) {
        super(props)
        this.state = {
            experiments: [{}] as IExperimentResult[]
        }
    }
    public render() {
        const experiments = this.state.experiments
        return (
            <table className="table">
                <thead>
                    <tr>
                        <th >SignalId</th>
                        <th >ExperimentId</th>
                        <th >Timestamp</th>
                        <th >Type</th>
                        <th >Emission</th>
                        <th >Excitation</th>
                        <th >Gain</th>
                        <th >Wavelength</th>
                    </tr>
                </thead>
                {experiments.map(experiment => (
                    <tbody key={experiment.id}>
                        {experiment.signals.map(
                            signal =>
                                <tr key={signal.id}>
                                    <td >{signal.id}</td>
                                    <td >{signal.experimentId}</td>
                                    <td >{signal.timestamp}</td>
                                    <td >{signal.type}</td>
                                    <td >{signal.emission}</td>
                                    <td >{signal.excitation}</td>
                                    <td >{signal.gain}</td>
                                    <td >{signal.wavelength}</td>
                                </tr>
                        )}
                    </tbody>
                ))}
            </table>

        )
    }
}