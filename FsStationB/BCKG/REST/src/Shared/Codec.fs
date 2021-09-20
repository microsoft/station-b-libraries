//
// THIS IS GENERATED, DO NOT MODIFY
//
module BCKG_REST_Server.Shared.Codec

#if FABLE_COMPILER
open Thoth.Json
#else
open Thoth.Json.Net
#endif

open BCKG.Domain

open Shared

module Decoders =
    let decode<'T> (decoder:Decoder<'T>) (name:string) (responseText:string) : 'T =
        let result = Decode.fromString decoder responseText
        match result with
        | Ok t ->
            printfn "Got %s %A" name t
            t
        | Error e ->
            failwithf "Error parsing %s json: %A" name e

    let partDecoder : Decoder<Part> = Part.decode

    let stringDecoder : Decoder<string> = Decode.Auto.generateDecoderCached<string>(CamelCase)

    let stringArrayDecoder : Decoder<string[]> = Decode.array stringDecoder

    let reagentDecoder : Decoder<Reagent> = Reagent.decode

    let fileRefDecoder : Decoder<FileRef> = Decode.Auto.generateDecoderCached<FileRef>(CamelCase)

    let fileRefArrayDecoder : Decoder<FileRef[]> = Decode.array fileRefDecoder

    let cellDecoder : Decoder<Cell> = Cell.decode

    let cellEntityDecoder : Decoder<CellEntity> = CellEntity.decode

    let cellEntityArrayDecoder : Decoder<CellEntity[]> = Decode.array cellEntityDecoder

    let sampleDecoder : Decoder<Sample> = Sample.decode

    let sampleDeviceDecoder : Decoder<SampleDevice> = SampleDevice.decode

    let sampleDeviceArrayDecoder : Decoder<SampleDevice[]> = Decode.array sampleDeviceDecoder

    let conditionDecoder : Decoder<Condition> = Condition.decode

    let conditionArrayDecoder : Decoder<Condition[]> = Decode.array conditionDecoder

    let observationDecoder : Decoder<Observation> = Observation.decode

    let observationArrayDecoder : Decoder<Observation[]> = Decode.array observationDecoder

    let experimentDecoder : Decoder<Experiment> = Experiment.decode

    let sampleArrayDecoder : Decoder<Sample[]> = Decode.array sampleDecoder

    let experimentOperationDecoder : Decoder<ExperimentOperation> = ExperimentOperation.decode

    let experimentOperationArrayDecoder : Decoder<ExperimentOperation[]> = Decode.array experimentOperationDecoder

    let signalDecoder : Decoder<Signal> = Signal.decode

    let signalArrayDecoder : Decoder<Signal[]> = Decode.array signalDecoder

    let getPart : responseText:string -> Part = decode partDecoder "partDecoder"

    let getstring : responseText:string -> string = decode stringDecoder "stringDecoder"

    let getstringArray : responseText:string -> string[] = decode stringArrayDecoder "stringArrayDecoder"

    let getReagent : responseText:string -> Reagent = decode reagentDecoder "reagentDecoder"

    let getFileRef : responseText:string -> FileRef = decode fileRefDecoder "fileRefDecoder"

    let getFileRefArray : responseText:string -> FileRef[] = decode fileRefArrayDecoder "fileRefArrayDecoder"

    let getCell : responseText:string -> Cell = decode cellDecoder "cellDecoder"

    let getCellEntity : responseText:string -> CellEntity = decode cellEntityDecoder "cellEntityDecoder"

    let getCellEntityArray : responseText:string -> CellEntity[] = decode cellEntityArrayDecoder "cellEntityArrayDecoder"

    let getSample : responseText:string -> Sample = decode sampleDecoder "sampleDecoder"

    let getSampleDevice : responseText:string -> SampleDevice = decode sampleDeviceDecoder "sampleDeviceDecoder"

    let getSampleDeviceArray : responseText:string -> SampleDevice[] = decode sampleDeviceArrayDecoder "sampleDeviceArrayDecoder"

    let getCondition : responseText:string -> Condition = decode conditionDecoder "conditionDecoder"

    let getConditionArray : responseText:string -> Condition[] = decode conditionArrayDecoder "conditionArrayDecoder"

    let getObservation : responseText:string -> Observation = decode observationDecoder "observationDecoder"

    let getObservationArray : responseText:string -> Observation[] = decode observationArrayDecoder "observationArrayDecoder"

    let getExperiment : responseText:string -> Experiment = decode experimentDecoder "experimentDecoder"

    let getSampleArray : responseText:string -> Sample[] = decode sampleArrayDecoder "sampleArrayDecoder"

    let getExperimentOperation : responseText:string -> ExperimentOperation = decode experimentOperationDecoder "experimentOperationDecoder"

    let getExperimentOperationArray : responseText:string -> ExperimentOperation[] = decode experimentOperationArrayDecoder "experimentOperationArrayDecoder"

    let getSignal : responseText:string -> Signal = decode signalDecoder "signalDecoder"

    let getSignalArray : responseText:string -> Signal[] = decode signalArrayDecoder "signalArrayDecoder"

module Encoders =
    let encode<'T> (encoder:Encoder<'T>) (name:string) (o:'T) : string =
        let json = Encode.toString 0 (encoder o)
        printfn "Set %s: %s" name json
        json

    let partEncoder : Encoder<Part> = Part.encode

    let stringEncoder : Encoder<string> = Encode.Auto.generateEncoderCached<string>(CamelCase)

    let stringArrayEncoder : Encoder<string[]> = fun ts -> Encode.array (Array.map stringEncoder ts)

    let reagentEncoder : Encoder<Reagent> = Reagent.encode

    let fileRefEncoder : Encoder<FileRef> = Encode.Auto.generateEncoderCached<FileRef>(CamelCase)

    let fileRefArrayEncoder : Encoder<FileRef[]> = fun ts -> Encode.array (Array.map fileRefEncoder ts)

    let cellEncoder : Encoder<Cell> = Cell.encode

    let cellEntityEncoder : Encoder<CellEntity> = CellEntity.encode

    let cellEntityArrayEncoder : Encoder<CellEntity[]> = fun ts -> Encode.array (Array.map cellEntityEncoder ts)

    let sampleEncoder : Encoder<Sample> = Sample.encode

    let sampleDeviceEncoder : Encoder<SampleDevice> = SampleDevice.encode

    let sampleDeviceArrayEncoder : Encoder<SampleDevice[]> = fun ts -> Encode.array (Array.map sampleDeviceEncoder ts)

    let conditionEncoder : Encoder<Condition> = Condition.encode

    let conditionArrayEncoder : Encoder<Condition[]> = fun ts -> Encode.array (Array.map conditionEncoder ts)

    let observationEncoder : Encoder<Observation> = Observation.encode

    let observationArrayEncoder : Encoder<Observation[]> = fun ts -> Encode.array (Array.map observationEncoder ts)

    let experimentEncoder : Encoder<Experiment> = Experiment.encode

    let sampleArrayEncoder : Encoder<Sample[]> = fun ts -> Encode.array (Array.map sampleEncoder ts)

    let experimentOperationEncoder : Encoder<ExperimentOperation> = ExperimentOperation.encode

    let experimentOperationArrayEncoder : Encoder<ExperimentOperation[]> = fun ts -> Encode.array (Array.map experimentOperationEncoder ts)

    let signalEncoder : Encoder<Signal> = Signal.encode

    let signalArrayEncoder : Encoder<Signal[]> = fun ts -> Encode.array (Array.map signalEncoder ts)

    let setPart : part:Part -> string = encode partEncoder "partEncoder"

    let setstring : string:string -> string = encode stringEncoder "stringEncoder"

    let setstringArray : stringArray:string[] -> string = encode stringArrayEncoder "stringArrayEncoder"

    let setReagent : reagent:Reagent -> string = encode reagentEncoder "reagentEncoder"

    let setFileRef : fileRef:FileRef -> string = encode fileRefEncoder "fileRefEncoder"

    let setFileRefArray : fileRefArray:FileRef[] -> string = encode fileRefArrayEncoder "fileRefArrayEncoder"

    let setCell : cell:Cell -> string = encode cellEncoder "cellEncoder"

    let setCellEntity : cellEntity:CellEntity -> string = encode cellEntityEncoder "cellEntityEncoder"

    let setCellEntityArray : cellEntityArray:CellEntity[] -> string = encode cellEntityArrayEncoder "cellEntityArrayEncoder"

    let setSample : sample:Sample -> string = encode sampleEncoder "sampleEncoder"

    let setSampleDevice : sampleDevice:SampleDevice -> string = encode sampleDeviceEncoder "sampleDeviceEncoder"

    let setSampleDeviceArray : sampleDeviceArray:SampleDevice[] -> string = encode sampleDeviceArrayEncoder "sampleDeviceArrayEncoder"

    let setCondition : condition:Condition -> string = encode conditionEncoder "conditionEncoder"

    let setConditionArray : conditionArray:Condition[] -> string = encode conditionArrayEncoder "conditionArrayEncoder"

    let setObservation : observation:Observation -> string = encode observationEncoder "observationEncoder"

    let setObservationArray : observationArray:Observation[] -> string = encode observationArrayEncoder "observationArrayEncoder"

    let setExperiment : experiment:Experiment -> string = encode experimentEncoder "experimentEncoder"

    let setSampleArray : sampleArray:Sample[] -> string = encode sampleArrayEncoder "sampleArrayEncoder"

    let setExperimentOperation : experimentOperation:ExperimentOperation -> string = encode experimentOperationEncoder "experimentOperationEncoder"

    let setExperimentOperationArray : experimentOperationArray:ExperimentOperation[] -> string = encode experimentOperationArrayEncoder "experimentOperationArrayEncoder"

    let setSignal : signal:Signal -> string = encode signalEncoder "signalEncoder"

    let setSignalArray : signalArray:Signal[] -> string = encode signalArrayEncoder "signalArrayEncoder"

