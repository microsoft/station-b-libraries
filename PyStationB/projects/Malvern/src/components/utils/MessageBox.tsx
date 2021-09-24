import * as React from "react";
import ReactDOM from "react-dom";
import { Dialog, DialogType, DialogFooter, PrimaryButton, DefaultButton } from "@fluentui/react";
import { SafePureComponent } from "./SafePureComponent";

const modalRoot = document.getElementById('portals') as HTMLElement;

export interface IMessageBoxSettings {
  buttons: { text: string; result: string }[];
  title: string;
  subText: string;
  content?: JSX.Element;
  maxWidth?: string;
  firstFocusableSelector?: string;
}

const defaultSettings: IMessageBoxSettings = {
  buttons: [
    { text: "OK", result: "success" },
    { text: "maybe?", result: "maybe" },
    { text: "Cancel", result: "failure" }
  ],
  title: "Hang on!",
  subText: "Do you really want to do that?"
};

const initialState = {
  hidden: false,
  config: defaultSettings
};

interface IMessageBoxProps {
  resolve: (value: any) => void;
  config: IMessageBoxSettings;
  maxWidth?: string | number;
}

export class MessageBox extends SafePureComponent<IMessageBoxProps, typeof initialState> {

  constructor(props: IMessageBoxProps) {
    super(props);
    const state = { ...initialState };
    state.config = { ...state.config, ...this.props.config };
    this.state = state;
  }

  render() {
    const { hidden, config } = this.state;
    const btn0 = config.buttons[0];
    const [, ...rest] = config.buttons;


    return <Dialog hidden={hidden}
      onDismiss={() => this.dismiss("dismissed")}
      dialogContentProps={{
        type: DialogType.close,
        title: config.title,
        subText: config.subText,
      }}
      modalProps={{
        isBlocking: true,
        styles: { main: { maxWidth: 450 } },
        firstFocusableSelector: config.firstFocusableSelector || "ms-Button--primary",
      }}
      maxWidth={config.maxWidth}
    >
      {config.content}
      <DialogFooter>
        <PrimaryButton key={0} onClick={this.close} data-result={btn0.result} text={btn0.text} />
        {
          rest.map((b, i) => <DefaultButton key={i + 1} onClick={this.close} data-result={b.result} text={b.text} />)
        }

      </DialogFooter>
    </Dialog>;
  }

  dismiss(result: string) {
    const { resolve } = this.props;
    this.cancellable.setState({ hidden: true });
    resolve(result);
  }

  close = (event: any) => {
    this.dismiss(event.currentTarget.getAttribute('data-result'));
  }

  static async show(config: IMessageBoxSettings = defaultSettings) {
    const el = document.createElement('div');
    modalRoot.appendChild(el);

    const ret = new Promise<string|undefined>((resolve) => {
      ReactDOM.render(<MessageBox resolve={resolve} config={config}>
      </MessageBox >, el);
    });

    const cleanupAsync = async () => {
      await ret;
      ReactDOM.unmountComponentAtNode(el);
      modalRoot.removeChild(el);
    };

    cleanupAsync();
    return ret;
  }
}