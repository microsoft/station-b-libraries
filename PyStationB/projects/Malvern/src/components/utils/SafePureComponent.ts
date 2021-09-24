import * as React from 'react';

export interface ICancellable<P, S> {
  setState: <K extends keyof S>(
    state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
    callback?: () => void)
    => void;
}

interface IRedirect {
  redirect?: JSX.Element;
}


export class SafePureComponent<P = unknown, S = unknown, SS = any> extends React.PureComponent<P, S, SS> {
  protected cancellable: ICancellable<P, S>;
  protected redirecting = false;

  constructor(props: P) {
    super(props);
    this.cancellable = { setState: this.nullSetState };
  }

  public componentDidMount(): void {
    super.componentDidMount && super.componentDidMount();
    this.cancellable.setState = this.setState.bind(this) as any;
  }

  public componentWillUnmount(): void {
    super.componentWillUnmount && super.componentWillUnmount();
    this.cancellable.setState = this.nullSetState;
  }

  // by default setState does nothing, but we can set it to do something once we're loaded.
  private nullSetState = <K extends keyof S>(
    _state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
    _callback?: () => void
  ) => { };

  public get isCancelled(): boolean {
    return this.cancellable.setState === this.nullSetState;
  }

  public redirect(e: JSX.Element): void {
    // We only allow one redirect to be set.  any subsequent ones will be ignored.
    if (!this.redirecting) {
      this.cancellable.setState({ redirect: e } as S & IRedirect);
      this.redirecting = true;
    }
  }

  public componentDidUpdate(_prevProps: any, _prevState: any): void {
    if (this.state && (this.state as IRedirect).redirect) {
      this.cancellable.setState({ redirect: undefined } as S & IRedirect);
    }
  }

  public renderRedirects(): JSX.Element | undefined {
    // after we render a redirect, we're allowed to set a new one.
    this.redirecting = false;
    return this.state && (this.state as S & IRedirect).redirect;
  }
}

// duplication is the best we can do atm...
export class SafeComponent<P = unknown, S = unknown, SS = any> extends React.Component<P, S, SS> {
  protected cancellable: ICancellable<P, S>;

  constructor(props: P) {
    super(props);
    this.cancellable = { setState: this.nullSetState };
  }

  public componentDidMount(): void {
    super.componentDidMount && super.componentDidMount();
    this.cancellable.setState = this.setState.bind(this) as any;
  }

  public componentWillUnmount(): void {
    super.componentWillUnmount && super.componentWillUnmount();
    this.cancellable.setState = this.nullSetState;
  }

  // by default setState does nothing, but we can set it to do something once we're loaded.
  private nullSetState = <K extends keyof S>(
    _state: ((prevState: Readonly<S>, props: Readonly<P>) => (Pick<S, K> | S | null)) | (Pick<S, K> | S | null),
    _callback?: () => void
  ) => { }
}
