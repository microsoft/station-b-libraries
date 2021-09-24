import { mergeStyleSets } from "@fluentui/react";
import * as React from "react";

const css = mergeStyleSets({
    tabs: {
        display: "inline"
    },
    tab: {
        borderStyle: "outset",
        //backgroundColor: "white",
        fontWeight: "500",
        transition: "color 0.2s"
    },
});

interface ITabsContext {
    activeName?: string;
    handleTabClick?: (name: string, content: React.ReactNode) => void;
}
const TabsContext = React.createContext<ITabsContext>({});

interface IState {
    activeName: string;
    activeContent: React.ReactNode;
}
interface ITabProps {
    name: string;
    initialActive?: boolean;
    heading: () => string | JSX.Element;
}

//const MasterFormHeader = (props: ITabProps) => {
//    // Here we retrieve the current value of the context
//    const context = React.useContext(TabsContext);
//    return (
//        <section>
//            <h2>{context.activeName}</h2>
//        </section>
//    );
//};


class Tabs extends React.Component<unknown, IState> {
    static ContextType = TabsContext;

    componentDidMount() {
        const activeName = this.context;
    }

    public static Tab: React.FC<ITabProps> = props => {

        // const { activeName, handleTabClick } = React.useContext(TabsContext)

        //React.useEffect(() => {
        //    if (!activeName && props.initialActive) {
        //        if (this.handleTabClick) {
        //            handleTabClick(props.name, props.children);
        //        }
        //    }
        //}, [props.name, props.initialActive, props.children, activeName, handleTabClick])

        return (
            <TabsContext.Consumer>
                {(context: ITabsContext) => {
                    const activeName = context.activeName
                        ? context.activeName
                        : props.initialActive
                            ? props.name
                            : "";
                    const handleTabClick = (e: React.MouseEvent<HTMLButtonElement>) => {
                        if (context.handleTabClick) {
                            context.handleTabClick(props.name, props.children);
                        }
                    };
                    return (
                        <button  onClick={handleTabClick}>
                            {props.heading()}
                        </button>
                    );
                }}
            </TabsContext.Consumer>
        )
    };
    public render() {
        return (
            <div className={css.tabs}>
                <TabsContext.Provider
                    value={{
                        activeName: this.state ? this.state.activeName : "",
                        handleTabClick: this.handleTabClick
                    }}
                >
                    <p>{this.props.children}</p>
                    <div>{this.state && this.state.activeContent}</div>
                </TabsContext.Provider>
            </div>
        );
    }

    private handleTabClick = (name: string, content: React.ReactNode) => {
        this.setState({ activeName: name, activeContent: content });
    };
}

export default Tabs;