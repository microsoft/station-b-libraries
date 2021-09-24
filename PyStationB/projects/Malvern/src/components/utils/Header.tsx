import { mergeStyleSets } from "@fluentui/react";
import * as React from "react";
import { connect } from "react-redux";
import { Link } from "react-router-dom";
import { connector } from "../../store/connectors";

const css = mergeStyleSets({
    header: {
        "textAlign": "center",
        //"backgroundColor": "#222",
        "height": "100px",
        "padding": "20px",
        //"color": "white",
    },
    headerLink: {
        //"color": "#fff",
        "textDecoration": "none",
        "padding": "10px",
        "margin": "10px"
    },
    boldHeaderLink: {
        //"color": "#fff",
        "textDecoration": "none",
        "font-weight": "bold",
        "padding": "10px",
        "margin": "10px"
    }
})

class Header extends React.Component<any, any> {
    // TODO: change text around logout to 'hello {user}. Log out'
    public render() {
        console.log('rendering header')
        console.log('props in header: ', this.props)
        return (
            <header className={css.header}>
                <h1 >Demo</h1>
                <nav>
                    <Link to="/master-form" className={css.headerLink}>Run experiment</Link>
                    <Link to="/experiments/reload" className={css.headerLink}>View experiment results</Link>
                    <Link to="/explore-data" className={css.headerLink}>Explore data</Link>
                    {this.props.loggedIn ?
                        (
                            <Link to="/logout" className={css.boldHeaderLink}>Logout</Link>
                        ) : (
                            <Link to="/login" className={css.headerLink}>Login</Link>
                        )
                    }

                </nav>
            </header>
        );
    }
}

export default connector(Header);

