import {  mergeStyleSets } from "@fluentui/react";
import React from "react"
import { Link } from "react-router-dom";


const css = mergeStyleSets({
    tabs: {
        display: "inline"
    },
    tab: {
        borderStyle: "solid",
        borderWidth: "1px",
        borderRadius: "5px",
        borderColor: "#000000000",
        backgroundColor: "#dddddddd",
        fontWeight: "500",
        fontFamily: "arial",
        textDecoration: "None",
        transition: "color 0.2s",
        margin: "0px 5px",
        padding: "0px, 5px"
    },
});

const ExperimentHeader: React.SFC = () => {
    return (
        <header className={css.tabs}>
            <h1 >Demo</h1>
            <nav>
                <Link to="/new-experiment/reload" className={css.tab}>New experiment</Link>
                <Link to="/new-iteration/reload" className={css.tab}>New iteration</Link>
                <Link to="/clone-experiment/reload" className={css.tab}>Clone experiment</Link>
            </nav>
        </header>
    );
};

export default ExperimentHeader;
