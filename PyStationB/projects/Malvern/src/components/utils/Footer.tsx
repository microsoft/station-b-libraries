import { mergeStyleSets } from "@fluentui/react";
import * as React from "react";
import { Link } from "react-router-dom";

const css = mergeStyleSets({
    footer: {
        "textAlign": "center",
        "height": "100px",
        "padding": "20px",
    },
    footerLink: {
        "textDecoration": "none",
        "padding": "10px",
        "margin": "10px"
      }
})

const Footer: React.SFC = () => {
    return (
      <header className={css.footer}>
        <nav>
                <Link to="/privacy" target="_blank" rel="noopener noreferrer" className={css.footerLink}>Privacy statement</Link>
        </nav>
      </header>
    );
  };
  
  export default Footer;
