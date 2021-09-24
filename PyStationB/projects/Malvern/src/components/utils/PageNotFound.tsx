import { mergeStyleSets } from "@fluentui/react";
import * as React from "react";

const css = mergeStyleSets({
    pageContainer: {}
})

const NotFoundPage = () => {
  return (
    <div className={css.pageContainer}>
      <h1>Sorry, this page cannot be found</h1>
    </div>
  );
};

export default NotFoundPage;