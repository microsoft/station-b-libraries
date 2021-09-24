import React from 'react';
import ReactDOM from 'react-dom';
import Routes from "./Routes";
import PropTypes from "prop-types"
import { Provider } from "react-redux";
import { PersistGate } from 'redux-persist/integration/react'
import configureStore from './store/configureStore';

const { store, persistor } = configureStore()

ReactDOM.render(

    <Provider store={store}>
        <PersistGate loading={null} persistor={persistor}>
            <Routes />
        </PersistGate>

     </Provider>,

    document.getElementById('root')
);
