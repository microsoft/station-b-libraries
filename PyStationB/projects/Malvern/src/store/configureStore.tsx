import { applyMiddleware, createStore } from "redux"
import thunk from 'redux-thunk'
import { persistStore, persistReducer } from 'redux-persist'
import { rootReducer } from "../reducers/RootReducer";
import storage from 'redux-persist/lib/storage' // defaults to localStorage for web
//import autoMergeLevel2 from "redux-persist/lib/stateReconciler/autoMergeLevel2";


const persistConfig = {
    key: 'root',
    storage,
    //stateReconciler: autoMergeLevel2, // only override properties that were persistsed (for nested persisted states)
    blacklist: [
        'getDatasetState',
        'getExperimentOptionsState',
        'getAMLRunIdsState',
        'getExperimentResultState',
        'errorState',
        'submitExperimentState'
    ] // things we don't want to persist on refresh
}

const persistedReducer = persistReducer(persistConfig, rootReducer)

export default () => {
    const store = createStore(persistedReducer, undefined, applyMiddleware(thunk))
    const persistor = persistStore(store)
    return { store, persistor }
}
