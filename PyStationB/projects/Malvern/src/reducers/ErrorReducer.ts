export interface IErrorState {
    error?: any
}

const initState = {
    error: null
};

export function errorReducer(state:any = initState, action: any) {
    const { error } = action;

    if (error) {
        return {
            error: error
        }
    }

    return state;
}