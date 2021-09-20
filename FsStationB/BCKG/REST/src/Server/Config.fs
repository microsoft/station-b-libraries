module BCKG_REST_Server.Server.Config

open BCKG_REST_Server.Shared.Shared

type IBCKGApiConfig =
    abstract GetApi : userId:string -> IBCKGApi
