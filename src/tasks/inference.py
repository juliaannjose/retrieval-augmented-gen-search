"""
This module has a function that will perform a vector search against
the milvus vector database and also fetch metadata corresponding to 
the mulvus search results from postgres.

Use this function for inference purposes.  
"""

from src.postgres.helpers import postgres_fetch_metadata
from src.milvus.helpers import milvus_query_results_openai
from src.model.helpers import generate_prompt_with_context, prompt_model


def inference(arguments):
    """
    This function is what will be called at inference
    time. Given a query, it returns search results.

    Arguments
    ----------
    arguments : dict
        a dict contaning search parameters
        {"query":"","no_of_results":"","model_name":"", "openai_api_key":""}

    Returns
    -------
    model_response : str

    """

    # db variables
    _MILVUS_COLLECTION_NAME = _POSTGRES_TABLE_NAME = "rag_search"
    _MILVUS_INDEX_NAME = "Embedding"
    _MILVUS_SEARCH_PARAM = {"metric_type": "IP", "params": {"nprobe": 128}}

    # cli variables
    _NLP_MODEL_NAME = arguments["model_name"]
    _NO_OF_RESULTS = (
        arguments["no_of_results"] if "no_of_results" in arguments else 10
    )  # optional with default value 10
    _QUERY = arguments["query"]
    _OPENAI_KEY = arguments['openai_api_key']
    
    # ANN search
    milvus_results = milvus_query_results_openai(
        openai_api_key=_OPENAI_KEY,
        collection_name=_MILVUS_COLLECTION_NAME,
        index_name=_MILVUS_INDEX_NAME,
        query=_QUERY,
        model_name=_NLP_MODEL_NAME,
        search_params=_MILVUS_SEARCH_PARAM,
        k=_NO_OF_RESULTS,
    )

    # metadata for top-k
    postgres_results = postgres_fetch_metadata(
        milvus_results=milvus_results, table_name=_POSTGRES_TABLE_NAME
    )
    
    # curate prompt using content from top-k
    prompt = generate_prompt_with_context(postgres_results, _QUERY)

    # chat completion
    model_response = prompt_model(prompt, _OPENAI_KEY)

    return model_response
