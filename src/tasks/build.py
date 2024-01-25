"""
This module has a function that will load your dataset, 
preprocess it, create vector embeddings, create a milvus collection and an index, 
push the vectors into the milvus collection, create a postgres table, 
and store metadata into postgres.

"""

from src.dataset.helpers import load_dataset, preprocess_dataset
from src.model.helpers import generate_openai_embeddings
from src.milvus.helpers import milvus_collection_creation, milvus_insert_into_db
from src.postgres.helpers import postgres_table_creation, postgres_insert_into_table


def build(arguments):
    """
    This function loads data into Milvus and
    Postgres. This is the "offline" part of this
    search system.

    Arguments
    ----------
    arguments : dict
        a dict contaning build arguments
        {"data_path":"","model_name":"","openai_api_key":""}

    """

    # cli variables
    _PATH_TO_DATA = arguments["data_path"]
    _NLP_MODEL_NAME = arguments["model_name"]
    _OPENAI_KEY = arguments['openai_api_key']

    # db variables
    _MILVUS_COLLECTION_NAME = _POSTGRES_TABLE_NAME = "sem_search"
    _MILVUS_INDEX_NAME = "Embedding"
    _MILVUS_INDEX_PARAM = {
        "metric_type": "IP",  # IP for inner product and L2 for euclidean
        "index_type": "IVF_SQ8",  # Quantization-based index, IVF_SQ8 - high speed query but minor compormise in recall rate than IVF_PQ
        "params": {"nlist": 4096},  # 4 × sqrt(n), n = entities in a segment
    }

    try:
        # load and pre-process dataset
        df = load_dataset(filepath=_PATH_TO_DATA)
        df = preprocess_dataset(df=df)

        # embedding generation
        dense_vectors = generate_openai_embeddings(openai_api_key=_OPENAI_KEY, df=df, model_name=_NLP_MODEL_NAME)
        
        # dump embeddings to vector db
        milvus_collection_creation(
            collection_name=_MILVUS_COLLECTION_NAME,
            index_name=_MILVUS_INDEX_NAME,
            index_param=_MILVUS_INDEX_PARAM,
        )
        
        # dump embedding id to vector db
        milvus_ids = milvus_insert_into_db(
            collection_name=_MILVUS_COLLECTION_NAME, dense_vectors=dense_vectors
        )
        
        # store metadata associated with embeddings in postgres
        postgres_table_creation(table_name=_POSTGRES_TABLE_NAME)

        postgres_insert_into_table(
            table_name=_POSTGRES_TABLE_NAME,
            df=df,
            corresponding_milvus_ids=milvus_ids,
        )

        print("Pushed Data to Milvus and Postgres\n")

    except Exception as e:
        print("Failed to Push Data into Milvus and Postgres\n")
        print(e)
