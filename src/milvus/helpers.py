"""
This module has functions to:
1. connect to milvus server, 
2. create a milvus collection, 
3. insert into a milvus collection, 
4. peform a vector search in a milvus collection
"""


def milvus_connect():
    """
    Connect to a Milvus Server
    """
    from pymilvus import connections
    import os

    MILVUS_HOST = os.environ["MILVUS_HOST"]

    try:
        connections.connect(host=MILVUS_HOST, port="19530")
        print("Successfully connected to Milvus")
    except Exception as e:
        print(e)


def milvus_collection_creation(collection_name, index_name, index_param):
    """
    This function creates a milvus collection and
    an index using the given index parameters

    Arguments
    ----------
    collection_name : string
        name of the collection
    index_name : string
        name of the index
    index_param : dict
        the metric_type, index_type, and params to be used
        (see https://milvus.io/docs/build_index.md)

    """
    from pymilvus import (
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

    milvus_connect()

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # define key and vector index schema
    key = FieldSchema(name="ID", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field = FieldSchema(
        name=index_name, dtype=DataType.FLOAT_VECTOR, dim=1536, description="vector"
    )
    schema = CollectionSchema(fields=[key, field], description="embedding collection")

    # create collection
    collection = Collection(name=collection_name, schema=schema)
    collection.release()

    # index creation
    collection.create_index(field_name=index_name, index_params=index_param)
    if utility.has_collection(collection_name) and collection.indexes:
        print(
            f"Collection {collection_name} created successfully. Index {index_name} created successfully. \n"
        )


def milvus_insert_into_db(collection_name, dense_vectors):
    """
    This function inserts the dense vectors into
    the milvus collection

    Arguments
    ----------
    collection_name : string
        milvus collection name
    dense_vectors : list(np.ndarray)
        list of dense vectors

    Returns
    -------
    milvus_ids : list
        a list containing ids corresponding to milvus vectors

    """
    from pymilvus import Collection

    milvus_connect()
    collection = Collection(collection_name)
    all_ids = []

    # insertion batch size
    batch_size = 10000
    # insert into collection in batches of [batch_size]
    for i in range(0, len(dense_vectors), batch_size):
        mr = collection.insert([dense_vectors[i : i + batch_size]])
        all_ids.append(mr.primary_keys)

    # flattening all_ids which is a list of list into a list
    milvus_ids = [item for sublist in all_ids for item in sublist]
    print(f"Inserted all {len(dense_vectors)} vectors into milvus vector database\n")
    collection.flush()

    return milvus_ids


def milvus_query_results_openai(
    openai_api_key,
    collection_name,
    index_name,
    query,
    model_name,
    search_params,
    k,
):
    """
    This function lets you query against the milvus vector database

    Arguments
    ----------
    openai_api_key : string
        openai api key
    collection_name : string
        name of the collection
    index : string
        name of the index
    query : string
        your query in natural language
    model_name : string
        the same model name that was used to create embeddings
    search_params : dict
        certain parameters such as nprobe, metric_type
        (see https://milvus.io/docs/v1.1.1/search_vector_python.md)
    k : int
        number of articles you want to retrieve

    Returns
    -------
    results : list(Tuple)
        a list of tuples containing milvus id and distance of the search result
    """
    from pymilvus import Collection
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)

    # obtain query embedding from OpenAI model
    text = query.replace("\n", " ")
    query_embedding = (
        client.embeddings.create(input=[text], model=model_name).data[0].embedding
    )

    milvus_connect()
    collection = Collection(collection_name)
    # loading collection
    collection.load()

    # performing a vector search
    search_results = collection.search(
        data=[query_embedding],
        anns_field=index_name,
        param=search_params,
        limit=k,
        expr=None,
    )[0]

    return search_results
