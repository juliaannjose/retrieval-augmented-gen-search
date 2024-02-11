"""
This module has functions to:
1. connect to postgres server, 
2. create a postgres table, 
3. insert into a postgres table, 
4. execute sql queries to fetch 
   metadata corresponding to milvus results
"""


def postgres_connect():
    """
    Connect to Postgres Server
    """
    import psycopg2

    connection = psycopg2.connect(
        host="postgres", port="5432", user="postgres", password="postgres"
    )
    cursor = connection.cursor()

    return connection, cursor


def postgres_table_creation(table_name):
    """
    This function creates a postgres table to
    store metadata corresponding to the
    vectors in the milvus vector db

    Arguments
    ----------
    table_name : string
        name of the postgres table

    """
    connection, cursor = postgres_connect()

    try:
        # delete_query = "drop table if exists " + table_name
        # cursor.execute(delete_query)
        # connection.commit()
        create_query = (
            "create table if not exists "
            + table_name
            + " (ids bigint, title text, abstract text, authors text, url text);"
        )
        cursor.execute(create_query)
        connection.commit()
        print(f"Postgres table {table_name} created successfully\n")
    except Exception as e:
        print(f"Postgres table {table_name} creation unsuccessful\n")
        print(e)


def postgres_insert_into_table(table_name, df, corresponding_milvus_ids):
    """
    This function inserts into a postgres table, metadata
    corresponding to the milvus vectors in the milvus
    vector db along with its ids.
    Metadata includes milvus_id, title, abstract, authors, url

    Arguments
    ----------
    table_name : string
        name of the postgres table
    df : pd.DataFrame
        your input dataset (pandas dataframe)
    corresponding_milvus_ids : list
        a list containing ids corresponding to milvus vectors

    """
    import csv
    import os

    connection, cursor = postgres_connect()

    titles = list(df["title"].values)
    abstract = list(df["abstract"].values)
    authors = list(df["authors"].values)
    url = list(df["url"].values)

    # create a temp file which we will use to copy data into postgres table
    temp_data_path = "data/artifacts/postgres_table.csv"
    with open(temp_data_path, "w") as f:
        writer = csv.writer(f)
        for i in range(0, len(corresponding_milvus_ids)):
            writer.writerow(
                [
                    str(corresponding_milvus_ids[i]),
                    titles[i],
                    abstract[i],
                    authors[i],
                    url[i],
                ]
            )

    # insert rows from stdin
    try:
        sql = "COPY " + table_name + " FROM STDIN DELIMITER ',' CSV HEADER"
        cursor.copy_expert(sql, open(temp_data_path, "r"))
        connection.commit()
        print(f"Inserted metadata into Postgress table {table_name}\n")
    except Exception as e:
        connection.rollback()
        print("Postgres Insertion Failed\n")
        print(e)

    # delete the temporary metadata file
    os.remove(temp_data_path)


def postgres_fetch_metadata(milvus_results, table_name):
    """
    This function executes sql queries to fetch
    metadata corresponding to the milvus results.
    eg: to retrieve title, abstract, authors, and url
    corresponding to a milvus result

    Arguments
    ----------
    milvus_results : list(Tuple)
        a list of tuples containing milvus id and distance of milvus search result
    table_name : string
        name of the postgres table

    Returns
    -------
    postgres_result : dict(dict)
        a dict of dict containing title,
        abstract, authors, url
        {"0":
            {"title":"",
            "abstract":"",
            ...
            ...
            },
        "1":
            {...}
        }

    """
    connection, cursor = postgres_connect()

    # dict of dict containing title, abstract, authors, url for each result row
    postgres_result = {}

    # indexing and fetching by milvus id
    milvus_result_ids = [result.id for result in milvus_results]
    fetch_query = (
        "select title, abstract, authors, url from "
        + table_name
        + " where ids = ANY (%s);"
    )
    cursor.execute(fetch_query, (milvus_result_ids,))
    all_rows = cursor.fetchall()

    for id, rows in enumerate(all_rows):
        id_result = {}
        if len(rows):
            id_result["title"] = rows[0]
            id_result["abstract"] = rows[1]
            id_result["authors"] = rows[2]
            id_result["url"] = rows[3]
            postgres_result[id] = id_result

    return postgres_result
