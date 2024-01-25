"""
This module has functions to:
1. load a csv file into a pandas dataframe
2. preprocess a pandas dataframe 
"""

def load_dataset(filepath):
    """
    This function loads the dataset into a pandas df.

    Parameters
    ----------
    filepath : string
        path to dataset

    Returns
    -------
    df : pd.DataFrame
        csv file loaded into a df

    """

    import pandas as pd

    try:
        df = pd.read_csv(filepath,header=0)
        print("Dataset Loaded Successfully\n")
        return df
    except Exception as e:
        print("Unable to Load Dataset\n")
        print(e)


def preprocess_dataset(df):
    """
    This function preprocesses a pandas dataframe
    by keeping only the required columns and
    filtering out null rows in the column that
    embedding generation is based on. 
    In this case, the required columns are title & abstract.

    Parameters
    ----------
    df : pd.DataFrame
        csv file loaded into a df

    Returns
    -------
    df : pd.DataFrame
        the preprocessed dataframe
    """
    
    import pandas as pd

    try:
        # keep only necessary columns
        cols_to_keep = ["title", "abstract", "authors", "url"]
        df = df[cols_to_keep]

        # we're creating embeddings based on title + abstract fields
        # so, first filter out rows where both fields are empty
        df = df[df.title.notna() | df.abstract.notna()]

        # create a new column that contains "title + abstract" data field
        df['embedding_text'] = (' ' + df['title']).fillna('') + (' ' + df['abstract']).fillna('')

        print(f"Preprocessed Dataset Successfully. Df has {df.count()} records\n")
        return df
    except Exception as e:
        print("Preprocessing of Dataset Failed\n")
        print(e)
