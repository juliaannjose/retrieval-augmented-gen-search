"""
Use this module to setup the backend of your RAG search system.
The setup includes: 
  - loading dataset
  - preprocessing
  - embedding generation
  - ingesting embeddings to milvus
  - ingesting metadata to postgres

Note: be sure to use this script *after* starting your Milvus & Postgres servers

$ python cli/build.py --data_path "/abs/path/to/data.csv" --model_name "text-embedding-ada-002" --openai_api_key "xyz"

"""

from src.tasks.build import build


def parse_arguments():
    """
    Use this function to pass the path to the dataset, and the name of the
    OpenAI NLP model for vector embedding generation

    Returns
    -------
    args : dict
        a dict contaning build paramters
        { "data_path":  "/abs/path/to/data.csv",
          "model_name":  "text-embedding-ada-002" }

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="absolute path to input csv file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the nlp model for embeddings generation",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        required=True,
        help="enter your OpenAI api key"
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":

    arguments = parse_arguments()
    build(arguments=arguments)
