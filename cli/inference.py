"""
Use this module to perform a search against the RAG system

$ python cli/inference.py --query "effect of face coverings for covid" --no_of_results 10 --model_name "text-embedding-ada-002" --openai_api_key "<enter key here>"

"""
from src.tasks.inference import inference


def parse_arguments():
    """
    Use this function to pass the query, the number of
    results to display, and the name of the
    NLP model for query embedding generation
    (must be the same as the one used during search system
    backend setup time)

    Returns
    -------
    args : dict
        a dict contaning search paramters
        { "query":  "effect of face coverings for covid",
          "no_of_results":  "10",
          "model_name":  "text-embedding-ada-002",
           "openai_api_key":  "<enter key here>" }

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help=" search query")
    parser.add_argument(
        "--no_of_results",
        type=int,
        default=10,
        help="number of search results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the nlp model for query embedding",
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
    print(inference(arguments=arguments))
