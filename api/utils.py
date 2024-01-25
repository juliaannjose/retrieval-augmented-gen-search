from src.tasks.inference import inference


def get_response(search_param):
    """
    This function calls the inference api

    Parameters
      ----------
      search_param : dict
          a dict contaning search arguments
          such as query, no_of_results, openai_api_key

    Returns
      -------
      result : str
          the model's response to the query

    """
    results = inference(search_param)
    return results



def parse_arguments():
    """
    Use this function to pass OpenAI api key

    Returns
    -------
    args : dict
        a dict contaning search paramters
        { "openai_api_key":  "<enter key here>" }

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai_api_key",
        type=str,
        required=True,
        help="enter your OpenAI api key"
    )
    args = parser.parse_args()

    return vars(args)