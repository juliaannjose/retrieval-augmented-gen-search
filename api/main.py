"""
This script sets up the streamlit UI and calls the backend code to process user query and returns the model's response.

$ python3 -m streamlit run api/main.py -- --openai_api_key "<enter key here>"

"""

import streamlit as st
from utils import *

st.set_page_config(page_title="ResearchHub", page_icon=":ghost:", layout="wide")


def search_engine():

    _OPENAI_KEY = parse_arguments()['openai_api_key']

    txt = f'<p style="font-size: 60px" align="left"> Article search engine </p>'
    st.markdown(txt, unsafe_allow_html=True)
    query = st.text_input("Search for an article")

    # milvus search limit - 16384
    no_of_results = st.slider(
        "number of search results", min_value=1, max_value=16384, value=5
    )

    if query:
        txt = f'<p style="font-style:italic;color:gray;">Showing top {no_of_results} related articles</p>'
        st.markdown(txt, unsafe_allow_html=True)
        search_param = {
            "query": query,
            "no_of_results": no_of_results,
            "openai_api_key": _OPENAI_KEY, 
            "model_name": "text-embedding-ada-002",  # hardcoding this for now
        }

        with st.spinner("Searching..."):
            response = get_response(search_param=search_param)

            st.write(response)


page_names_to_funcs = {
    "Search Engine": search_engine,
}

pages = st.sidebar.selectbox("What would you like to do?", page_names_to_funcs.keys())
page_names_to_funcs[pages]()
