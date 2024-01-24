"""
This module has functions to:
- generate embeddings for vector search using OpenAI models
- generate a prompt template using context and user query
- openAI chat completion function
"""


def generate_openai_embeddings(openai_api_key, df, column_name, model_name):
    """
    Given an OpenAI model, this function uses their embeddings.create()
    function to generate embeddings for texts.
    """

    import time
    import numpy as np
    from openai import OpenAI
    
    try:
        client = OpenAI(api_key=openai_api_key)
        
        embeddings_list = [] #list of all embeddings
        def get_embeddings(text, model_name="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model_name).data[0].embedding

        # embedding generation time over whole corpus
        start_time = time.time()
        for index, row in df.iterrows():
            embedding = get_embeddings(text=row['embedding_text'])
            embeddings_list.append(np.asarray(embedding))
        end_time = time.time()
        total = end_time - start_time
        print(f"Successfully generated embeddings in {total} seconds\n")

        dense_vectors = embeddings_list #list of np.array vectors
        return dense_vectors
    except Exception as e:
        print ("Embedding generation failed")
        print (e)



def generate_prompt_with_context(top_k_context, query):
    """
    Given the top k content obtained from the IR part
    of this system, we now use this content as additional
    context to the chat generation system. 

    This function concatenates top k content with user query."""

    try:
        context = ""
        users_query = query

        # concatenate the "abstracts" of top k articles
        for id, value in top_k_context.items():
            if value['abstract'] is not None:
                context = context + "\n" + value['abstract'] 
        
        # using this context, ask it to generate an answer to user's query
        prompt = f"""Answer the following query using only the given context. 
        
        Context: {context}
        
        Query: {users_query}
        """
        return prompt

    except Exception as e: 
        print ('Failed to generate prompt')
        print (e)



def prompt_model(prompt, openai_api_key):
    """
    This function calls OpenAI's chat.completion() function
    to generate text, given a prompt"""
    
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

        return response.choices[0].message.content
    except Exception as e: 
        print ('Failed to generate response from the model')
        print (e)
