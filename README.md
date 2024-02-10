# Retrieval Augmented Generation Search System
This is an example of a search system leveraging retrieval augmented generation techniques to enhance information retrieval and provide contextually relevant search results.

The example in this repository uses the [CORD (COVID-19 Open Research Dataset) dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) which is a collection of COVID-19-related scientific papers, as its knowledge base. 


## Setting up the Backend of the Search System

### 1. Download the dataset (metadata.csv) from [here](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv) and place it in /data/raw/metadata.csv 

### 2. Start Milvus Server
Download and install Milvus Standalone using docker compose. Refer to the [Installation Guide](https://milvus.io/docs/install_standalone-docker.md) **or** follow the instructions below.

```
$ wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -O docker-compose.yml
$ sudo docker-compose up -d
```

**docker ps -a** should show the following:
```
      Name                     Command                  State                            Ports
--------------------------------------------------------------------------------------------------------------------
milvus-etcd         etcd -advertise-client-url ...   Up             2379/tcp, 2380/tcp
milvus-minio        /usr/bin/docker-entrypoint ...   Up (healthy)   9000/tcp
milvus-standalone   /tini -- milvus run standalone   Up             0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp

```
### 3. Start Postgres Server

```
$ docker run --name postgres0 -d  -p 5438:5432 -e POSTGRES_HOST_AUTH_METHOD=trust postgres
```
**docker ps -a** should show that it is up and running

### 4. Dumping data into Milvus and Postgres
We will use Milvus to store vectors and Postgres to store metadata corresponding to these vectors. The milvus id corresponding to the milvus vector will act as the primary key for both these tables. In this example, postgres stores article metadata such as the title of the article, abstract, authors, and the url. 

Before doing so, install the source code using the lines below at the root of this repo.  

```
$ python -m pip install .
```

Now run the following to build the backend of the search system. Note: you can use any of the embedding models supported by [OpenAI](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) 

```
$ python cli/build.py --data_path "/abs/path/to/data.csv" --model_name "text-embedding-ada-002" --openai_api_key "<enter key here>"
``` 
Specify the absolute path to the dataset using --data_path, the model name using --model_name, and OpenAI API key using --openai_api_key

## Inference
The inference code has been wrapped as an API and can be called either via command line interface or using the streamlit app.

### For command line, use: 

```
$ python cli/inference.py --query "effect of face coverings for covid" --no_of_results 10 --model_name "text-embedding-ada-002" --openai_api_key "<enter key here>"
```

Specify the query using --query argument, the number of results using --no_of_results which is an optional argument with a default value of 10 and the model name using --model_name (this should be the same model that was used during backend build time), and OpenAI API key using --openai_api_key

### For streamlit, use: 

```
$ python3 -m streamlit run api/main.py -- --openai_api_key "<enter key here>"
```
