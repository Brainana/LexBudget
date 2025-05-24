"""
Usage:
python scripts/create_chromadb_remote.py --inputDirectory /path/to/directory/ --fileName fileName --collectionName name --chromaHost host --chromaPort port

Example:
python scripts/create_chromadb_remote.py --inputDirectory "lhs_building_project_docs/websites docs" --fileName "lhs_project_faqs.pdf" --collectionName lc_chroma_lhsproject --chromaHost host --chromaPort port
"""

import os
import argparse
import tiktoken
import chromadb
from chromadb import Client, HttpClient
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
import streamlit as st
import json

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
parser.add_argument("--fileName", type=str, required=False)
parser.add_argument("--collectionName", type=str, required=True)
parser.add_argument("--chromaHost", type=str, required=True)
parser.add_argument("--chromaPort", type=int, required=True)
args = parser.parse_args()

file_list = []
if args.fileName: 
    # use the file the user inputed
    file_list.append(args.fileName)
else: 
    # get list of files in directory
    file_list = os.listdir(args.inputDirectory)

# initialize OpenAI vector embeddings 
openaiInst = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def text_to_vector(text):
    response = openaiInst.embeddings.create(input=text, model='text-embedding-ada-002')
    vector = response.data[0].embedding
    return vector

# initialize chroma client
chroma_client = HttpClient(host=args.chromaHost, port=args.chromaPort)

# get or create collection
collection = chroma_client.get_or_create_collection(name=args.collectionName)

# Create tokenizer for the embedding model
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

def count_tokens(text):
    return len(encoding.encode(text))

for file_name in file_list:
    # for each pdf, chunk document and append to list of docs
    if file_name.endswith(".pdf"):
        all_chunks = []
        file_path = os.path.join(args.inputDirectory, file_name)
        print("processing " + file_name)
        loader = PyPDFLoader(file_path)
        chunks = loader.load_and_split()

        # process files in batches so that it won't exceed the token limit for the embedding API
        MAX_TOKENS_PER_BATCH = 30000  

        batches = []
        current_batch = []
        current_token_count = 0

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            tokens = count_tokens(text)
            
            if current_token_count + tokens > MAX_TOKENS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_token_count = 0

            current_batch.append((i, chunk))
            current_token_count += tokens
           
        if current_batch:
            batches.append(current_batch)

        print(f"Number of batches: {len(batches)}")

        batch_num = 1
        for batch in batches:
            print(f"processing batch {batch_num}")
            documents = []
            embeddings = []
            metadatas = []
            ids = []

            for i, chunk in batch:
                documents.append(chunk.page_content)
                embeddings.append(text_to_vector(chunk.page_content))
                metadatas.append(chunk.metadata)
                ids.append(f"{chunk.metadata['source']}_{i}")

            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            batch_num += 1