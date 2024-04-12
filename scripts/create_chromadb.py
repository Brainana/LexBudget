"""
Usage:
python scripts/create_chromadb.py --inputDirectory /path/to/directory/ --inputMetadataFile /path/to/file/ --outputDirectory /path/to/directory/
"""

import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import json

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
parser.add_argument("--inputMetadataFile", type=str, required=True)
parser.add_argument("--outputDirectory", type=str, required=True)
args = parser.parse_args()

# load metadata
metadata = None
with open(args.inputMetadataFile, 'r') as file:
    metadata = json.load(file)

# get list of files in directory
file_list = os.listdir(args.inputDirectory)
all_docs = []

for file_name in file_list:
    # for each pdf, chunk document and append to list of docs
    if file_name.endswith(".pdf"):
        file_path = os.path.join(args.inputDirectory, file_name)
        print("processing " + file_name)
        loader = PyPDFLoader(file_path)
        chunks = loader.load_and_split()
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk.metadata["end_page"] = 0
            else:
                chunk.metadata["end_page"] = chunks[index+1].metadata["page"]

            chunk.metadata['updated_time'] = metadata[file_name]['updated_time']
            all_docs.append(chunk)

# initialize OpenAI vector embeddings 
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'])

# create database w/ chunks + embeddings
chroma_db = Chroma.from_documents(
    documents=all_docs, 
    collection_metadata={"hnsw:space": "cosine"},
    embedding=embeddings, 
    persist_directory=args.outputDirectory, 
    collection_name="lc_chroma_lexbudget"
)