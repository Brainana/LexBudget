"""
Usage:
python scripts/create_chromadb.py --inputDirectory /path/to/directory/
"""

import os
import numpy as np
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
args = parser.parse_args()

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
        all_docs.append(chunks)
all_docs = np.concatenate(all_docs)

# initialize OpenAI vector embeddings 
embeddings = OpenAIEmbeddings()

# create database w/ chunks + embeddings
chroma_db = Chroma.from_documents(
    documents=all_docs, 
    embedding=embeddings, 
    persist_directory="chromadb_data", 
    collection_name="lc_chroma_lexbudget"
)