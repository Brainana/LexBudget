"""
Usage:
python scripts/create_chromadb.py --inputDirectory /path/to/directory/ --inputMetadataFile /path/to/file/ --outputDirectory /path/to/directory/ --collectionName name

Example:
python scripts/create_chromadb.py --inputDirectory brown_books --inputMetadataFile scripts/vector_metadata.json --outputDirectory chromadb --collectionName lc_chroma_lexbudget
python scripts/create_chromadb.py --inputDirectory school_docs --inputMetadataFile scripts/school_docs_metadata.json --outputDirectory chromadb --collectionName lc_chroma_schoolbudget
python scripts/create_chromadb.py --inputDirectory arlington_reports --inputMetadataFile scripts/arlington_vector_metadata.json --outputDirectory chromadb_arlington --collectionName lc_chroma_arlingtonbudget
python scripts/create_chromadb.py --inputDirectory high_school_project --outputDirectory high_school_project_chromadb --collectionName lc_chroma_high_school_project
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
parser.add_argument("--inputMetadataFile", type=str, required=False)
parser.add_argument("--outputDirectory", type=str, required=True)
parser.add_argument("--collectionName", type=str, required=True)
args = parser.parse_args()

# load metadata
metadata = None
if args.inputMetadataFile:
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
            if metadata is not None:
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
    collection_name=args.collectionName
)