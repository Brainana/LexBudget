"""
Usage:
python scripts/create_chromadb.py --inputDirectory /path/to/directory/ --inputMetadataFile /path/to/file/ --outputDirectory /path/to/directory/ --fileName fileName --collectionName name

Example:
python scripts/update_chromadb.py --inputDirectory brown_books --inputMetadataFile scripts/vector_metadata.json --outputDirectory chromadb --fileName FY2025.pdf --collectionName lc_chroma_lexbudget
python scripts/update_chromadb.py --inputDirectory school_docs --inputMetadataFile scripts/school_docs_metadata.json --outputDirectory chromadb --fileName FY2025.pdf --collectionName lc_chroma_schoolbudget
python scripts/update_chromadb.py --inputDirectory high_school_project --outputDirectory high_school_project_chromadb --fileName LHS_Building_Project_FAQ.pdf --collectionName lc_chroma_high_school_project
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
parser.add_argument("--fileName", type=str, required=True)
parser.add_argument("--collectionName", type=str, required=True)
args = parser.parse_args()

# load metadata
metadata = None
if args.inputMetadataFile:
    with open(args.inputMetadataFile, 'r') as file:
        metadata = json.load(file)

# initialize OpenAI vector embeddings 
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'])

# get existing collection
collection = Chroma(persist_directory=args.outputDirectory, embedding_function = embeddings, collection_name=args.collectionName)

# remove old file
vectors = collection.get()
ids = []
for index in range(len(vectors['ids'])):
    source = vectors['metadatas'][index]['source']
    if source.endswith(args.fileName):
        ids.append(vectors['ids'][index])

if len(ids) > 0:
    collection.delete(ids)

# add updated file to database
updated_doc = []
if args.fileName.endswith(".pdf"):
    file_path = os.path.join(args.inputDirectory, args.fileName)
    print("processing " + args.fileName)
    loader = PyPDFLoader(file_path)
    chunks = loader.load_and_split()
    for index, chunk in enumerate(chunks):
        if metadata is not None:
            chunk.metadata['updated_time'] = metadata[args.fileName]['updated_time']
        updated_doc.append(chunk)

collection.add_documents(updated_doc)