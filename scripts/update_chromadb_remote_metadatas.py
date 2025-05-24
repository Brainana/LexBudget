"""
Usage:
python scripts/update_chromadb_remote_metadatas.py --inputDirectory /path/to/directory/ --collectionName name --chromaHost host --chromaPort port

Example:
python scripts/update_chromadb_remote_metadatas.py --inputDirectory "lhs_building_project_docs/websites docs" --collectionName lc_chroma_lhsproject --chromaHost host --chromaPort port
"""

import argparse
import chromadb
from chromadb import Client, HttpClient

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
parser.add_argument("--collectionName", type=str, required=True)
parser.add_argument("--chromaHost", type=str, required=True)
parser.add_argument("--chromaPort", type=int, required=True)
args = parser.parse_args()

# initialize chroma client
chroma_client = HttpClient(host=args.chromaHost, port=args.chromaPort)

# get collection
collection = chroma_client.get_collection(name=args.collectionName)
vectors = collection.get()

# find vectors associated with the file
update_ids = []
new_metadatas = []
for vector_id, document, metadata in zip(vectors["ids"], vectors["documents"], vectors["metadatas"]):
    source = metadata["source"].replace("\\","/")
    if source.startswith("./"):
        update_ids.append(vector_id)
        metadata["source"] = source[2:]
        new_metadatas.append(metadata)
        print(f"{source}, page {metadata["page"]}")
        
BATCH_SIZE = 41666
for i in range(0, len(update_ids), BATCH_SIZE):
    batch = update_ids[i:i + BATCH_SIZE]
    collection.update(ids = update_ids, metadatas = new_metadatas)
