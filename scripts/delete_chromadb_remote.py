"""
Usage:
python scripts/delete_chromadb_remote.py --inputDirectory /path/to/directory/ --fileName fileName --collectionName name --chromaHost host --chromaPort port

Example:
python scripts/delete_chromadb_remote.py --inputDirectory "lhs_building_project_docs/websites docs" --fileName "Update for Town Meeting (presentation).pdf" --collectionName lc_chroma_lhsproject --chromaHost host --chromaPort port
"""

import argparse
import chromadb
from chromadb import Client, HttpClient

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
parser.add_argument("--fileName", type=str, required=True)
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
delete_ids = []
for vector_id, document, metadata in zip(vectors["ids"], vectors["documents"], vectors["metadatas"]):
    source = metadata["source"].replace("\\","/")
    if source == args.inputDirectory + "/" + args.fileName:
        print(f"Marked for deletion: {source}, page {metadata["page"]}")
        delete_ids.append(vector_id)
        
if len(delete_ids) > 0:
    collection.delete(ids=delete_ids)
else:
    print("File not found.")
