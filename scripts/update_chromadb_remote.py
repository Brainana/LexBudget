"""
Usage:
python scripts/update_chromadb_remote.py --inputDirectory /path/to/directory/ --fileName fileName --collectionName name --chromaHost host --chromaPort port

Example:
python scripts/update_chromadb_remote.py --inputDirectory "lhs_building_project_docs/websites docs" --fileName "lhs_project_faqs.pdf" --collectionName lc_chroma_lhsproject --chromaHost host --chromaPort port
"""

import argparse
import subprocess

# get input directory
parser = argparse.ArgumentParser()
parser.add_argument("--inputDirectory", type=str, required=True)
parser.add_argument("--fileName", type=str, required=True)
parser.add_argument("--collectionName", type=str, required=True)
parser.add_argument("--chromaHost", type=str, required=True)
parser.add_argument("--chromaPort", type=int, required=True)
args = parser.parse_args()

# run delete script
subprocess.run(['python', 'scripts/delete_chromadb_remote.py', "--inputDirectory", args.inputDirectory, "--fileName", 
                args.fileName, "--collectionName", args.collectionName, "--chromaHost", args.chromaHost, "--chromaPort", str(args.chromaPort)])

# run create script
subprocess.run(['python', 'scripts/create_chromadb_remote.py', "--inputDirectory", args.inputDirectory, "--fileName", 
                args.fileName, "--collectionName", args.collectionName, "--chromaHost", args.chromaHost, "--chromaPort", str(args.chromaPort)])

