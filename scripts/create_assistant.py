"""Script for creating an OpenAI retrieval assistant and uploading files for its context.

Usage:
python scripts/create_assistant.py --directory /path/to/directory/ --extension pdf

Has only been tested with PDFs.

Note: this script will:
1. Create a new assistant;
2. Upload files to OpenAI;
3. Link the files to the assistant.

If the script is run twice on the same files, duplicate files will be created.

You can manage your assistants and files in the OpenAI platform at 
- https://platform.openai.com/assistants
- https://platform.openai.com/files

You may find it useful to delete all files associated with a given assistant. To do this:
```
from openai import OpenAI

client = OpenAI()
assistant = client.beta.assistants.retrieve(assistant_id)
for file_id in assistant.file_ids:
    client.files.delete(file_id)
```
"""

import argparse
import os
from pathlib import Path

from openai import OpenAI
from openai.types.beta import Assistant
import streamlit as st


def create_assistant(client: OpenAI) -> Assistant:
    """Create OpenAI assistant."""
    return client.beta.assistants.create(
        name="Lexington Budget Assistant",
        instructions=(
            "You are a helpful assistant who answers questions about "
            "the town budget for Lexington, MA."
        ),
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview",
    )


def upload_files(
    file_paths: list[str], client: OpenAI, assistant_id: str, verbose: bool = False
) -> None:
    """Upload files in file_paths to assistant."""
    for file_path in file_paths:
        with open(file_path, "rb") as fp:
            uploaded_file = client.files.create(file=fp, purpose="assistants")
        client.beta.assistants.files.create(
            assistant_id=assistant_id, file_id=uploaded_file.id
        )
        if verbose:
            print(f"Uploaded {file_path} to assistant {assistant_id}.")


def get_file_paths(directory: str, extension: str) -> list[str]:
    """Get files ending with extension in directory.

    Example: get_file_paths("sample_budget_docs", "pdf")

    Args:
        directory: (str) name of directory in filesystem
        extension: (str) extension (e.g., "pdf")

    Returns: list of absolute file paths
    """
    pdf_paths = []
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            if file_name.lower().endswith(f".{extension.lower()}"):
                absolute_path = Path(root).joinpath(file_name).resolve()
                pdf_paths.append(str(absolute_path))
    return pdf_paths


def main(directory: str, extension: str) -> None:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    assistant = create_assistant(client)
    file_paths = get_file_paths(directory, extension)
    upload_files(file_paths, client, assistant.id, verbose=True)
    print(f"Created assistant. Assistant ID: {assistant.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--extension", type=str, required=True)

    args = parser.parse_args()
    main(args.directory, args.extension)
