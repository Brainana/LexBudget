# LexBudget
**Setup Local Dev Environment**
  - Install Python and PIP on the local system
  - We’ll be using [poetry](https://python-poetry.org/) for dependency management: install poetry with the command “pip install poetry”. Further basic usage information is [here](https://python-poetry.org/docs/basic-usage/).
  - Git clone from this repository: https://github.com/Brainana/LexBudget
  - Modify .streamlit/secrets.toml to use your own ChatGPT API key
  - If you already have a ChatGPT assistant created with the necessary files, then simply modify config.ini to use your own assistant id. Otherwise, you can create a new assistant with the command "python scripts/create_assistant.py --directory /path/to/directory/ --extension pdf"
  - Use poetry to create the virtual environment with the command “poetry install” under the project root folder. You may see the message "The current project could not be installed: No file/folder found for package lexbudget If you do not want to install the current project use --no-root" but it’s only a warning and you can ignore it.
  - Get a shell for the poetry virtual environment with the command “poetry shell”
  - In the virtual environment shell, start the Streamlit server with the command “streamlit run BudgetChatbot_AssistAPI.py”

