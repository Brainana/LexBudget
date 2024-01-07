# LexBudget
**Setup Local Dev Environment**
  - Install Python and PIP on the local system
  - We’ll be using poetry for dependency management: install poetry with the command “pip install poetry”. Further basic usage information is here.
  - Git clone from this repository: https://github.com/Brainana/LexBudget
  - Modify .streamlit/secrets.toml to use your own ChatGPT API key
  - Modify config.ini to use your own assistant id
  - Use poetry to create the virtual environment with the command “poetry install” under the project root folder. You may see the message "The current project could not be installed: No file/folder found for package lexbudget If you do not want to install the current project use --no-root" but it’s only a warning and you can ignore it.
  - Get a shell for the poetry virtual environment with the command “poetry shell”
  - In the virtual environment shell, start the Streamlit server with the command “streamlit run BudgetChatbot_AssistAPI.py”

