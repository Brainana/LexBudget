from openai import OpenAI
import streamlit as st
import configparser
import time
# import regex library
import re

# Get the specific configuration for the app
config = configparser.ConfigParser()
config.read('config.ini')

# Get the debug configuration mode
debug = config.get('Server', 'debug')

# Set page title
st.title(config.get('Template', 'title'))

# Initialize OpenAI client with your own API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Cache the thread on session state, so we don't keep creating
# new thread for the same browser session
thread = None
if "openai_thread" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state["openai_thread"] = thread
else:
    thread = st.session_state["openai_thread"]

# Get all previous messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages upon page refresh
assistantAvatar = config.get('Template', 'assistantAvatar')
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar=assistantAvatar):
            st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# We use a predefined assistant with uploaded files
# Should not create a new assistant every time the page refreshes
assistantId=config.get('OpenAI', 'assistantId')

# Display the input text box
chatInputPlaceholder = config.get('Template', 'chatInputPlaceholder')
if prompt := st.chat_input(chatInputPlaceholder):
    # User has entered a question -> save it to the session state 
    st.session_state.messages.append({"role": "user", "content": prompt})

    # copy the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistantAvatar):
        # Create the ChatGPT message with the user's question
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # query ChatGPT
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistantId
        )
        
        # Check query status
        runStatus = None
        # Display progress bar
        progressText = "Retrieving in progress. Please wait."
        progressValue = 0
        progressBar = st.progress(0, text=progressText)
        while runStatus != "completed":
            # Update progress bar and make sure it's not exceeding 100
            if progressValue < 99:
                progressValue += 1
            else:
                progressValue = 1
            progressBar.progress(progressValue, text=progressText)
            time.sleep(0.1)

            # Keep checking query status
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if runStatus == "completed":
                progressBar.progress(100, text=progressText)
            runStatus = run.status

        # Remove progress bar
        progressBar.empty()

        # Get all messages from the thread
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
   
        # Retrieve the message object from the assistant
        message = client.beta.threads.messages.retrieve(
            thread_id=thread.id,
            # Use the latest message from the assistant
            message_id=messages.data[0].id
        )
        if debug == "true":
            # Use Streamlit Magic commands, which is a feature that allows the dev to write almost anything
            # (markdown, data, charts, etc) without having to type an explicit command
            message

        # Extract the message content
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []

        # When there are multiple files associated with an assistant, annotations will return as empty: 
        # see https://community.openai.com/t/assistant-api-always-return-empty-annotations/489285
        if len(annotations) == 0:
            message_content.value = re.sub(r'【\d+†source】', '', message_content.value)

        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):

            # Gather citations based on annotation attributes
            if (file_citation := getattr(annotation, 'file_citation', None)):
                if not file_citation.file_id:
                    # Do not provide footnote if file id of citation is empty
                    message_content.value = message_content.value.replace(annotation.text, '')
                else:
                    # Replace the annotations with a footnote
                    message_content.value = message_content.value.replace(annotation.text, f' <sup><a href="#cite_note-{message.id}-{index}">[{index}]</a></sup>')
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f'<div id="cite_note-{message.id}-{index}" style="font-size: 90%">[{index}]: {cited_file.filename} <br><br> {file_citation.quote}</div>')
            else:
                # Do not provide footnote if file for citation cannot be found
                message_content.value = message_content.value.replace(annotation.text, '')
 
        # Add footnotes to the end of the message before displaying to user
        if len(citations) > 0:
            message_content.value += '<h5 style="border-bottom: 1px solid">References</h5>'
            message_content.value += '\n\n' + '\n'.join(citations)

        # Prevent latex formatting by replacing $ with html dollar literal
        message_content.value = message_content.value.replace('$','&dollar;')

        # Display assistant message
        st.markdown(message_content.value, unsafe_allow_html=True)

        # Save the assistant's message in session state
        st.session_state.messages.append({"role": "assistant", "content": message_content.value})