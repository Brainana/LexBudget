from openai import OpenAI
import streamlit as st
import configparser
import time

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
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# We use a predefined assistant with uploaded files
# Should not create a new assistant every time the server starts
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

        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the annotations with a footnote
            message_content.value = message_content.value.replace(annotation.text, f' [{index}]')

            # Gather citations based on annotation attributes
            if (file_citation := getattr(annotation, 'file_citation', None)):
                try:
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f'[{index}]: from {cited_file.filename} \n\n{file_citation.quote}')
                except Exception as err:
                    print("Oops! There is an error getting the citation information")
                    print(f"Unexpected {err=}, {type(err)=}")
                    file_citation
 
        # Add footnotes to the end of the message before displaying to user
        message_content.value += '\n\n' + '\n'.join(citations)

        # Prevent latex formatting by escaping $ sign
        message_content.value = message_content.value.replace('$','\\$')

        # Display assistant message
        st.markdown(message_content.value)

        # Testing the latex formatting removal
        # str = "The budget is $1,246,568 for the year of 2024, an increase of $100,000 from last year"
        # st.markdown(str)
        # str = str.replace('$','\\$')
        # st.markdown(str)

        # Save the assistant's message in session state
        st.session_state.messages.append({"role": "assistant", "content": message_content.value})