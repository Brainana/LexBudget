from openai import OpenAI
import streamlit as st
import configparser
import time
from datetime import datetime
import pytz
# import libraries for user feedback
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback
from streamlit_javascript import st_javascript


# import regex library
import re
# import serialization/deserialization library
import pickle

# Get the specific configuration for the app
config = configparser.ConfigParser()
config.read('config.ini')

# Get the debug configuration mode
debug = config.get('Server', 'debug')

# Set page title
st.title(config.get('Template', 'title'))

# Initialize OpenAI client with your own API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize feedback collector
collector = FeedbackCollector(
    project="default",
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
)

def client_ip():
    url = 'https://api.ipify.org?format=json'
    script = (f'await fetch("{url}").then('
                'function(response) {'
                    'return response.json();'
                '})')
    try:
        result = st_javascript(script)
        if isinstance(result, dict) and 'ip' in result:
            return result['ip']
        else: return None
    except: return None


def get_user_agent():
    try:
        user_agent = st_javascript('navigator.userAgent')
        if user_agent: return user_agent
        else: return None
    except: return None


# user_ip = client_ip()
# user_agent = get_user_agent()

# handle feedback submissions
def _submit_feedback():
    if st.session_state.feedback_key is None:
        st.session_state.feedback_key = {'type': ""}
    st.session_state.feedback_key['text'] = st.session_state.feedback_response
    collector.log_feedback(
        component="default",
        model=st.session_state.logged_prompt.config_model.model,
        user_response=st.session_state.feedback_key,
        prompt_id=st.session_state.logged_prompt.id
    )

# Helper function to convert Unix timestamp to datetime object in EST timezone
def convert_to_est(unix_timestamp):
    utc_datetime = datetime.utcfromtimestamp(unix_timestamp)
    est_timezone = pytz.timezone('US/Eastern')
    est_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(est_timezone)
    return est_datetime.strftime('%B %d, %Y %H:%M:%S %Z')

# serialize object to pickle file
def serialize(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

# deserialize pickle file to object
def deserialize(path):
    with open(path, 'rb') as file:
       return  pickle.load(file)

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
numMsgs = len(st.session_state.messages)
for index,message in enumerate(st.session_state.messages):
    if message["type"] == "message":
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=assistantAvatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                if index < numMsgs - 1:
                    # Add references in st.expander if applicable
                    if st.session_state.messages[index+1]["type"] == "reference":
                        with st.expander("References"):
                            st.markdown(st.session_state.messages[index+1]["content"], unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

# We use a predefined assistant with uploaded files
# Should not create a new assistant every time the page refreshes
assistantId=config.get('OpenAI', 'assistantId')

# Get UITest config
UITest = config.get('Server', 'UITest')

# Display the input text box
chatInputPlaceholder = config.get('Template', 'chatInputPlaceholder')
if prompt := st.chat_input(chatInputPlaceholder):
    # User has entered a question -> save it to the session state 
    st.session_state.messages.append({"role": "user", "type": "message", "content": prompt})

    # Copy the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistantAvatar):
        # Create the ChatGPT message with the user's question
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # If UITest is true then use dummy message instead of real message
        if UITest == "true":
            message = deserialize('dummyResponse.pickle')
        else:

            # Track query start time
            start_time = time.time()

            # Query ChatGPT
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

            # Track query end time
            end_time = time.time()
            query_time = end_time - start_time

            # construct metadata to be logged
            metadata={
                "query_time": f"{query_time:.2f} sec",
                "start_time": convert_to_est(start_time),
                "end_time": convert_to_est(end_time),
                "assistant_id": assistantId
                # "user_ip": user_ip,
                # "user_agent": user_agent
            }

            # Get all messages from the thread
            messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                # make sure thread is ordered with latest messages first
                order='desc'
            )

            # If latest message is not from assistant, sleep to give Assistants API time to add GPT-4 response to thread
            if messages.data[0].role != 'assistant':
                time.sleep(2)
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    # Make sure thread is ordered with latest messages first
                    order='desc'
                )    
            
            message = None
            if messages.data[0].role != 'assistant':
                errorResponse = """OpenAI's Assistants API failed to return a response, please change your query and try again.<br>
                Tips to improve your queries:<br>
                &nbsp;&nbsp;- Provide a specific year or years<br>
                &nbsp;&nbsp;- Provide a specific area of the town budget<br>
                &nbsp;&nbsp;- Avoid abbreviations"""
                metadata["threadmsgs"] = messages
                # If assistant still doesn't return response, make an error response
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=errorResponse
                )
            else:
                # Retrieve the message object from the assistant
                message = client.beta.threads.messages.retrieve(
                    thread_id=thread.id,
                    # Use the latest message from the assistant
                    message_id=messages.data[0].id
                )

        if debug == "true":
            # Use Streamlit Magic commands to output message, which is a feature that allows dev to write
            # markdown, data, charts, etc without having to write an explicit command
            message

            # Used for serialize response to a pickle file for use as a dummy
            # message_content = message.content[0].text
            # message_content.value = "<b>This is a dummy assistant response used for testing.</b><br>" + message_content.value
            # serialize(message, 'dummyResponse.pickle')

        # Extract the message content
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []

        # When there are multiple files associated with an assistant, annotations will return as empty: 
        # see https://community.openai.com/t/assistant-api-always-return-empty-annotations/489285
        if len(annotations) == 0:
            message_content.value = re.sub(r'【[\d:]+†source】', '', message_content.value)

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
 
        # Prevent latex formatting by replacing $ with html dollar literal
        message_content.value = message_content.value.replace('$','&dollar;')

        # Display assistant message
        st.markdown(message_content.value, unsafe_allow_html=True)

        # Save the assistant's message in session state (we do this in addition to 
        # saving the thread because we processed the message after retrieving it)
        st.session_state.messages.append({"role": "assistant",  "type": "message", "content": message_content.value})

        # Add footnotes to the end of the message before displaying to user
        if len(citations) > 0:
            with st.expander("References"):
                references = '\n'.join(citations).replace('$', '&dollar;')
                st.markdown(references, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant",  "type": "reference", "content": references})

        # log user query + assistant response + metadata 
        if UITest != "true":
            st.session_state.logged_prompt = collector.log_prompt(
                config_model={"model": "gpt-4-turbo-preview"},
                prompt=prompt,
                generation=message_content.value,
                metadata=metadata
            )

            # not functional because user feedback comes back empty
            # # display feedback ui
            # user_feedback = collector.st_feedback(
            #     component="default",
            #     feedback_type="thumbs",
            #     model=st.session_state.logged_prompt.config_model.model,
            #     prompt_id=st.session_state.logged_prompt.id,
            #     open_feedback_label='[Optional] Provide additional feedback'
            # )

            # if user_feedback:

            #     # log user feedback
            #     trubrics.log_feedback(
            #         component="default",
            #         model=st.session_state.logged_prompt.config_model.model,
            #         user_response=user_feedback,
            #         prompt_id=st.session_state.logged_prompt.id
            #     )

            with st.form('form'):
                streamlit_feedback(
                    feedback_type = "thumbs",
                    align = "flex-start",
                    key='feedback_key'
                )
                st.text_input(
                    label="Please elaborate on your response.",
                    key="feedback_response"
                )
                st.form_submit_button('Submit', on_click=_submit_feedback)