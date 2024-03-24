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
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
import anthropic


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
if "claudeMessages" not in st.session_state:
    st.session_state.claudeMessages = []

# Display all previous messages upon page refresh
assistantAvatar = config.get('Template', 'assistantAvatar')
numMsgs = len(st.session_state.claudeMessages)
for index,message in enumerate(st.session_state.claudeMessages):
    if message["type"] == "message":
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=assistantAvatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                if index < numMsgs - 1:
                    # Add references in st.expander if applicable
                    if st.session_state.claudeMessages[index+1]["type"] == "reference":
                        with st.expander("References"):
                            st.markdown(st.session_state.claudeMessages[index+1]["content"], unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

# Get UITest config
UITest = config.get('Server', 'UITest')

embeddings = OpenAIEmbeddings()

chroma_db = Chroma(persist_directory="chromadb_data", embedding_function=embeddings, collection_name="lc_chroma_lexbudget")

client = anthropic.Client(api_key=st.secrets["CLAUDE_API_KEY"])
    
# Display the input text box
chatInputPlaceholder = config.get('Template', 'chatInputPlaceholder')
if prompt := st.chat_input(chatInputPlaceholder):
    # User has entered a question -> save it to the session state 
    st.session_state.claudeMessages.append({"role": "user", "type": "message", "content": prompt})

    # Copy the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistantAvatar):

        message_placeholder = st.empty()
        message_placeholder.markdown("Running retrieval...")

        # Track query start time
        start_time = time.time()

        # get relevant documents from vector db w/ similarity search
        docs = chroma_db.similarity_search(prompt)

        context = ""
        for doc in docs:
            context += " " + doc.page_content

        query = f"""If the user does not specify a year or says 'current', assume the question is asking about FY2025. Answer the question based only on the following context:
        {context}

        Question: {prompt}
        """

        full_response = ""

        # stream response
        with client.messages.stream(
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
            model="claude-3-opus-20240229",
        ) as stream:
            message_placeholder.empty()
            for text in stream.text_stream:
                full_response += (text or "")
                # Prevent latex formatting by replacing $ with html dollar literal
                full_response = full_response.replace('$','&dollar;')
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Track query end time
        end_time = time.time()
        query_time = end_time - start_time

        # construct metadata to be logged
        metadata={
            "query_time": f"{query_time:.2f} sec",
            "start_time": convert_to_est(start_time),
            "end_time": convert_to_est(end_time)
            # "assistant_id": assistantId,
            # "user_ip": user_ip,
            # "user_agent": user_agent
        }

        # Save the assistant's message in session state (we do this in addition to 
        # saving the thread because we processed the message after retrieving it)
        st.session_state.claudeMessages.append({"role": "assistant",  "type": "message", "content": full_response})

        # log user query + assistant response + metadata 
        if UITest != "true":
            st.session_state.logged_prompt = collector.log_prompt(
                config_model={"model": "gpt-4-turbo-preview"},
                prompt=prompt,
                generation=full_response,
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