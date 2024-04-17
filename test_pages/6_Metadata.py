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
# import libraries for RAG + streaming
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
# import libraries for metadata
import json


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
# st.markdown('Using LangChain framework + ChromaDB w/ metadata + OpenAI Chat Completions API')

# Initialize OpenAI client with your own API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize feedback collector
collector = FeedbackCollector(
    project="default",
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
)

# load metadata
metadata = None
with open("./scripts/vector_metadata.json", 'r') as file:
    metadata = json.load(file)

def getUpdatedTime(item):
    return item[1]["updated_time"]

# sort docs by recency, then pull latest book to get latest fiscal year
metadata = dict(sorted(metadata.items(), key=getUpdatedTime, reverse=True))
latestBrownBook = next(iter(metadata))
currentFY = latestBrownBook[:6]

# info from one year can be found in docs from later years -> update years list: e.g. an FY2022 answer may be found in the FY2022-FY2025 document
def modifyYears(years):
    newYearNums = set()
    for year in years:
        currYear = (int)(year[2:])
        for i in range(0,4):
            newYearNums.add(currYear+i)

    modifiedYears = set()  
    for yearNum in newYearNums:
        year = "FY" + str(yearNum) + ".pdf"
        if year in metadata:
            modifiedYears.add(year[:-4])
    
    return modifiedYears


# get client ip
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

# get user agent info 
def get_user_agent():
    try:
        user_agent = st_javascript('navigator.userAgent')
        if user_agent: return user_agent
        else: return None
    except: return None


user_ip = client_ip()
user_agent = get_user_agent()

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
if "metadataMessages" not in st.session_state:
    st.session_state.metadataMessages = []

# construct chat history string
chatHistory = ""

# get last query from session state
lastQuery = ""
if "lastQuery" in st.session_state:
    lastQuery = st.session_state.lastQuery

# Display all previous messages upon page refresh
assistantAvatar = config.get('Template', 'assistantAvatar')
numMsgs = len(st.session_state.metadataMessages)
for index,message in enumerate(st.session_state.metadataMessages):
    if message["type"] == "message":
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=assistantAvatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                chatHistory += "LLM responded: " + message["content"] + "\n"
                # if index < numMsgs - 1:
                #     # Add references in st.expander if applicable
                #     if st.session_state.metadataMessages[index+1]["type"] == "reference":
                #         with st.expander("Retrieved Chunks (for debugging)"):
                #             st.markdown(st.session_state.metadataMessages[index+1]["content"], unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                chatHistory += "The user asked: " + message["content"] + "\n"

embeddings = OpenAIEmbeddings()

chroma_db = Chroma(persist_directory="chromadb", embedding_function=embeddings, collection_name="lc_chroma_lexbudget")
    
# Display the input text box
chatInputPlaceholder = config.get('Template', 'chatInputPlaceholder')
if prompt := st.chat_input(chatInputPlaceholder):
    # User has entered a question -> save it to the session state 
    st.session_state.metadataMessages.append({"role": "user", "type": "message", "content": prompt})

    # Copy the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistantAvatar):

        message_placeholder = st.empty()

        # Track query start time
        start_time = time.time()

        # figure out relevant years for query
        query = f"""Based on this prompt: {prompt} 
        And this chat history: {chatHistory} 
        What fiscal year or years is the user asking about? Give your answer in the format 'FY____, FY____, ...' with nothing else. 
        If the user didn't specify a year or says 'current', assume they are talking about {currentFY}."""

        years = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4-turbo-preview",
        )
        strYears = years.choices[0].message.content
        years = strYears.split(", ")
        # info from one year can be found in docs from later years -> update years list: e.g. an FY2022 answer may be found in the FY2022-FY2025 document
        years = list(modifyYears(years))

        # creating metadata filter that only searches documents in the relevant years
        filter = None
        if len(years) == 1:
            filename = years[0] + ".pdf"
            filter = {'updated_time': metadata[filename]['updated_time']}
        else:
            filter={
                '$or': []
            }
            for year in years:
                filename = year + ".pdf"
                yearFilter = {
                    'updated_time': {
                        '$eq': metadata[filename]['updated_time']
                    }
                }
                filter['$or'].append(yearFilter)

        # rephrase user query to explicitly restate prompt -> similarity search is more accurate
        query = f"""Rephrase the following query to explicitly state the question: {prompt}
        Given this previous user query: {lastQuery}
        And that the query is asking about the following year(s): {strYears}
        """

        rephrasedPrompt = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4-turbo-preview",
        )
        rephrasedPrompt = rephrasedPrompt.choices[0].message.content
        st.session_state.lastQuery = rephrasedPrompt

        message_placeholder.markdown("Searching for: <b>" + rephrasedPrompt + "</b>", unsafe_allow_html=True)

        # get relevant documents from vector db w/ similarity search
        # we fetch 4*<num of queried years> docs since if multiple years are asked more docs need to be fetched
        docs = chroma_db.similarity_search_with_relevance_scores(rephrasedPrompt, k=4*len(years), filter=filter)

        # sort docs by updated time + relevance score
        top_docs = sorted(docs, key=lambda doc: (-doc[0].metadata['updated_time'], -doc[1]))

        context = ""
        references = ""
        for doc in top_docs:
            source = doc[0].metadata['source'].replace("\\","/")
            page = str(doc[0].metadata['page'])
            end_page = str(doc[0].metadata['end_page'])
            link = "<a href='https://brainana.github.io/LexBudgetDocs/" + source + "#page=" + page + "'>" + source + " (page(s) " + page + " to " + end_page + ")</a> <br>"
            context += "Please exactly reference the following link in the generated response: " + link + " if the following content is used to generate the response: " + doc[0].page_content + "\n"
            references += link
        
        # with st.expander("Rephrased Prompt (for debugging)"):
        #     st.markdown(rephrasedPrompt, unsafe_allow_html=True)
        # with st.expander("Most Relevant Chunks w/ Similarity Score (for debugging)"):
        #     st.write(docs)
        # with st.expander("Links to Relevant Chunks (for debugging)"):
        #     st.markdown(references, unsafe_allow_html=True)

        query = f"""Given this chat history: {chatHistory}
        And the following context: {context}
        Answer this user query: {rephrasedPrompt}
        Prioritize finding information on the actual historical spending amounts if applicable, and if these cannot be found search for information on the estimated or projected spending amounts.
        """

        full_response = ""

        # stream response
        for response in client.chat.completions.create(
            stream=True,
            messages=[{"role": "user", "content": query}],
            model="gpt-4-turbo-preview",
        ): 
            full_response += (response.choices[0].delta.content or "")
            # Prevent latex formatting by replacing $ with html dollar literal
            full_response = full_response.replace('$','&dollar;')
            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
       
        message_placeholder.markdown(full_response, unsafe_allow_html=True)

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
        st.session_state.metadataMessages.append({"role": "assistant",  "type": "message", "content": full_response})
        st.session_state.metadataMessages.append({"role": "assistant",  "type": "reference", "content": references})

        # log user query + assistant response + metadata 
        st.session_state.logged_prompt = collector.log_prompt(
            config_model={"model": "gpt-4-turbo-preview"},
            prompt=prompt,
            generation=full_response,
            metadata=metadata
        )

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