from openai import OpenAI
from ChatbotHelper import awaitable_function, convert_to_est, determineYears, getSchoolDocYears, getUpdatedTime, getVectorText, modifyYears, rephraseQuery, runAgent
import streamlit as st
import configparser
import time
from datetime import datetime
import pytz
from typing import List
# import libraries for user feedback
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback
from streamlit_javascript import st_javascript
# import libraries for RAG + streaming
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
# import libraries for ReACT
from langchain.tools import BaseTool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
# import libraries for metadata
import json
# import regex library
import re
# import serialization/deserialization library
import pickle
# import for structured response
from langchain_core.pydantic_v1 import BaseModel

st.markdown(
    """
<style>
button {
    text-align: left;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

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

# llm model version
model = "gpt-4o-2024-05-13"

# Initialize feedback collector
collector = FeedbackCollector(
    project="default",
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
)

# load metadata
metadata = None
schoolMetadata = None
with open("./scripts/vector_metadata.json", 'r') as file:
    metadata = json.load(file)
with open("./scripts/school_docs_metadata.json", 'r') as file:
    schoolMetadata = json.load(file)


# sort docs by recency, then pull latest brown book to get latest fiscal year
metadata = dict(sorted(metadata.items(), key=getUpdatedTime, reverse=True))
schoolMetadata = dict(sorted(schoolMetadata.items(), key=getUpdatedTime, reverse=True))
latestBrownBook = next(iter(metadata))
currentFY = latestBrownBook[:6]

# initialize message placeholder
message_placeholder = None

# info from one year can be found in docs from later years -> update years list: e.g. an FY2022 answer may be found in the FY2022-FY2025 document


# user_ip = client_ip()
# user_agent = get_user_agent()

# handle feedback submissions
# def _submit_feedback():
#     if st.session_state.feedback_key is None:
#         st.session_state.feedback_key = {'type': ""}
#     st.session_state.feedback_key['text'] = st.session_state.feedback_response
#     collector.log_feedback(
#         component="default",
#         model=st.session_state.logged_prompt.config_model.model,
#         user_response=st.session_state.feedback_key,
#         userQuery_id=st.session_state.logged_prompt.id
#     )


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
if "reactMessages" not in st.session_state:
    st.session_state.reactMessages = []

# construct chat history string
chat_history = []

# get last query from session state
lastQuery = ""
if "lastQuery" in st.session_state:
    lastQuery = st.session_state.lastQuery

embeddings = OpenAIEmbeddings()

brownBookCollection = Chroma(persist_directory="chromadb", embedding_function=embeddings, collection_name="lc_chroma_lexbudget")
schoolCollection = Chroma(persist_directory="chromadb", embedding_function=embeddings, collection_name="lc_chroma_schoolbudget")

st.session_state.debugText = ""

class SchoolTool(BaseTool):
    name = "school_budget_search"
    description = "Good for answering questions about the school or education budget for " + ', '.join(getSchoolDocYears(schoolMetadata)) + ". For budget inquiries pertaining to other years, we recommend utilizing the General Budget Search tool."

    def _helper(self, query, client, chat_history, lastQuery, message_placeholder, schoolCollection, schoolMetadata, currentFY, model):
        st.session_state.debugText += f"""Key idea extracted by agent:
    {query}\n
        """
        years = determineYears(lastQuery, chat_history, currentFY, client, model)
        rephrasedQuery = rephraseQuery(query, years, client, chat_history, lastQuery, message_placeholder, model)
        return getVectorText(schoolCollection, schoolMetadata, rephrasedQuery, years)

    def _run(self, query):
        return self._helper(query, client, chat_history, lastQuery, message_placeholder, schoolCollection, schoolMetadata, currentFY, model)
    
    def _arun(self, query):
        vectorText = self._helper(query, client, chat_history, lastQuery, message_placeholder, schoolCollection, schoolMetadata, currentFY, model)
        return awaitable_function(vectorText)

class BrownBookTool(BaseTool):
    name = "general_budget_search"
    description = "Good for answering general budget questions."

    def _helper(self, query, metadata, brownBookCollection, lastQuery, chat_history, currentFY, client, model):
        st.session_state.debugText += f"""Key idea extracted by agent:
        {query}\n
        """
        years = determineYears(lastQuery, chat_history, currentFY, client, model)
        rephrasedQuery = rephraseQuery(query, years, client, chat_history, lastQuery, message_placeholder, model)
        # info from one year can be found in docs from later years -> update years list: e.g. an FY2022 answer may be found in the FY2022-FY2025 document
        years = list(modifyYears(years, metadata))
        return getVectorText(brownBookCollection, metadata, rephrasedQuery, years)

    def _run(self, query):
        return self._helper(query, metadata, brownBookCollection, lastQuery, chat_history, currentFY, client, model)
    
    def _arun(self, query):
        vectorText = self._helper(query, metadata, brownBookCollection, lastQuery, chat_history, currentFY, client, model)
        return awaitable_function(vectorText)

schoolTool = SchoolTool()
brownBookTool = BrownBookTool()
tools = [schoolTool, brownBookTool]

llm = ChatOpenAI(temperature=0, model_name=model)
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial assistant that is very knowledgable on the budget of the town of Lexington.

            Generate your prompt by priotizing the vectors with the highest similarity score.
            Ensure the response reflects the content of the search vector that matches most closely to the input query.

            If the user inquires about percentages, prioritize providing the direct percentage number from the document rather than calculating it.

            Budget info for one year can be found in documents from subsequent years up to 4 years after. 
            For example, a data point for FY2022 can be found in docs from FY2022 to FY2025.
            Prioritize using data from more recent years to form your answer, since actual figures rather than projected figures are more likely to be found in more recent documents.

            Please link the vectors you used to generate your response.
            """
        ),
        ("user", "{input}"),
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True).with_config(
    {"run_name": "Agent"}
)


# listIcons = [
#     ":large_green_circle:",
#     ":large_green_circle:",
#     ":large_green_circle:"
# ]
sampleQuestions = config.get('Template', 'sampleQuestions').split('|')
st.markdown('<p style="font-size: 18px;"><b><i>Sample questions that you could try:</i></b></p>', unsafe_allow_html=True)
questionBtns = []
for index, question in enumerate(sampleQuestions):
    # iconIndex = index % len(listIcons)
    questionBtns.append(st.button(f"{question}", type="secondary"))

if 'clicked_follow_up' not in st.session_state:
    st.session_state.clicked_follow_up = None
if 'follow_ups' not in st.session_state:
    st.session_state.follow_ups = []

def click_follow_up(question):
    st.session_state.clicked_follow_up = question

follow_up_questions = []
follow_up_btns = []

class FollowUpQuestions(BaseModel):
    """Follow up questions."""
    questions: List[str]

def suggest_follow_ups():

    follow_up_query = f"""
    Given this chat history {chat_history[-2:]}, Suggest 2 follow-up questions the user 
    might ask next."""

    structured_llm = llm.with_structured_output(FollowUpQuestions)
    response = structured_llm.invoke(follow_up_query)
    follow_up_questions = response.questions

    for index, question in enumerate(follow_up_questions):
        st.session_state.follow_ups.append(question)

def display_follow_ups():
    st.markdown('<p style="font-size: 16px;"><b><i>Follow-up questions that you could try:</i></b></p>', unsafe_allow_html=True)
    for follow_up in (st.session_state.follow_ups):
        follow_up_btns.append(st.button(f"{follow_up}", on_click=click_follow_up, args=[follow_up]))

def answerQuery(userQuery):
    chat_history.append(HumanMessage(content=userQuery))

    # User has entered a question -> save it to the session state 
    st.session_state.reactMessages.append({"role": "user", "type": "message", "content": userQuery})

    # Copy the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(userQuery)

    with st.chat_message("assistant", avatar=assistantAvatar):
        
        global message_placeholder
        message_placeholder = st.empty()

        message_placeholder.markdown('Please wait...&nbsp;&nbsp;<img src="https://brainana.github.io/LexBudgetDocs/images/loading_icon.gif" width=25>', unsafe_allow_html=True)

        # Track query start time
        start_time = time.time()

        full_response = ""

        asyncio.run(runAgent(userQuery, chat_history, agent_executor, message_placeholder))
       
        message_placeholder.markdown(st.session_state.full_response, unsafe_allow_html=True)
        # debugExpander = st.expander("Langchain Agent Steps (for debugging)")
        # debugExpanderText = debugExpander.text(st.session_state.debugText)

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
        st.session_state.reactMessages.append({"role": "assistant",  "type": "message", "content": st.session_state.full_response})

        # log user query + assistant response + metadata 
        st.session_state.logged_prompt = collector.log_prompt(
            config_model={"model": model},
            prompt=userQuery,
            generation=full_response,
            metadata=metadata
        )

        # log user feedback
        user_feedback = collector.st_feedback(
            component="default",
            feedback_type="thumbs",
            open_feedback_label="[Optional] Provide additional feedback",
            model=st.session_state.logged_prompt.config_model.model,
            prompt_id=st.session_state.logged_prompt.id,
            key="feedback_key",
            align="flex-start"
        )

        st.session_state.follow_ups = []

        suggest_follow_ups()

        # with st.form('form'):
        #     streamlit_feedback(
        #         feedback_type = "thumbs",
        #         align = "flex-start",
        #         key='feedback_key'
        #     )
        #     st.text_input(
        #         label="Please elaborate on your response.",
        #         key="feedback_response"
        #     )
        #     st.form_submit_button('Submit', on_click=_submit_feedback)

# Display all previous messages upon page refresh
assistantAvatar = config.get('Template', 'assistantAvatar')
numMsgs = len(st.session_state.reactMessages)
for index,message in enumerate(st.session_state.reactMessages):
    if message["type"] == "message":
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=assistantAvatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                chat_history.append(AIMessage(content=message["content"]))
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                chat_history.append(HumanMessage(content=message["content"]))

# Display the input text box
chatInputPlaceholder = config.get('Template', 'chatInputPlaceholder')
if userQuery := st.chat_input(chatInputPlaceholder):
    answerQuery(userQuery)

for index, questionBtn in enumerate(questionBtns):
    if questionBtn:
        answerQuery(sampleQuestions[index])

if st.session_state.clicked_follow_up:
    answerQuery(st.session_state.clicked_follow_up)
    st.session_state.clicked_follow_up = None

if st.session_state.follow_ups:
    display_follow_ups()
