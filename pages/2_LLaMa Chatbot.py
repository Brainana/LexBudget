from openai import OpenAI
from groq import Groq
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
from langchain_groq import ChatGroq
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
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# llm model version
model = "llama3-70b-8192"

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

def getUpdatedTime(item):
    return item[1]["updated_time"]

# sort docs by recency, then pull latest brown book to get latest fiscal year
metadata = dict(sorted(metadata.items(), key=getUpdatedTime, reverse=True))
schoolMetadata = dict(sorted(schoolMetadata.items(), key=getUpdatedTime, reverse=True))
latestBrownBook = next(iter(metadata))
currentFY = latestBrownBook[:6]

# initialize message placeholder
message_placeholder = None

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

def getSchoolDocYears():
    schoolDocYears = []
    for year in schoolMetadata:
        schoolDocYears.append(year[0:6])
    return schoolDocYears

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

# figure out relevant years for query
def determineYears():

    lastQuerySection = ""
    if lastQuery != "":
        lastQuerySection = f"Given this previous user query: \"{lastQuery}\" and "

    yearQuery = f"""
    {lastQuerySection}
    given the current user query: {chat_history[-1]}
    What fiscal year or years is the user asking about? Years are typically in the format "YYYY" (e.g. 2024). Give your answer in the format 'FY____, FY____, ...' with nothing else. 
    If the user didn't specify a year or says 'current', assume they are talking about {currentFY}."""

    years = client.chat.completions.create(
        messages=[{"role": "user", "content": yearQuery}],
        model=model,
    )
    years = years.choices[0].message.content.split(', ')

    st.session_state.debugText += f"""
Determine years of query: {yearQuery}\n
    LLM responded: {years}\n
    """
    return years

# rephrase user query to explicitly restate userQuery -> similarity search is more accurate
def rephraseQuery(query, years):

    lastQuerySection = ""
    if lastQuery != "":
        lastQuerySection = f"There may be additional required context that is found in the previous user query: {lastQuery}"
    rephrasedQuery = f"""
    Rephrase the following query and only return the rephrased query without any additional text:
    {chat_history[-1].content}
    Such that it queries about the following year(s):
    {years}
    {lastQuerySection}
    """

    rephrasedPrompt = client.chat.completions.create(
        messages=[{"role": "user", "content": rephrasedQuery}],
        model=model,
    )
    rephrasedPrompt = rephrasedPrompt.choices[0].message.content

    message_placeholder.markdown("Searching for: <b>" + rephrasedPrompt + "</b>", unsafe_allow_html=True)

    st.session_state.lastQuery = rephrasedPrompt
    st.session_state.debugText += f"""
Rephrase this query: {rephrasedQuery}\n
    LLM responded: {rephrasedPrompt}\n
    """

    return rephrasedPrompt

def getVectorText(collection, metadata, rephrasedQuery, years):
    # creating metadata filter that only searches documents in the relevant years
    years = list(filter(lambda year : year + ".pdf" in metadata, years))
    metadataFilter = None
    if len(years) == 1:
        filename = years[0] + ".pdf"
        metadataFilter = {'updated_time': metadata[filename]['updated_time']}
    elif len(years) > 1:
        metadataFilter = {
            '$or': []
        }
        for year in years:
            filename = year + ".pdf"
            yearFilter = {
                'updated_time': {
                    '$eq': metadata[filename]['updated_time']
                }
            }
            metadataFilter['$or'].append(yearFilter)

    st.session_state.debugText += f"""
metadata filter: 
    {metadataFilter}\n
    """

    # get relevant documents from vector db w/ similarity search
    # we fetch 4*numYears docs since if multiple years are asked more docs need to be fetched
    k = 4
    if len(years) > 1:
        k=4*len(years)
    vectors = collection.similarity_search_with_relevance_scores(rephrasedQuery, k=k, filter=metadataFilter)

    # sort docs by updated time + relevance score
    top_vectors = sorted(vectors, key=lambda vector: (-vector[0].metadata['updated_time'], -vector[1]))

    # get context from docs
    context = ""
    references = ""
    # for doc in top_vectors:
    for index, doc in enumerate(top_vectors):
        source = doc[0].metadata['source'].replace("\\","/")
        page = str(doc[0].metadata['page'])
        end_page = str(doc[0].metadata['end_page'])
        link = "<a href='https://brainana.github.io/LexBudgetDocs/" + source + "#page=" + page + "'>" + source + " (page(s) " + page + " to " + end_page + ")</a>"
        # context += "Please exactly reference the following link in the generated response: " + link + " if the following content is used to generate the response: " + doc[0].page_content + "\n"
        context += "vector #: " + str(index + 1) + "\n\nSimilarity search score: " + str(doc[1]) + "\n\nReference link: " + link + "\n\nText: " + doc[0].page_content + "\n\n"
        references += link
    
    with st.expander("(for debugging)"):
        st.markdown(st.session_state.debugText, unsafe_allow_html=True)
    with st.expander("Most Relevant Chunks w/ Similarity Score (for debugging)"):
        st.write(top_vectors)
    with st.expander("Links to Relevant Chunks (for debugging)"):
        st.markdown(references, unsafe_allow_html=True)

    st.session_state.debugText += f"""
References:
    {references}\n
    """
    
    return context

async def awaitable_function(obj):
    return obj

class SchoolTool(BaseTool):
    name = "school_budget_search"
    description = "Good for answering questions about the school or education budget for " + ', '.join(getSchoolDocYears()) + ". For budget inquiries pertaining to other years, we recommend utilizing the General Budget Search tool."

    def _helper(self, query):
        st.session_state.debugText += f"""Key idea extracted by agent:
    {query}\n
        """
        years = determineYears()
        rephrasedQuery = rephraseQuery(query, years)
        return getVectorText(schoolCollection, schoolMetadata, rephrasedQuery, years)

    def _run(self, query):
        return self._helper(query)
    
    def _arun(self, query):
        vectorText = self._helper(query)
        return awaitable_function(vectorText)


class BrownBookTool(BaseTool):
    name = "general_budget_search"
    description = "Good for answering general budget questions."

    def _helper(self, query):
        st.session_state.debugText += f"""Key idea extracted by agent:
        {query}\n
        """
        years = determineYears()
        rephrasedQuery = rephraseQuery(query, years)
        # info from one year can be found in docs from later years -> update years list: e.g. an FY2022 answer may be found in the FY2022-FY2025 document
        years = list(modifyYears(years))
        return getVectorText(brownBookCollection, metadata, rephrasedQuery, years)

    def _run(self, query):
        return self._helper(query)
    
    def _arun(self, query):
        vectorText = self._helper(query)
        return awaitable_function(vectorText)

schoolTool = SchoolTool()
brownBookTool = BrownBookTool()
tools = [brownBookTool, schoolTool]

llm = ChatGroq(temperature=0, model_name=model)
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

async def runAgent(userQuery, chat_history):
    st.session_state.debugText = ""
    st.session_state.full_response = ""

    async for event in agent_executor.astream_events(
        {"input": userQuery, "chat_history": chat_history},
        version="v1",
    ):
        kind = event["event"]

        if kind == "on_chain_start":
            if event["name"] == "Agent":
                st.session_state.debugText += f"Starting agent: {event['name']} with input: {event['data'].get('input')}\n"
        elif kind == "on_chain_end":
            if event["name"] == "Agent":
                st.session_state.debugText += f"Ending agent: {event['name']} with input: {event['data'].get('output')['output']}\n"
                chat_history.append(AIMessage(content=st.session_state.full_response))
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                st.session_state.full_response += content 
                st.session_state.full_response = st.session_state.full_response.replace('$', '&#36;')
                message_placeholder.markdown(st.session_state.full_response, unsafe_allow_html=True)
        elif kind == "on_tool_start":
            st.session_state.debugText += f"--\n\nStarting tool: {event['name']}\n\n"
        elif kind == "on_tool_end":
            st.session_state.debugText += f"\nEnding tool: {event['name']}\n--\n\n"

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

        asyncio.run(runAgent(userQuery, chat_history))
       
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

