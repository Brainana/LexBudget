from openai import OpenAI
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


def getUpdatedTime(item):
    return item[1]["updated_time"]

def modifyYears(years, metadata):
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

def getSchoolDocYears(schoolMetadata):
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

# Helper function to convert Unix timestamp to datetime object in EST timezone
def convert_to_est(unix_timestamp):
    utc_datetime = datetime.utcfromtimestamp(unix_timestamp)
    est_timezone = pytz.timezone('US/Eastern')
    est_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(est_timezone)
    return est_datetime.strftime('%B %d, %Y %H:%M:%S %Z')


# figure out relevant years for query
def determineYears(lastQuery, chat_history, currentFY, client, model):

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
def rephraseQuery(query, years, client, chat_history, lastQuery, message_placeholder, model):

    lastQuerySection = ""
    if lastQuery != "":
        lastQuerySection = f"There may be additional required context that is found in the previous user query: {lastQuery}"
    rephrasedQuery = f"""
    Rephrase the following query:
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

    st.session_state.debugText += f"""
References:
    {references}\n
    """
    
    return context

async def awaitable_function(obj):
    return obj

async def runAgent(userQuery, chat_history, agent_executor, message_placeholder):
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

