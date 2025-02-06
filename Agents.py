import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, TavilySearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, TavilySearchResults, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv 

#Arxiv and Tavily Tools 
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=TavilySearchAPIWrapper(top_k_results=1, doc_content_chars_max=200)
tavily=TavilySearchResults(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("Search with Arxiv/Tavily")
"""
In this example, we're using 'StreamlitCallbackHandler' to display the thoughts and actions.
"""
# Sidebar for settings 
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is the difference between conventional and generative AI?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma-7b-It", streaming=True)
    tools=[search,arxiv,tavily]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)



