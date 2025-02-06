#import Validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv


# Streamlit app
st.set_page_config(page_title="Summarize text")
st.title("Summarize")
st.subheader('Share url')

# get the Groq api key 
with st.sidebar:
    groq_api_key=st.text_input("Groq api key", value="",type="password")
    
generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template=""" 
Provide a summary of the following content in 300 words:
Content:{text}

"""

prompt=PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    ## Validation 
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
 #   elif not validators.url(generic_url):
 #       st.error("Please enter a valid yourtube or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                ##Loading the website ot video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"})
                    
                docs=loader.load()    

                ## Chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff", prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")  
