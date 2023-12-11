from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
from datetime import timedelta
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'

# read env vars from local .env file
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
openai_api_key = os.environ['OPENAI_API_KEY']

#instantiate client
retriever = WeaviateClient(api_key, url, 'sentence-transformers/all-MiniLM-L6-v2', openai_api_key)

## RERANKER
reranker = ReRanker('cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM  --> openai_interface.py
llm = GPT_Turbo()

## ENCODING  --> tiktoken library
encodings = encoding_for_model('gpt-3.5-turbo-0613')
# = get_encoding('gpt-3.5-turbo-0613')

## INDEX NAME  --> name of your class on Weaviate cluster
class_name = "Impact_theory_minilmL6_256"

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            if guest:
                st.write(f'It looks like you selected {guest} as a filter (It is ignored for now).')
            # make hybrid call to weaviate
            hybrid_response = retriever.hybrid_search(query, class_name)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, query, 3)

            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response,
                                                      question_answering_prompt_series,
                                                      query=query,
                                                      tokenizer=encodings,
                                                      token_threshold=4000,
                                                      verbose=True)

            # generate LLM prompt
            prompt = generate_prompt_series(query=query,
                                            results=valid_response)
            
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                # creates container for LLM response
                chat_container, response_box = [], st.empty()

                # execute chat call to LLM
                #resp = llm.get_chat_completion(prompt=prompt,
                #                                  system_message='answer this question based on the podcast material',
                #                                  temperature=0,
                #                                  max_tokens=500,
                #                                  stream=False,
                #                                  show_response=False)

                resp = "Do not waste my time"

                try:
                    # inserts chat stream from LLM
                    with response_box:
                        content = resp
                        if content:
                            chat_container.append(content)
                            result = "".join(chat_container).strip()
                            st.write(f'{result}')
                except Exception as e:
                    print(e)

            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url'] # get thumbnail_url
                episode_url = hit['episode_url'] # get episode_url
                title = hit["title"] # get title
                show_length = hit["length"] # get length
                time_string = str(timedelta(seconds=show_length)) # convert show_length to readable time string

                with col1:
                    st.write(search_result(i=i,
                                            url=episode_url,
                                            guest=hit['guest'],
                                            title=title,
                                            content=hit['content'],
                                            length=time_string),
                                            unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    #st.write(f"<a href={episode_url} <img src={image} width='200'></a>",
                    #         unsafe_allow_html=True)
                    #st.markdown(f"[![{title}]({image})]({episode_url})")
                    # st.markdown(f'<a href="{episode_url}">'
                    #            f'<img src={image} '
                    #            f'caption={title.split("|")[0]} width=200, use_column_width=False />'
                    #            f'</a>',
                    #            unsafe_allow_html=True)

                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()