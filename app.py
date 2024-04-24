import streamlit as st
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer,util
import requests
from dotenv import load_dotenv
import os, time

load_dotenv()

headers = {
    "accept": "application/json",
    "Authorization": os.getenv("API_KEY")
}

client = chromadb.PersistentClient(path="C:\\Users\\neera\\OneDrive\\Desktop\\Innomatics_Labs_data_science\\Semantic_Search_Engine_Final_Project\\db_files")
client.heartbeat()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_collection(name="subtitles_embedded", embedding_function=sentence_transformer_ef)

model = SentenceTransformer("all-MiniLM-L6-v2")

def encoding_content(x):
    return model.encode(x, normalize_embeddings=True)

def clean_text(text):
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\r\n', '', text)
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'watch any video online with opensubtitles free browser extension osdblinkext', '', text)
    text = text.strip()
    return text

def get_results(query_text):
        """ Returns a list of results from the chromadb related to the queries in priority order from high to low"""

        query_clean = clean_text(query_text)
        query_em = encoding_content(query_clean)

        search_results = collection.query(query_embeddings=query_em.tolist(), n_results=10)
        
        return search_results

st.set_page_config(
    page_title="CineSense 🎥",
    page_icon="🔎",
    layout="wide"
)

def set_style():
    st.markdown("""
    <style>
    .title {
        font-size: 36px !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }
    .subtitle {
        font-size: 24px !important;
        text-align: center !important;
        margin-bottom: 20px !important;
    }
    .text {
        font-size: 18px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_style()


st.title('CineSense 🎥')
st.subheader('सिनेसेंस A semantic search engine for seamless movie, tv series and subtitle discovery 🔍')

query_text = st.text_input('Enter your query (similar dialogues, scene, event, anything related to movies/ TV series)')
if st.button('Search'):

    start_time = time.time()
    result_data = get_results(query_text)
    end_time = time.time()
    st.success(f'Results (took - {end_time - start_time:.3f} sec):')
    st.markdown(':green[Top 10 Relevent movies/ TV series and their respective subtitle links]')

    row1 = st.columns(5)
    row2 = st.columns(5)

    cols = row1 + row2

    for i, res in enumerate(result_data['metadatas'][0]):
        subtitle_name = res['subtitle_name']
        subtitle_id = res['subtitle_id']
        subtitle_link = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"

        match = re.search(r'^(.*?)\s(?:s\d+\se\d+|\(\d{4}\))', subtitle_name)
        title_name = match.group(1)

        url = f"https://api.themoviedb.org/3/search/multi?query={title_name}&include_adult=false&language=en-US&page=1"
        # tv_url = f"https://api.themoviedb.org/3/search/tv?query={title_name}&include_adult=false&language=en-US&page=1"
        
        response = requests.get(url, headers=headers)
        
        
        data = response.json()
        if data['results'][0]['poster_path']:
            movie_image = data['results'][0]['poster_path']
            cols[i].image(f'https://image.tmdb.org/t/p/w500{movie_image}',width=200, caption='')
            
        #st.write(data)
        cols[i].markdown(f"[{subtitle_name.title()}]({subtitle_link})")
