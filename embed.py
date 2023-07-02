import pandas as pd
import regex as re
import os

import cohere 
import umap

import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def extract_list_items(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # extract all lines starting with a number
    lines = [line for line in lines if re.match(r"^\d+\.", line)]
    return lines

@st.cache_data
def embed(text):
    return co.embed(texts=text, model='embed-english-v2.0').embeddings


@st.cache_data
def reduce(embeds):
    reducer = umap.UMAP(n_neighbors=20) 
    return reducer.fit_transform(embeds)

st.title("""Kaggle Tips and Tricks""")

# build a list of line, file_name for all files under tricks folder
tips_tricks = []
for file_name in os.listdir("tricks"):
    file_name = os.path.join("tricks", file_name)
    tricks = extract_list_items(file_name)
    tips_tricks.extend([{'trick': t, 'file': file_name} for t in tricks])

tips_tricks_df = pd.DataFrame(tips_tricks, columns=['trick', 'file'])
co = cohere.Client()

# embed the tips 
with st.spinner("Embedding tips..."):
    embeds = embed(tips_tricks_df['trick'].tolist())

# reduce the dimensionality of the embeddings
with st.spinner("Reducing dimensionality..."):
    umap_embeds = reduce(embeds)
    tips_tricks_df['x'] = umap_embeds[:,0]
    tips_tricks_df['y'] = umap_embeds[:,1]

# now cluster using k-means
k = st.slider("Number of clusters", 2, 10, 8)
from sklearn.cluster import KMeans
with st.spinner("Clustering tips..."):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(umap_embeds)
    tips_tricks_df['cluster'] = kmeans.labels_

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta']
tips_tricks_df['color'] = tips_tricks_df['cluster'].apply(lambda x: colors[x])


from textwrap import wrap
tips_tricks_df['trick'] = tips_tricks_df['trick'].apply(lambda x: "<br>".join(wrap(x, width=50)))

hover_template = "<b>Trick:</b>%{text}<br><b>"
c_fig = go.Figure(go.Scatter(x=tips_tricks_df["x"], y=tips_tricks_df["y"], mode='markers', marker=dict(color=tips_tricks_df["color"]), text=tips_tricks_df['trick'], hovertemplate=hover_template))

st.plotly_chart(c_fig, use_container_width=True)

st.dataframe(tips_tricks_df[['trick', 'file']])
