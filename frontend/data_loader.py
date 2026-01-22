import streamlit as st
import pandas as pd
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def _load_parquet(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f" File non trovato: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_main_data():
    return _load_parquet("flickr_time_features.parquet")

@st.cache_data
def load_user_profiles():
    return _load_parquet("flickr_user_profiles.parquet")

@st.cache_data
def load_tags():
    return _load_parquet("flickr_roi_tags.parquet")

@st.cache_data
def load_od_matrix():
    return _load_parquet("view_od_matrix.parquet")

@st.cache_data
def load_heatmap():
    return _load_parquet("view_hourly_heatmap.parquet")

@st.cache_data
def load_gems():
    return _load_parquet("view_hidden_gems.parquet")

@st.cache_data
def load_clusters():
    return _load_parquet("flickr_clusters.parquet")