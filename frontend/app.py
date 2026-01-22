import streamlit as st
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..'))) #per importare il backend

# Configurazione
st.set_page_config(
    page_title="FlickrFlow Rome",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    /* Sfondo scuro generale */
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA !important; }

    /* Stile delle metriche */
    div[data-testid="stMetric"] { background-color: #262730; border-radius: 8px; padding: 15px; border: 1px solid #333; }

    /* NASCONDE IL MENU LATERALE AUTOMATICO DI STREAMLIT (Fondamentale!) */
    [data-testid="stSidebarNav"] { display: none !important; }

</style>
""", unsafe_allow_html=True)

# Import Pagine (Lazy Loading)
# try-except per evitare crash se un file manca

# 1. Analisi Descrittiva
try:
    from pages import overview
except ImportError:
    overview = None

try:
    from pages import query_explorer
except ImportError:
    query_explorer = None

try:
    from pages import time_trends
except ImportError:
    time_trends = None

# 2. Analisi Spaziale & Clustering
try:
    from pages import geo_spatial
except ImportError:
    geo_spatial = None

try:
    from pages import cluster
except ImportError:
    cluster = None

# 3. Semantic Analysis , User Intelligence & AI Predictor
try:
    from pages import semantic_analysis
except ImportError:
    semantic_analysis = None

try:
    from pages import user_intelligence
except ImportError:
    user_intelligence = None

try:
    from pages import recommender
except ImportError:
    recommender = None

# 4. Generative AI
try:
    from pages import ai_insight
except ImportError:
    ai_insight = None

# Sidebar
st.sidebar.image("assets/colosseo.jpg", width='stretch')
st.sidebar.title("🏛️ FLICKR FLOW  📸 ")

# Menu di Navigazione
options = [
    "📊 Overview",
    "🔍 Query Explorer",
    "🌍 Geo Spatial 3D",
    "🧠 Cluster Analysis",
    "🔗 Semantic Analysis",
    "📈 Time Trends",
    "👥 User Intelligence",
    "🔮 AI Predictor",
    "🤖 AI Strategic Advisor"
]

selection = st.sidebar.radio("Menu Principale", options)

st.sidebar.divider()
st.sidebar.caption("Modelli e tecniche per BigData")
st.sidebar.info("Progetto Universitario a cura di Mattia Marasco")

#Routing

if selection == "📊 Overview":
    if overview:
        overview.render()
    else:
        st.error("Modulo Overview non trovato.")

elif selection == "🔍 Query Explorer":
    if query_explorer:
        query_explorer.render()
    else:
        st.error("Modulo Query Explorer non trovato.")

elif selection == "🌍 Geo Spatial 3D":
    if geo_spatial:
        geo_spatial.render()
    else:
        st.error("Modulo Geo Spatial non trovato.")

elif selection == "🧠 Cluster Analysis":
    if cluster:
        cluster.render()
    else:
        st.error("Modulo Cluster non trovato.")

elif selection == "🔗 Semantic Analysis":
    if semantic_analysis:
        semantic_analysis.render()
    else:
        st.error("Modulo Semantic Analysis non trovato.")

elif selection == "📈 Time Trends":
    if time_trends:
        time_trends.render()
    else:
        st.error("Modulo Time Trends non trovato.")

elif selection == "👥 User Intelligence":
    if user_intelligence:
        user_intelligence.render()
    else:
        st.error("Modulo User Intelligence non trovato.")

elif selection == "🔮 AI Predictor":
    if recommender:
        recommender.render()
    else:
        st.error("Modulo Recommender non trovato.")

elif selection == "🤖 AI Strategic Advisor":
    if ai_insight:
        ai_insight.render()
    else:
        st.error("Modulo AI Insight non trovato.")