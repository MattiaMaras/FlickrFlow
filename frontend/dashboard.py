import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FlickrFlow | Rome Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E0E0E0 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 10px;
    }
    
    /* Cards/Metrics */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #41424C;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1C24;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    data = {}
    
    files = {
        "main": "flickr_time_features.parquet",
        "users": "flickr_user_profiles.parquet",
        "tags": "flickr_roi_tags.parquet",
        "view_hourly": "view_hourly_heatmap.parquet",
        "view_od": "view_od_matrix.parquet",
        "view_gems": "view_hidden_gems.parquet"
    }
    
    for key, filename in files.items():
        path = os.path.join(base, filename)
        if os.path.exists(path):
            try:
                data[key] = pd.read_parquet(path)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
                data[key] = None
        else:
            data[key] = None
            
    return data

data_store = load_data()
df = data_store.get("main")
df_users = data_store.get("users")

if df is None or df_users is None:
    st.error("⚠️ Critical Data Missing! Please run the backend pipeline first.")
    st.code("python backend/feature_engineering.py && python backend/tag_analysis.py && python backend/prepare_queries.py")
    st.stop()

# --- SIDEBAR NAV ---
st.sidebar.image("https://images.unsplash.com/photo-1552832230-c0197dd311b5?q=80&w=1000&auto=format&fit=crop", use_column_width=True)
st.sidebar.title("🏛️ Rome Analytics")
st.sidebar.caption("Big Data Project - Flickr Analysis")

nav = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "🔍 Query Explorer",
    "🌍 Geo-Spatial 3D",
    "📈 Time & Trends",
    "👥 User Intelligence",
    "🏷️ Semantic Analysis"
])

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Global Filters")
selected_roi = st.sidebar.multiselect("Filter by ROI", options=df['roi'].unique())
selected_user_type = st.sidebar.multiselect("User Type", options=["Tourist", "Resident", "Recurring Visitor"], default=["Tourist", "Resident", "Recurring Visitor"])

# Apply Global Filters (where applicable)
df_filtered = df.copy()
if selected_roi:
    df_filtered = df_filtered[df_filtered['roi'].isin(selected_roi)]

# --- PAGE 1: OVERVIEW ---
if nav == "🏠 Overview":
    st.title("🏛️ Project Overview")
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Photos", f"{len(df):,}", delta="100% Data")
    c2.metric("Unique Users", f"{df['user_id'].nunique():,}")
    c3.metric("Mapped Locations", f"{df['roi'].nunique() - 1}") # -1 for Unknown
    c4.metric("Time Span", f"{df['year'].min()} - {df['year'].max()}")

    st.markdown("### 🌟 Project Status")
    st.info("""
    This platform analyzes **Flickr geotagged posts** to understand tourist flows, resident behavior, and spatial semantics in Rome.
    The backend uses **Apache Spark** for ETL, **FPGrowth** for trajectory mining, and **NLP** for tag analysis.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📸 Monthly Activity Trend")
        monthly_trend = df.groupby(['year', 'month']).size().reset_index(name='count')
        
        # Robust Date Conversion
        # 1. Convert to numeric, coercing errors to NaN
        monthly_trend['year'] = pd.to_numeric(monthly_trend['year'], errors='coerce')
        monthly_trend['month'] = pd.to_numeric(monthly_trend['month'], errors='coerce')
        
        # 2. Drop NaNs and filter valid ranges
        monthly_trend = monthly_trend.dropna(subset=['year', 'month'])
        monthly_trend = monthly_trend[
            (monthly_trend['year'] > 2000) & 
            (monthly_trend['year'] < 2030) & 
            (monthly_trend['month'] >= 1) & 
            (monthly_trend['month'] <= 12)
        ]
        
        if not monthly_trend.empty:
            monthly_trend['date'] = pd.to_datetime(monthly_trend[['year', 'month']].assign(day=1))
            fig = px.line(monthly_trend, x='date', y='count', title="Data Volume over Time", line_shape="spline")
            fig.update_layout(xaxis_title="Date", yaxis_title="Photos Uploaded")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid date data found for trends.")
        
    with col2:
        st.subheader("👥 User Demographics")
        if 'user_type' in df_users.columns:
            user_dist = df_users['user_type'].value_counts().reset_index()
            fig_pie = px.pie(
                user_dist,
                values='count',
                names='user_type',
                title="Tourists vs Residents",
                hole=0.6
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("User Type data not available. Re-run feature engineering.")

# --- PAGE 2: QUERY EXPLORER (CORE REQUEST) ---
elif nav == "🔍 Query Explorer":
    st.title("🔍 Advanced Query Laboratory")
    st.markdown("Esplora i dati attraverso query complesse pre-calcolate.")
    
    tabs = st.tabs(["Q1: Flows & Trajectories", "Q2: The Perfect Time", "Q3: Hidden Gems", "Q4: Seasonality"])
    
    # Q1: Flows
    with tabs[0]:
        st.header("🔀 Where do people go next?")
        st.markdown("Analisi dei flussi di movimento tra i monumenti.")
        
        df_od = data_store.get("view_od")
        if df_od is not None:
            # Filter by Source
            sources = df_od['source'].unique()
            sel_source = st.selectbox("I am at...", sources, index=list(sources).index("Colosseo") if "Colosseo" in sources else 0)
            
            flows = df_od[df_od['source'] == sel_source].sort_values("weight", ascending=False).head(10)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Top destinations from {sel_source}:**")
                st.dataframe(flows[['target', 'user_type', 'weight']], use_container_width=True)
            
            with c2:
                # Sankey Diagram preparation
                # Create a simple Sankey for the top 10 flows
                labels = list(set(flows['source']).union(set(flows['target'])))
                label_map = {l: i for i, l in enumerate(labels)}
                
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15, thickness=20, line=dict(color="black", width=0.5),
                        label=labels,
                        color="blue"
                    ),
                    link=dict(
                        source=[label_map[s] for s in flows['source']],
                        target=[label_map[t] for t in flows['target']],
                        value=flows['weight']
                    )
                )])
                fig_sankey.update_layout(title=f"Flows starting from {sel_source}", font_size=12)
                st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("OD Matrix data missing.")

    # Q2: Perfect Time
    with tabs[1]:
        st.header("🕒 The Perfect Time to Visit")
        st.markdown("Heatmap oraria per identificare i momenti di picco.")
        
        df_hourly = data_store.get("view_hourly")
        if df_hourly is not None:
            # Pivot data for heatmap
            pivot_hm = df_hourly.groupby(['roi', 'hour'])['count'].sum().reset_index()
            
            fig_hm = px.density_heatmap(
                pivot_hm, x='hour', y='roi', z='count',
                color_continuous_scale="Viridis",
                title="Crowd Density: Location vs Hour",
                labels={'hour': 'Hour of Day', 'roi': 'Location', 'count': 'Photos'}
            )
            fig_hm.update_layout(height=600)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("Hourly view data missing.")

    # Q3: Hidden Gems
    with tabs[2]:
        st.header("💎 Hidden Gems & Local Favorites")
        st.markdown("Luoghi preferiti dai Residenti rispetto ai Turisti.")
        
        df_gems = data_store.get("view_gems")
        if df_gems is not None:
            # Formattazione
            st.dataframe(
                df_gems[['roi', 'Tourist', 'Resident', 'localness_index']].sort_values("localness_index", ascending=False),
                column_config={
                    "localness_index": st.column_config.ProgressColumn(
                        "Localness Score",
                        help="Higher means more residents",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "roi": "Location"
                },
                use_container_width=True,
                height=600
            )
        else:
            st.warning("Hidden Gems data missing.")

    # Q4: Seasonality
    with tabs[3]:
        st.header("📅 Best Season per Monument")
        sel_roi_seas = st.selectbox("Select Monument", df['roi'].unique())
        
        subset = df[df['roi'] == sel_roi_seas]
        season_trend = subset.groupby('month').size().reset_index(name='count')
        
        fig_seas = px.bar(season_trend, x='month', y='count', title=f"Seasonal Trend for {sel_roi_seas}")
        st.plotly_chart(fig_seas, use_container_width=True)


# --- PAGE 3: GEO SPATIAL ---

elif nav == "🌍 Geo-Spatial 3D":
    st.title("🌍 3D Spatial Analysis")
    
    # Filters specific to map
    col1, col2 = st.columns([3, 1])
    with col2:
        # MAP_STYLES rimosso nelle versioni recenti di pydeck
        available_styles = [
            "mapbox://styles/mapbox/dark-v10",
            "mapbox://styles/mapbox/light-v10",
            "mapbox://styles/mapbox/outdoors-v11",
            "mapbox://styles/mapbox/satellite-v9"
        ]
        map_style = st.selectbox("Map Style", available_styles, index=0)
        radius = st.slider("Hexagon Radius", 20, 200, 50)
    
    with col1:
        # PyDeck Hexagon Layer
        map_df = df_filtered[['latitude', 'longitude']].copy()

        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')

        map_df = map_df.dropna()

        layer = pdk.Layer(
            "HexagonLayer",
            map_df,
            get_position=["longitude", "latitude"],
            auto_highlight=True,
            elevation_scale=10,
            pickable=True,
            elevation_range=[0, 3000],
            extruded=True,
            coverage=1,
            radius=radius
        )
        
        view_state = pdk.ViewState(
            longitude=12.4964,
            latitude=41.9028,
            zoom=11,
            min_zoom=5,
            max_zoom=15,
            pitch=40.5,
            bearing=-27.36
        )
        
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Count: {elevationValue}"},
            map_style=map_style
        )
        
        st.pydeck_chart(r)

# --- PAGE 4: SEMANTIC ANALYSIS (TAGS) ---
elif nav == "🏷️ Semantic Analysis":
    st.title("🏷️ What are people saying?")
    st.markdown("Analisi semantica dei Tag associati alle foto.")
    
    df_tags = data_store.get("tags")
    
    if df_tags is not None:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            sel_roi_tag = st.selectbox("Select Location for Tag Cloud", df_tags['roi'].unique())
            top_tags = df_tags[df_tags['roi'] == sel_roi_tag].sort_values("count", ascending=False).head(20)
            st.dataframe(top_tags, use_container_width=True)
            
        with c2:
            st.subheader(f"Top Keywords for {sel_roi_tag}")
            # Treemap visualization as a structured word cloud alternative
            fig_tree = px.treemap(
                top_tags,
                path=['word'],
                values='count',
                color='count',
                color_continuous_scale='Greens',
                title=f"Semantic Landscape of {sel_roi_tag}"
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            
            st.markdown("### 💡 Insight")
            if "ruins" in top_tags['word'].values:
                st.info("Questo luogo è fortemente associato alla storia e all'archeologia.")
            elif "night" in top_tags['word'].values:
                st.info("Questo luogo è popolare per la vita notturna.")
    else:
        st.warning("Tag data missing.")

# --- PAGE 5: USER INTELLIGENCE ---
elif nav == "👥 User Intelligence":
    st.title("👥 User Profiling & Segmentation")
    
    if df_users is not None:
        # Scatter Plot dinamico
        st.subheader("User Behavior Clusters")
        
        fig_scatter = px.scatter(
            df_users.sample(min(5000, len(df_users))),
            x="days_active",
            y="unique_rois",
            color="user_type",
            size="total_photos",
            log_x=True,
            hover_data=["user_id"],
            title="Days Active vs Unique Locations Visited"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Stats per group
        st.subheader("Statistics by Group")
        group_stats = df_users.groupby("user_type").agg({
            "total_photos": "mean",
            "days_active": "mean",
            "unique_rois": "mean",
            "user_id": "count"
        }).rename(columns={"user_id": "count"}).reset_index()
        
        st.dataframe(group_stats, use_container_width=True)

# --- PAGE 6: TIME & TRENDS ---
elif nav == "📈 Time & Trends":
    st.title("📈 Temporal Analytics")
    
    tab1, tab2 = st.tabs(["Weekly Patterns", "Hourly Patterns"])
    
    with tab1:
        st.subheader("Day of Week Distribution")
        dow_counts = df.groupby('day_name').size().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).reset_index(name='count')
        
        fig_bar = px.bar(dow_counts, x='day_name', y='count', color='count', title="Activity by Day of Week")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab2:
        st.subheader("Hourly Distribution (Global)")
        hour_counts = df.groupby('hour').size().reset_index(name='count')
        fig_line = px.line(hour_counts, x='hour', y='count', markers=True, title="Activity by Hour of Day")
        st.plotly_chart(fig_line, use_container_width=True)
