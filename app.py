import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
import requests
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UFC Style Map", layout="wide")

st.title("🥊 The Geometry of Combat: UFC Style Analyzer")
st.markdown("""
This app uses **Principal Component Analysis (PCA)** and **K-Means Clustering** to mathematically analyze 
UFC fighting styles based on career statistics. 
""")

# --- 1. ROBUST DATA LOADING ---
@st.cache_data
def load_data_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Try semicolon first
            try:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')
            except:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=',')
    except:
        return None
    return None

# Sidebar for Data Control
st.sidebar.header("Data Settings")

# 1. Automatic Loading Logic
mirrors = [
    "https://raw.githubusercontent.com/mdcurtis/UFC-Fight-Data/master/raw_total_fight_data.csv",
    # We removed the Greco1899 mirror as it causes column mismatch errors
]

raw_df = None
data_source = "None"

# Try URLs first
with st.sidebar:
    st.write("Source: GitHub")
    for url in mirrors:
        temp_df = load_data_from_url(url)
        if temp_df is not None and 'R_fighter' in temp_df.columns:
            raw_df = temp_df
            data_source = "GitHub"
            st.success("✅ Data loaded from GitHub!")
            break

    st.markdown("---")
    st.write("Or Upload Manually:")
    # ALWAYS show the uploader so you can override the URL data
    uploaded_file = st.file_uploader("Upload 'raw_total_fight_data.csv'", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Reset pointer
            uploaded_file.seek(0)
            try:
                raw_df = pd.read_csv(uploaded_file, sep=';')
                if len(raw_df.columns) < 5: raise ValueError
            except:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, sep=',')
            
            data_source = "Upload"
            st.success("✅ File Uploaded! (Using this instead of URL)")
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN LOGIC ---
if raw_df is not None:
    
    # --- DATA PROCESSING ---
    @st.cache_data
    def process_data(df):
        df_clean = df.copy()
        
        # 1. Normalize Column Names (Lowercase strip)
        # This fixes mismatches like "R_fighter" vs "r_fighter"
        df_clean.columns = [c.strip() for c in df_clean.columns]
        
        # 2. Map known variations to Standard Names
        # Target: [R_fighter, R_SIG_STR_landed, R_SIG_STR_att, R_TD_landed, R_SUB_att, R_CTRL_time]
        
        # Helper to find column case-insensitively
        def find_col(target_partial):
            for c in df_clean.columns:
                if target_partial.lower() in c.lower():
                    return c
            return None

        # Build renaming map dynamically
        rename_map = {}
        
        # R_fighter
        col = find_col("R_fighter") or find_col("Red_fighter")
        if col: rename_map[col] = 'R_fighter'

        # SIG_STR
        col_land = find_col("R_SIG_STR_landed") or find_col("R_avg_SIG_STR_landed")
        col_att = find_col("R_SIG_STR_att") or find_col("R_avg_SIG_STR_att")
        if col_land: rename_map[col_land] = 'R_SIG_STR_landed'
        if col_att: rename_map[col_att] = 'R_SIG_STR_att'

        # TD
        col_td = find_col("R_TD_landed") or find_col("R_avg_TD_landed")
        if col_td: rename_map[col_td] = 'R_TD_landed'
        
        # SUB
        col_sub = find_col("R_SUB_att") or find_col("R_avg_SUB_att")
        if col_sub: rename_map[col_sub] = 'R_SUB_att'

        # CTRL
        col_ctrl = find_col("R_CTRL_time") or find_col("R_avg_CTRL_time")
        if col_ctrl: rename_map[col_ctrl] = 'R_CTRL_time'

        # Apply renaming
        df_clean.rename(columns=rename_map, inplace=True)
        
        # Check requirements
        reqs = ['R_fighter', 'R_SIG_STR_landed', 'R_SIG_STR_att', 'R_TD_landed']
        missing = [r for r in reqs if r not in df_clean.columns]
        
        if missing:
            return None, missing

        # 3. Clean Data
        data = df_clean[list(rename_map.values())].copy()
        
        # Time Convert
        def convert_time(x):
            try:
                if isinstance(x, str) and ':' in x:
                    m, s = map(int, x.split(':'))
                    return m*60 + s
                return float(x)
            except: return 0
            
        # If Control time is missing, fill with 0
        if 'R_CTRL_time' in data.columns:
            data['R_CTRL_time_sec'] = data['R_CTRL_time'].apply(convert_time)
        else:
            data['R_CTRL_time_sec'] = 0

        # Aggregation
        fighter_stats = data.groupby('R_fighter').agg({
            'R_SIG_STR_landed': 'mean',
            'R_SIG_STR_att': 'mean',
            'R_TD_landed': 'mean',
            'R_CTRL_time_sec': 'mean',
            'R_fighter': 'count'
        }).rename(columns={'R_fighter': 'Fight_Count'})
        
        # Handle subs if present
        if 'R_SUB_att' in data.columns:
            fighter_stats['R_SUB_att'] = data.groupby('R_fighter')['R_SUB_att'].mean()
        else:
            fighter_stats['R_SUB_att'] = 0

        # Filter
        veterans = fighter_stats[fighter_stats['Fight_Count'] >= 5].copy()
        veterans['Strike_Accuracy'] = veterans['R_SIG_STR_landed'] / (veterans['R_SIG_STR_att'] + 0.001)
        veterans['Strike_Volume'] = veterans['R_SIG_STR_landed']
        veterans['Grappling_Threat'] = veterans['R_TD_landed'] + veterans['R_SUB_att']
        
        features = ['Strike_Volume', 'Strike_Accuracy', 'Grappling_Threat', 'R_CTRL_time_sec']
        X = veterans[features].fillna(0)
        return X, veterans

    # Process
    X, veterans = process_data(raw_df)

    if X is not None:
        st.sidebar.success(f"Processed {len(X)} Fighters")
        
        # --- MATH & PLOTTING ---
        n_clusters = st.sidebar.slider("Clusters", 2, 6, 4)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=X.index)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_pca['Cluster'] = kmeans.fit_predict(coords).astype(str)
        df_plot = df_pca.join(X)
        
        # Labels
        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)
        def get_label(pc):
            feat = loadings[pc].abs().idxmax()
            return f"{pc} ({feat})"
            
        # Plot
        fig = px.scatter(
            df_plot, x='PC1', y='PC2', color='Cluster',
            hover_name=df_plot.index,
            title="UFC Style Map",
            labels={'PC1': get_label('PC1'), 'PC2': get_label('PC2')},
            height=600
        )
        
        # Search
        search = st.text_input("Find Fighter:", "")
        if search:
            m = [n for n in df_plot.index if search.lower() in n.lower()]
            if m:
                target = m[0]
                row = df_plot.loc[[target]]
                fig.add_scatter(x=row['PC1'], y=row['PC2'], mode='markers+text', 
                                marker=dict(symbol='star', size=25, color='yellow', line=dict(width=2)),
                                text=[target], name='Found')
                
                # Doppelganger
                dists = df_plot.apply(lambda r: distance.euclidean(r[['PC1','PC2']], row.loc[target, ['PC1','PC2']]), axis=1)
                st.write(f"Most similar to **{target}**: {', '.join(dists.sort_values().index[1:4])}")

        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # FAIL STATE
        st.error("❌ Data loaded, but columns were missing.")
        st.write("Debug info - Columns found in file:", raw_df.columns.tolist())
        st.warning("Please upload 'raw_total_fight_data.csv' using the sidebar.")

else:
    st.info("👈 Please upload 'raw_total_fight_data.csv' in the sidebar.")
