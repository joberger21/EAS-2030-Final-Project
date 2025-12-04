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
            # Try semicolon first (common for this dataset)
            try:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')
            except:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=',')
    except:
        return None
    return None

# Sidebar for Data Control
st.sidebar.header("Data Settings")

# 1. Try Automatic Loading
raw_df = None
mirrors = [
    "https://raw.githubusercontent.com/mdcurtis/UFC-Fight-Data/master/raw_total_fight_data.csv",
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/ufc_fight_stats.csv",
]

# Attempt load
with st.sidebar:
    st.write("Checking data sources...")
    for url in mirrors:
        raw_df = load_data_from_url(url)
        if raw_df is not None:
            st.success("✅ Data loaded from URL!")
            break

    # 2. Manual Fallback
    if raw_df is None:
        st.warning("⚠️ URL load failed. Please upload file.")
        uploaded_file = st.file_uploader("Upload 'raw_total_fight_data.csv'", type=['csv'])
        if uploaded_file is not None:
            try:
                # Reset file pointer just in case
                uploaded_file.seek(0)
                # Try semicolon
                try:
                    raw_df = pd.read_csv(uploaded_file, sep=';')
                    if len(raw_df.columns) < 5:
                         raise ValueError("Wrong separator")
                except:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file, sep=',')
                
                st.success("✅ File Uploaded!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

# --- MAIN LOGIC (Only runs if data exists) ---
if raw_df is not None:
    
    # --- DATA PROCESSING FUNCTION ---
    @st.cache_data
    def process_data(df):
        # Flexible Column Selection (Handles different dataset versions)
        required_cols = {
            'R_fighter': ['R_fighter', 'R_Fighter'],
            'R_SIG_STR_landed': ['R_SIG_STR_landed', 'R_avg_SIG_STR_landed'],
            'R_SIG_STR_att': ['R_SIG_STR_att', 'R_avg_SIG_STR_att'],
            'R_TD_landed': ['R_TD_landed', 'R_avg_TD_landed'],
            'R_SUB_att': ['R_SUB_att', 'R_avg_SUB_att'],
            'R_CTRL_time': ['R_CTRL_time', 'R_avg_CTRL_time_seconds', 'R_CTRL_time_seconds']
        }
        
        # Standardize column names
        df_clean = df.copy()
        for standard, options in required_cols.items():
            for opt in options:
                if opt in df_clean.columns:
                    df_clean.rename(columns={opt: standard}, inplace=True)
                    break
        
        # Check if we have what we need
        if 'R_SIG_STR_landed' not in df_clean.columns:
            return None, None

        data = df_clean[list(required_cols.keys())].copy()

        # Time Conversion
        def convert_time(x):
            try:
                if isinstance(x, str) and ':' in x:
                    m, s = map(int, x.split(':'))
                    return m*60 + s
                return float(x)
            except: return 0
            
        data['R_CTRL_time_sec'] = data['R_CTRL_time'].apply(convert_time)

        # Aggregate by Fighter
        fighter_stats = data.groupby('R_fighter').agg({
            'R_SIG_STR_landed': 'mean',
            'R_SIG_STR_att': 'mean',
            'R_TD_landed': 'mean',
            'R_SUB_att': 'mean',
            'R_CTRL_time_sec': 'mean',
            'R_fighter': 'count'
        }).rename(columns={'R_fighter': 'Fight_Count'})

        # Filter Veterans & Create Ratios
        veterans = fighter_stats[fighter_stats['Fight_Count'] >= 5].copy()
        veterans['Strike_Accuracy'] = veterans['R_SIG_STR_landed'] / (veterans['R_SIG_STR_att'] + 0.001)
        veterans['Strike_Volume'] = veterans['R_SIG_STR_landed']
        veterans['Grappling_Threat'] = veterans['R_TD_landed'] + veterans['R_SUB_att']
        
        # Final Clean
        features = ['Strike_Volume', 'Strike_Accuracy', 'Grappling_Threat', 'R_CTRL_time_sec']
        X = veterans[features].fillna(0)
        return X, veterans

    # Process Data
    X, veterans = process_data(raw_df)

    if X is not None:
        st.sidebar.success(f"Processed {len(X)} Veteran Fighters")

        # --- 2. PCA & K-MEANS ENGINE ---
        n_clusters = st.sidebar.slider("Number of Style Clusters", 2, 6, 4)

        # Run Math
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)

        # Create Main DataFrame
        df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=X.index)

        # Run Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_pca['Cluster'] = kmeans.fit_predict(coords)
        df_pca['Cluster'] = df_pca['Cluster'].astype(str)

        # Merge stats back for hover data
        df_plot = df_pca.join(X)

        # --- 3. DYNAMIC LABELS ---
        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)

        def get_label(pc):
            top_feature = loadings[pc].abs().idxmax()
            direction = "Pos" if loadings.loc[top_feature, pc] > 0 else "Neg"
            return f"{pc} ({top_feature} {direction})"

        x_label = get_label('PC1')
        y_label = get_label('PC2')

        # --- 4. VISUALIZATION (Plotly) ---
        st.subheader("Interactive Style Map")

        # Search Bar
        search_term = st.text_input("Find a Fighter (e.g., Jon Jones):", "")
        highlight_data = None

        if search_term:
            matches = [n for n in df_plot.index if search_term.lower() in n.lower()]
            if matches:
                target = matches[0]
                highlight_data = df_plot.loc[[target]]
                st.info(f"Showing location for: **{target}**")
            else:
                st.warning("Fighter not found.")

        # The Plot
        fig = px.scatter(
            df_plot, 
            x='PC1', y='PC2', 
            color='Cluster',
            hover_name=df_plot.index,
            hover_data=['Strike_Volume', 'Grappling_Threat', 'R_CTRL_time_sec'],
            title="UFC Fighter Clusters (Hover for Stats)",
            labels={'PC1': x_label, 'PC2': y_label},
            height=600
        )

        if highlight_data is not None:
            fig.add_scatter(
                x=highlight_data['PC1'], 
                y=highlight_data['PC2'], 
                mode='markers+text',
                marker=dict(symbol='star', size=25, color='yellow', line=dict(width=2, color='black')),
                text=highlight_data.index,
                textposition="top center",
                name='Search Result'
            )

        st.plotly_chart(fig, use_container_width=True)

        # --- 5. DOPPELGANGER ENGINE ---
        if search_term and highlight_data is not None:
            st.markdown("---")
            st.subheader(f"👥 The Doppelgänger Engine: Who fights like {target}?")
            
            target_coords = df_plot.loc[target, ['PC1', 'PC2']]
            distances = df_plot.apply(
                lambda row: distance.euclidean(row[['PC1', 'PC2']], target_coords), axis=1
            )
            similars = distances.sort_values().iloc[1:6]
            
            cols = st.columns(5)
            for i, (fighter, dist) in enumerate(similars.items()):
                score = int((1 / (1 + dist)) * 100)
                with cols[i]:
                    st.metric(label=fighter, value=f"{score}% Match")

        # --- 6. RAW DATA VIEW ---
        with st.expander("View Raw Data"):
            st.dataframe(df_plot)
    else:
        st.error("Could not find required columns in the data. Please ensure CSV has 'R_fighter', 'R_SIG_STR_landed', etc.")

else:
    st.info("Waiting for data... (Try uploading the CSV manually in the sidebar if the URL failed)")
