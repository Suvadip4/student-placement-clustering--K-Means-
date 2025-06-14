import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time

# -------------------------
# 🎯 Configurations
# -------------------------
st.set_page_config(page_title="Student Placement Clustering", layout="centered")
st.title("🎓 Student Placement Clustering App (K-Means)")
st.write("Upload a dataset to cluster students into High, Medium, or Low placement chance groups.")

# -------------------------
# 📦 Utility Functions
# -------------------------

def validate_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def preprocess_data(df):
    binary_map = {'Yes': 1, 'No': 0}
    df_encoded = df.copy()
    df_encoded['ExtracurricularActivities'] = df_encoded['ExtracurricularActivities'].map(binary_map)
    df_encoded['PlacementTraining'] = df_encoded['PlacementTraining'].map(binary_map)
    return df_encoded

def scale_features(df, feature_columns):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_columns])
    return X_scaled, scaler

def cluster_students(X_scaled, feature_columns, df_encoded):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_encoded['Cluster'] = clusters

    # Ranking logic
    cluster_means = df_encoded.groupby('Cluster')[feature_columns].mean().mean(axis=1)
    ranking = cluster_means.sort_values(ascending=False)
    label_map = {
        ranking.index[0]: 'High',
        ranking.index[1]: 'Medium',
        ranking.index[2]: 'Low'
    }
    df_encoded['PlacementChance'] = df_encoded['Cluster'].map(label_map)
    return df_encoded, label_map

def create_final_df(original_df, df_encoded):
    result_df = original_df.copy()
    result_df['Cluster'] = df_encoded['Cluster']
    result_df['PlacementChance'] = df_encoded['PlacementChance']
    return result_df
def plot_pca_clusters(df_encoded, X_scaled):
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    df_encoded['PCA1'] = pca_components[:, 0]
    df_encoded['PCA2'] = pca_components[:, 1]

    palette = {"High": "green", "Medium": "orange", "Low": "red"}

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_encoded,
        x='PCA1',
        y='PCA2',
        hue='PlacementChance',
        palette=palette,
        alpha=0.7,
        s=60,
        ax=ax
    )
    ax.set_title("K-Means Student Clusters (PCA Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Placement Chance")
    st.pyplot(fig)

# -------------------------
# 📂 File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# -------------------------
# 🚀 Main App Logic
# -------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        feature_columns = [
            'CGPA', 'IQ', 'Internships', 'Projects',
            'Workshops/Certifications', 'AptitudeTestScore',
            'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
            'ExtracurricularActivities', 'PlacementTraining'
        ]

        # ✅ Validate
        validate_columns(df, feature_columns)

        # 🔧 Process
        df_encoded = preprocess_data(df)
        X_scaled, _ = scale_features(df_encoded, feature_columns)
        df_encoded, label_map = cluster_students(X_scaled, feature_columns, df_encoded)
        result_df = create_final_df(df, df_encoded)

        # 📊 Display Data
        st.subheader("📊 Processed Data with Placement Clusters:")
        st.dataframe(result_df[['StudentID','Cluster', 'PlacementChance'] + feature_columns].head())

        # 📥 Download Option
        csv = result_df.to_csv(index=False)
        st.download_button("📥 Download Clustered Dataset", data=csv, file_name='clustered_placement_data.csv', mime='text/csv')

        # 📈 PCA Visualization
        if st.button("📊 Show Cluster Visualization"):
            st.title("Cluster Visualization")
            with st.spinner("Creating Graph"):
                time.sleep(0.3)
                plot_pca_clusters(df_encoded, X_scaled)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")