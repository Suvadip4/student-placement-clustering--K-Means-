import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("🎓 Student Placement Clustering App (K-Means)")
st.write("Upload a dataset to cluster students into High, Medium, or Low placement chance groups.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        # One-hot encode specific categorical columns
        df_encoded = pd.get_dummies(df, columns=['ExtracurricularActivities', 'PlacementTraining'], drop_first=True)

        # Define features
        features = [
            'CGPA',
            'Internships',
            'Projects',
            'Workshops/Certifications',
            'AptitudeTestScore',
            'SoftSkillsRating',
            'SSC_Marks',
            'HSC_Marks',
            'ExtracurricularActivities_Yes',
            'PlacementTraining_Yes'
        ]
        X = df_encoded[features]

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_encoded['Cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze cluster means (based on CGPA or composite score)
        cluster_means = df_encoded.groupby('Cluster')[features].mean().mean(axis=1)
        ranking = cluster_means.sort_values(ascending=False)

        # Assign labels
        placement_labels = {
            ranking.index[0]: 'High',
            ranking.index[1]: 'Medium',
            ranking.index[2]: 'Low'
        }
        df_encoded['PlacementChance'] = df_encoded['Cluster'].map(placement_labels)

        # Show output
        st.subheader("📊 Processed Data with Placement Clusters:")
        st.dataframe(df_encoded[['Cluster', 'PlacementChance'] + features].head())

        # Download option
        output_file = df_encoded.copy()
        csv = output_file.to_csv(index=False)
        st.download_button("📥 Download Clustered Dataset", data=csv, file_name='clustered_placement_data.csv', mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
