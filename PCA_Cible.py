import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Analyse Cancer du Sein",
    page_icon="🔬",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .diagnostic-benign {
        color: green;
        font-weight: bold;
    }
    .diagnostic-malign {
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Définition des colonnes
COLUMN_NAMES = [
    "ID", "Épaisseur_De_Tumeur", "Uniformité_Taille_Cellule", "Uniformité_Forme_Cellule",
    "Adhésion_Marginale", "Taille_Cellule_Epithéliale_Simple", "Noyaux_Nus",
    "Chromatine_Sans_Intérêt", "Nucléoles_normaux", "Mitose", "Class"
]

# Mapping des classes pour le diagnostic
DIAGNOSTIC_MAPPING = {
    2: "Bénin",
    4: "Malin"
}

def load_and_preprocess_data(file):
    """Charge et prétraite les données."""
    df = pd.read_csv(file, names=COLUMN_NAMES)
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Mapping du diagnostic
    df['Diagnostic'] = df['Class'].map(DIAGNOSTIC_MAPPING)
    
    # Gestion des valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.any():
        handling_method = st.radio(
            "Méthode de gestion des valeurs manquantes:",
            ["Supprimer", "Remplacer par la moyenne"]
        )
        if handling_method == "Supprimer":
            df.dropna(inplace=True)
        else:
            df.fillna(df.mean(), inplace=True)
    
    return df

def create_correlation_heatmap(df):
    """Crée une carte de chaleur des corrélations."""
    # Exclure les colonnes non numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        title="Matrice de Corrélation",
        color_continuous_scale="RdBu"
    )
    return fig

def show_diagnostic_summary(df):
    """Affiche un résumé des diagnostics."""
    diagnostic_counts = df['Diagnostic'].value_counts()
    total_cases = len(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution des Diagnostics")
        fig = px.pie(
            values=diagnostic_counts.values,
            names=diagnostic_counts.index,
            title="Répartition des cas bénins et malins",
            color=diagnostic_counts.index,
            color_discrete_map={"Bénin": "green", "Malin": "red"}
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("📋 Résumé des Diagnostics")
        st.write(f"**Nombre total de cas:** {total_cases}")
        st.write(f"**Cas bénins:** {diagnostic_counts.get('Bénin', 0)} ({(diagnostic_counts.get('Bénin', 0)/total_cases*100):.1f}%)")
        st.write(f"**Cas malins:** {diagnostic_counts.get('Malin', 0)} ({(diagnostic_counts.get('Malin', 0)/total_cases*100):.1f}%)")

def perform_pca_analysis(df, selected_features, n_components):
    """Effectue l'analyse PCA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_features])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du DataFrame PCA
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df["Diagnostic"] = df["Diagnostic"]
    
    # Calcul des contributions des features
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=selected_features
    )
    
    return pca_df, pca.explained_variance_ratio_, loadings

def main():
    st.title("🔬 Analyse Avancée du Cancer du Sein")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("Paramètres d'analyse")
        uploaded_file = st.file_uploader(
            "Charger les données (CSV)",
            type="csv"
        )
        
        if uploaded_file:
            st.info("""
            **Guide de lecture du diagnostic:**
            - 2 = Bénin (non cancéreux)
            - 4 = Malin (cancéreux)
            """)
    
    if uploaded_file is not None:
        # Chargement et prétraitement
        df = load_and_preprocess_data(uploaded_file)
        
        # Onglets pour différentes analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Diagnostic",
            "📊 Exploration des données",
            "🔄 Analyse PCA",
            "📈 Visualisations avancées"
        ])
        
        with tab1:
            st.header("Analyse du Diagnostic")
            show_diagnostic_summary(df)
            
            # Caractéristiques moyennes par diagnostic
            st.subheader("📊 Caractéristiques par type de diagnostic")
            features_mean = df.groupby('Diagnostic')[COLUMN_NAMES[1:-1]].mean()
            
            # Visualisation radar des caractéristiques
            fig = go.Figure()
            for diagnostic in df['Diagnostic'].unique():
                fig.add_trace(go.Scatterpolar(
                    r=features_mean.loc[diagnostic],
                    theta=features_mean.columns,
                    fill='toself',
                    name=diagnostic
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                title="Profil moyen des caractéristiques par diagnostic"
            )
            st.plotly_chart(fig)
        
        with tab2:
            st.header("Exploration des données")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Aperçu des données")
                st.dataframe(df.head())
            
            with col2:
                st.subheader("Statistiques descriptives")
                st.dataframe(df.describe())
            
            st.subheader("Matrice de corrélation")
            correlation_fig = create_correlation_heatmap(df)
            st.plotly_chart(correlation_fig, use_container_width=True)
        
        with tab3:
            st.header("Analyse en Composantes Principales")
            
            # Sélection des features
            features = df.columns.drop(["ID", "Class", "Diagnostic"]).tolist()
            selected_features = st.multiselect(
                "Sélectionner les variables",
                features,
                default=features
            )
            
            if len(selected_features) >= 2:
                n_components = st.slider(
                    "Nombre de composantes principales",
                    2, min(len(selected_features), 10), 2
                )
                
                # Effectuer PCA
                pca_df, explained_variance, loadings = perform_pca_analysis(
                    df, selected_features, n_components
                )
                
                # Affichage des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Variance expliquée")
                    fig_variance = px.bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=explained_variance,
                        labels={"x": "Composante", "y": "Variance expliquée"}
                    )
                    st.plotly_chart(fig_variance)
                
                with col2:
                    st.subheader("Projection PCA")
                    if n_components >= 3:
                        fig_pca = px.scatter_3d(
                            pca_df,
                            x="PC1", y="PC2", z="PC3",
                            color="Diagnostic",
                            title="Projection PCA 3D",
                            color_discrete_map={"Bénin": "green", "Malin": "red"}
                        )
                    else:
                        fig_pca = px.scatter(
                            pca_df,
                            x="PC1", y="PC2",
                            color="Diagnostic",
                            title="Projection PCA 2D",
                            color_discrete_map={"Bénin": "green", "Malin": "red"}
                        )
                    st.plotly_chart(fig_pca)
                
                # Contribution des variables
                st.subheader("Contribution des variables")
                fig_loadings = px.imshow(
                    loadings,
                    aspect="auto",
                    title="Contribution des variables aux composantes principales"
                )
                st.plotly_chart(fig_loadings)
                
                # Export des résultats
                st.download_button(
                    label="📥 Télécharger les résultats PCA",
                    data=pca_df.to_csv(index=False).encode('utf-8'),
                    file_name="resultats_pca.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning("⚠️ Veuillez sélectionner au moins deux variables.")
        
        with tab4:
            st.header("Visualisations avancées")
            
            # Distribution des diagnostics
            fig_dist = px.histogram(
                df,
                x="Diagnostic",
                title="Distribution des diagnostics",
                color="Diagnostic",
                color_discrete_map={"Bénin": "green", "Malin": "red"}
            )
            st.plotly_chart(fig_dist)
            
            # Boxplots pour chaque feature
            selected_feature = st.selectbox(
                "Sélectionner une variable pour le boxplot",
                features
            )
            fig_box = px.box(
                df,
                y=selected_feature,
                color="Diagnostic",
                title=f"Distribution de {selected_feature} par diagnostic",
                color_discrete_map={"Bénin": "green", "Malin": "red"}
            )
            st.plotly_chart(fig_box)

if __name__ == "__main__":
    main()