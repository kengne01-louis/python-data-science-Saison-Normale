import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# Couleurs des pages
def set_background(page):
    if page == "Accueil":
        bg_color = "#5A37374D"  
        accent = "#15C08485"
    elif page == "Prédiction":
        bg_color = "#E8F5EA68"  
        accent = "#352E7DF1"
    elif page == "Visualisations":
        bg_color = "#FFE9E0FF" 
        accent = "#EF6C00"
    elif page == "À propos":
        bg_color = "#F3E5F5"  
        accent = "#8E24AA"
    else:
        bg_color = "#F0F4C3"  
        accent = "#827717"
    
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: #1A237E;
        }}
        /* Titres */
        h1, h2, h3, h4, h5, h6 {{
            color: {accent};
        }}
        /* Champs de saisie */
        .stNumberInput input {{
            background-color: #FFFFFFDD;
            color: #1B1B1B;
            font-weight: 600;
            border-radius: 10px;
            border: 2px solid {accent};
        }}
        /* Boutons */
        .stButton>button {{
            background-color: {accent};
            color: white;
            border-radius: 10px;
            padding: 8px 18px;
            font-weight: bold;
            font-size: 16px;
        }}
        .stButton>button:hover {{
            background-color: #0D47A1;
            color: #fff;
        }}
        /* Tableaux et données */
        div[data-testid="stDataFrame"] {{
            background-color: #FFFFFFCC;
            border-radius: 10px;
            padding: 10px;
        }}
        /* Graphiques */
        .js-plotly-plot .plotly {{
            background-color: transparent !important;
        }}


        </style>
    """, unsafe_allow_html=True)

sidebar_css = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #1E3A8A;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stTitle,
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        color: white;
    }
    </style>
    """
st.markdown(sidebar_css, unsafe_allow_html=True)
   

# Titre de l'application
st.title("APPLICATION DU MACHINE LEARNING POUR L'ESTIMATION DE LA RESISTANCE DU BETON.")
st.markdown("""
LE BUT PRINCIPAL EST DE PREDIRE LA RESISTANCE DU BETON.
""")
# Sidebar avec logo
with st.sidebar:
    # Ajout du logo
    try:
        # Logo 
        logo = Image.open("deco.jpg")  # le chemin
        st.image(logo, width=300)  # largeur de image
    
    except Exception as e:
        st.error(f"Logo non trouvé: {e}")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Prédiction", "Visualisations", "À propos"])


set_background(page)


# Fonction pour charger le modèle
def load_model():
    try:
        # Essayer joblib d'abord, puis pickle
        try:
            model = joblib.load('random_forest_model.joblib')
        except:
            model = pickle.load(open('random_forest_model.pkl', 'rb'))
        return model
    except Exception as e:
    
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Page d'accueil
if page == "Accueil":
    st.header("🏠 Accueil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Description et Modele")
        st.markdown("""
        La résistance du béton est un facteur essentiel pour garantir la durabilité et la sécurité des constructions.  
        Elle dépend de plusieurs composants. Dans notre contexte, nous avons texté deux modèles:
        DECISION TREE ET RANDOM FOREST dont **Random Forest** est notre meilleur modele.

        Les variables utilisées sont :

        - **cement** : Quantité de ciment (en kg/m³)  
        - **slag** : Quantité de laitier (en kg/m³)  
        - **ash** : Quantité de cendres volantes (en kg/m³)  
        - **water** : Quantité d'eau (en kg/m³)  
        - **superplastic** : Quantité de superplastifiant (en kg/m³)  
        - **coarseagg** : Quantité d'agrégats grossiers (en kg/m³)  
        - **fineagg** : Quantité d'agrégats fins (en kg/m³)  
        - **age** : Âge du béton (en jours)  
        - **strength** : Résistance à la compression (en MPa)
                    
        Le modele utilisé ici est RANDOM FOREST, pour prédire la résistance du béton en fonction des autres variables.♐👁♐
    """)

    
    with col2:
        st.subheader("Suivez les instructions")
        st.markdown("""
        1. Allez dans l'onglet **Prédiction**
        2. Entrez les valeurs des features
        3. Cliquez sur **Prédire**
        4. Visualisez les résultats
        """)
    
    # Afficher les informations du modèle chargé
    model = load_model()
    if model is not None:
        st.success("✅ Modèle chargé avec succès!")
        st.info(f"Nombre d'arbres dans la forêt: {model.n_estimators}")

# Page de prédiction
elif page == "Prédiction":
    st.header("✍👇✍ Prédiction ")
    
    model = load_model()
    
    if model is not None:
        # Section pour l'entrée des données
        st.subheader("Entrez les valeurs des features")
        
        # Les entrees 
        input_method = st.radio("Choisissez la méthode de saisie:", 
                                  ["Formulaire", "Appartir d'un fichier"]) 
        
        if input_method == "Formulaire":
            # Créer des champs d'entrée basés sur le nombre de features attendues
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                feature1 = st.number_input("cement", value=0.0)
                feature2 = st.number_input("slag", value=0.0)
                
            with col2:
                feature3 = st.number_input("ash", value=0.0)
                feature4 = st.number_input("water", value=0.0)
                
            with col3:
                feature5 = st.number_input("superplastic", value=0.0)
                feature6 = st.number_input("coarseagg", value=0.0)
            
            with col4:
                feature7 = st.number_input("fineagg", value=0.0)
                feature8 = st.number_input("age", value=0)
            
            # Créer le tableau d'entrée
            input_data = np.array([[feature1, feature2, feature3,feature4, feature5, feature6, feature7, feature8]])
            
            if st.button("Faire la Prédiction", type="primary"):
                try:
                    prediction = model.predict(input_data)
                    st.success(f"📗👀 **Prédiction:** {prediction[0]:.4f}Mpa")
                    
                    # Afficher des informations supplémentaires
                    with st.expander("Détails de la prédiction"):
                        st.write(f"Valeurs d'entrée: {input_data[0]}")
                        st.write(f"Modèle utilisé: Random Forest ({model.n_estimators} arbres)")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
        
        else:  # Charger le fichier CSV
            st.subheader("Importer le fichier CSV")
            uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Lire le fichier CSV
                    df = pd.read_csv(uploaded_file)
                    st.write("Aperçu des données chargées:")
                    st.dataframe(df.head())
                    
                    if st.button("Prédire sur le fichier", type="primary"):
                        predictions = model.predict(df)
                        df['Prediction'] = predictions
                        
                        st.success("Prédictions terminées!")
                        st.write("Résultats:")
                        st.dataframe(df)
                        
                        # Télécharger les résultats
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=" 📙✍Télécharger les prédictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier: {e}")

# Page de visualisations
elif page == "Visualisations":
    st.header("📈 Visualisations📊")
    
    model = load_model()
    
    if model is not None:
        # Feature Importance
        st.subheader("Importance des Features")
        
        if hasattr(model, 'feature_importances_'):
            # Créer un DataFrame pour l'importance des features
            feature_importance = pd.DataFrame({
                'feature': [f'Feature {i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Graphique d'importance des features
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(8), x='importance', y='feature', ax=ax)
            ax.set_title('- Importance des Features')
            st.pyplot(fig)
            
            # Afficher le tableau
            st.write("Détail de l'importance des features:")
            st.dataframe(feature_importance)
        else:
            st.warning("Impossible d'afficher l'importance des features pour ce modèle.")
        
        # Informations sur le modèle
        st.subheader("Informations du Modèle")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre d'arbres", model.n_estimators)
            st.metric("Profondeur max", str(model.max_depth) if model.max_depth else "None")
            
        with col2:
            st.metric("Samples split min", model.min_samples_split)
            st.metric("Samples leaf min", model.min_samples_leaf)

# Page À propos
elif page == "À propos":
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ### Application de Déploiement Random Forest
    
    **Fonctionnalités:**
    - 📊 Prédictions en temps réel
    - 📈 Visualisation de l'importance des features
    - 📁 Support des fichiers CSV
    - 🎯 Interface utilisateur intuitive
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("Téléphone: 659 060 681")
st.sidebar.markdown("Email: louiskngn01@gmail.com")
st.sidebar.markdown("---")
