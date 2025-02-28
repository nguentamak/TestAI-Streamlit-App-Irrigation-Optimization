import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Charger le modèle sauvegardé
MODEL_PATH = 'best_model_GradientBoosting.pkl'  # Remplacer par le bon fichier si nécessaire
model = joblib.load(MODEL_PATH)

# Initialiser un scaler si nécessaire (assurez-vous qu'il est compatible avec les données d'entraînement)
scaler = StandardScaler()

def predict_irrigation(temperature, humidity, rainfall):
    # Standardisation des entrées
    input_data = np.array([[temperature, humidity, rainfall]])
    input_scaled = scaler.fit_transform(input_data)  # Assurez-vous d'utiliser le scaler d'origine si possible
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)

# Interface utilisateur Streamlit
st.title("Optimisation de l'irrigation avec IA")
st.write("Entrez les paramètres pour prédire le besoin en irrigation.")

# Saisie utilisateur
temperature = st.number_input("Température (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidité (%)", min_value=0.0, max_value=100.0, value=50.0)
rainfall = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Prédire le besoin en irrigation"):
    prediction = predict_irrigation(temperature, humidity, rainfall)
    st.success(f"Besoin en irrigation estimé: {prediction} unités")
    
    # Génération d'un graphique simple
    fig, ax = plt.subplots()
    parameters = ['Température', 'Humidité', 'Pluviométrie']
    values = [temperature, humidity, rainfall]
    ax.bar(parameters, values, color=['red', 'blue', 'green'])
    ax.set_ylabel("Valeurs")
    ax.set_title("Paramètres d'entrée")
    st.pyplot(fig)
    
    # Génération d'une courbe de tendance
    temp_range = np.linspace(0, 50, 100)
    irrigation_trend = [predict_irrigation(t, humidity, rainfall) for t in temp_range]
    
    fig, ax = plt.subplots()
    ax.plot(temp_range, irrigation_trend, label="Besoin en irrigation vs Température", color='purple')
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Besoin en irrigation")
    ax.set_title("Courbe de tendance du besoin en irrigation en fonction de la température")
    ax.legend()
    st.pyplot(fig)
