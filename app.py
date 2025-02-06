import streamlit as st 
from tensorflow.keras.models import load_model
from PIL import Image 
import numpy as np

# Cargar el modelo
model = load_model('fashon_mnist.keras')

# Crear la interfaz de usuario
st.title("Clasificador Fashion MNIST")
st.write("Sube una imágen para clasificarla como una categoría de ropa.")

uploaded_file = st.file_uploader("Sube una imágen en escala de grises de 28x28 pixeles.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Procesar la imágen
    image = Image.open(uploaded_file).convert('L') # Convertir RGB a ByN
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0 # Normalizar
    ''' El primer 1 indica que sólo hay una imágen, luego las dimensiones
        y el último 1 indica que sólo hay un canal de color.'''
    img_array = img_array.reshape(1, 28, 28, 1)

    # Motrar la imágen subida
    st.image(image, caption='Imágen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(img_array)
    classes = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo", "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]
    st.write(f"Predicción: {classes[np.argmax(prediction)]}")