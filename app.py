import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Cargar el modelo y el tokenizador
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Puedes usar "distilgpt2" para un modelo más ligero
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Configurar la aplicación de Streamlit
st.title("Chatbot Local con GPT-2")
st.write("Escribe un mensaje y el chatbot te responderá.")

# Entrada de texto del usuario
user_input = st.text_input("Tú:")

if user_input:
    # Tokenizar la entrada del usuario
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Mover los tensores a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)

    # Generar una respuesta
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,  # Longitud máxima de la respuesta
            num_return_sequences=1,  # Número de respuestas a generar
            no_repeat_ngram_size=2,  # Evitar repeticiones
            top_k=50,  # Muestreo top-k
            top_p=0.95,  # Muestreo top-p (nucleus sampling)
            temperature=0.7,  # Controlar la creatividad
        )

    # Decodificar la respuesta
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Mostrar la respuesta
    st.text_area("Chatbot:", value=response, height=200)