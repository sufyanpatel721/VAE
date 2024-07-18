import streamlit as st
import requests
from PIL import Image
import io

st.title('Autoencoder Image Reconstruction')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Reconstructing...')

    # Send image to Flask app
    files = {'file': uploaded_file.getvalue()}
    response = requests.post('http://127.0.0.1:5000/predict', files=files)

    if response.status_code == 200:
        reconstructed_img_hex = response.json()['image']
        reconstructed_img_bytes = bytes.fromhex(reconstructed_img_hex)
        reconstructed_img = Image.open(io.BytesIO(reconstructed_img_bytes))
        st.image(reconstructed_img, caption='Reconstructed Image', use_column_width=True)
    else:
        st.write('Error: ' + response.json()['error'])
