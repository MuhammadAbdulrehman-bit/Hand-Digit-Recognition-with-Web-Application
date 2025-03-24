import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

def preprocess_image(image): #preprocess the image 
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    
    image = ImageOps.invert(image)
    
    image = image.resize((28, 28))
    
    image = np.array(image) / 255.0
    
    image = image.reshape(1, 28, 28, 1)
    
    return image

@st.cache_resource  # Cache the model in memory
def load_model():
    return keras.models.load_model('mnist_cnn_model.keras')

model = load_model()

# Set up the Streamlit interface
st.title("Handwritten Digit Recognition")
st.write("Draw a digit in the box below and click Predict.")

canvas_result = st_canvas(
    fill_color="white",  
    stroke_color="black",  
    stroke_width=20,  
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


if st.button('Predict'):
    if canvas_result.image_data is not None:
        
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        img = img.convert('L')
        
        processed_image = preprocess_image(img)
        
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        

        st.image(img, caption='Your Drawing', use_column_width=True)
        st.write(f"**Predicted Digit:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        st.bar_chart(prediction[0])
    else:
        st.write("⚠️ Please draw a digit first!")

print("Done")