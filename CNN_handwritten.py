import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

def preprocess_image(image):
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Invert colors (MNIST uses white digits on black background)
    image = ImageOps.invert(image)
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    
    # Reshape for CNN input: (28, 28, 1) with batch dimension
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Load your trained CNN model (fix the filename)
@st.cache_resource  # Cache the model in memory
def load_model():
    return keras.models.load_model('mnist_cnn_model.keras')

model = load_model()
# Set up the Streamlit interface
st.title("Handwritten Digit Recognition")
st.write("Draw a digit in the box below and click Predict.")

# Set up canvas for drawing
canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_color="black",  # Drawing color
    stroke_width=20,  # Increased stroke width for better visibility
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Button to make prediction
if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Convert canvas image to PIL Image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        # Convert to grayscale (single channel)
        img = img.convert('L')
        
        # Preprocess image for CNN model
        processed_image = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display results
        st.image(img, caption='Your Drawing', use_column_width=True)
        st.write(f"**Predicted Digit:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Optional: Show all probabilities
        st.bar_chart(prediction[0])
    else:
        st.write("⚠️ Please draw a digit first!")

print("Done")