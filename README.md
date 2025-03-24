 # CNN Digit Recognition Web App

##  Overview
This project is a deep learning-based web application that allows users to draw a digit (0-9) on an interactive canvas. The application uses a Convolutional Neural Network (CNN) model trained on the MNIST dataset to predict the drawn digit. It provides both the classification result and a visual representation of the model's confidence across all possible digits.

##  Features
-  **Interactive Drawing Canvas**: Draw digits with your mouse or touchpad
-  **Real-time Prediction**: Instant digit classification using a pre-trained CNN model
-  **Confidence Visualization**: Bar chart showing prediction probabilities for all digits (0-9)
-  **Deep Learning Model**: CNN architecture optimized for handwritten digit recognition
-  **Web Interface**: User-friendly Streamlit application
-  **Lightweight**: Fast predictions with minimal computational requirements

##  Installation

###  Prerequisites
Before running the application, ensure you have:
- Python 3.7-3.10 installed only for tensorflow
- pip (Python package manager) available in your PATH

### 🔹 Setting Up the Environment
1. **Clone the repository**:

    ```bash
   git clone https://github.com/yourusername/digit-classifier.git
   cd digit-classifier

### Install dependencies:
The project requires several Python packages which can be installed using:
        
    
    pip install streamlit tensorflow opencv-python numpy matplotlib pillow

# Running the Application
   
   ### Starting the Web App
  To launch the application, run the following command in your python terminal or you can do the same in CMD but you will have to navigate to the folder first.

      streamlit run HandWrittenCNN.py
      
### Result:
- **Start a local Streamlit server**

- **Automatically open your default web browser**

- **Display the digit classification interface at http://localhost:8501**


 ## Using the Application
**Drawing Area:**
  Left-click and drag to draw your digit. The canvas supports both mouse and touch input

### Controls:
 - **Draw**: Naturally draw your digit
 
 - **Clear**: Reset the canvas
 
 - **Predict**: Classify your drawn digit

### Output:
Top prediction displayed prominently

 - Confidence percentages for all digits

 - Visual bar chart of prediction probabilities

## Model Details

### Architecture
The CNN model follows this architecture:

Copy
Input (28x28 grayscale) → 
Conv2D (32 filters) → ReLU → MaxPooling → 
Conv2D (64 filters) → ReLU → MaxPooling → 
Flatten → Dense (128 units) → ReLU → 
Dropout (0.5) → Output (10 units) → Softmax

### Training
- **Dataset**: MNIST (60,000 training images, 10,000 test images)

- **Epochs**: 15

- **Batch Size**: 64

- **Optimizer** : Adam

## Loss Function: Categorical Crossentropy

**Accuracy**: ~97% on test set

## Project Structure

digit-classifier/
├── HandWrittenCNN.py        # Main Streamlit application
├── CNN_handwritten.py      # Model training script
├── mnist_cnn_model.keras   # Pre-trained model weights
├── README.md               # This documentation


## License
MIT License

## Acknowledgments
- MNIST dataset creators

- TensorFlow/Keras team

- Streamlit developers![image](https://github.com/user-attachments/assets/0f8396ce-df2f-4227-a7a5-59fe14e06e00)

- ![image](https://github.com/user-attachments/assets/43fe21a0-f9ca-4dfb-9b2d-6d5faa9eb1fa)

