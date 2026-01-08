import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore

# Add parent directory to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.xai_engine import XAIEngine

# Paths
MODEL_PATH = os.path.join(current_dir, 'saved_model', 'best_chest_xray_model.h5')
# Data was downloaded into lung_cancer_experiments/data by setup_data.py
# We use 'malignant' samples as proxies for 'Abnormal/Pneumonia' to test the model's positive class detection
DATA_DIR = os.path.join(current_dir, 'data', 'malignant') 
RESULTS_DIR = os.path.join(current_dir, 'results')

class LungCancerModelWrapper:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print("Model Loaded Successfully.")
        
        # Identify Conv layer for Grad-CAM
        self.last_conv_layer_name = None
        # Loop to find the last Conv2D layer
        for layer in reversed(self.model.layers):
             if 'conv' in layer.name.lower():
                 self.last_conv_layer_name = layer.name
                 break
        
        if self.last_conv_layer_name:
            print(f"Targeting Layer for Grad-CAM: {self.last_conv_layer_name}")
        else:
            print("Warning: No Convolutional Layer found for Grad-CAM.")

    def get_model(self):
        return self.model
    
    def predict(self, x):
        return self.model.predict(x)

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Rescale 
    return img_array, img

def main():
    print("Loading Model...")
    wrapper = LungCancerModelWrapper(MODEL_PATH)
    engine = XAIEngine(wrapper)

    # Pick a random image
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print("No images found to test.")
        return
    
    test_file = files[0] 
    img_path = os.path.join(DATA_DIR, test_file)
    print(f"Testing on image: {img_path}")
    
    input_data, original_img = load_image(img_path)
    
    # Prediction
    preds = wrapper.predict(input_data)
    print(f"Raw Prediction: {preds}")
    
    # Handle Binary vs Categorical
    if preds.shape[-1] == 1:
        # Binary: prob < 0.5 = Normal, >=0.5 = Pneumonia
        prob = preds[0][0]
        label = "Pneumonia/Abnormal" if prob >= 0.5 else "Normal"
        print(f"Result: {label} (Prob: {prob:.4f})")
    else:
        # Categorical
        predicted_class = np.argmax(preds[0])
        label = f"Class {predicted_class}"
        print(f"Result: {label}")

    # Ensure results dir exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Grad-CAM
    print("Running Grad-CAM...")
    try:
        fig_gradcam = engine.grad_cam_explain(input_data)
        if fig_gradcam:
            save_path = os.path.join(RESULTS_DIR, f'gradcam_{test_file}.png')
            fig_gradcam.savefig(save_path)
            print(f"Grad-CAM saved to {save_path}")
        else:
            print("Grad-CAM failed (None returned)")
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    # LIME
    print("Running LIME...")
    try:
        fig_lime = engine.lime_explain(input_data[0], num_samples=50) 
        save_path = os.path.join(RESULTS_DIR, f'lime_{test_file}.png')
        fig_lime.savefig(save_path)
        print(f"LIME saved to {save_path}")
    except Exception as e:
        print(f"LIME error: {e}")

if __name__ == "__main__":
    main()
