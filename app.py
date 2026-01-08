import os
# Env vars removed to match working script environment

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_file_type, save_uploaded_file, create_spectrogram, preprocess_image, load_and_preprocess_image
from src.model_loader import ModelFactory
from src.xai_engine import XAIEngine

st.set_page_config(page_title="Unified Explainable AI Interface", layout="wide")

def main():
    st.title("Unified Explainable AI Interface")
    st.markdown("""
    This platform integrates Deepfake Audio Detection and Lung Cancer Detection into a single interface.
    Upload an audio file (.wav) or a chest X-ray image to get started.
    """)

    # Sidebar for inputs
    st.sidebar.header("Input Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Audio or Image", type=['wav', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_type = get_file_type(uploaded_file.name)
        st.sidebar.info(f"Detected File Type: {file_type.capitalize()}")

        # Model Selection
        model_options = []
        if file_type == 'audio':
            model_options = ['Custom_PB', 'ResNet50', 'MobileNet'] # These map to the same loaded model in our simplified loader, or we could implement switching if we had multiple models
        elif file_type == 'image':
            model_options = ['LungCancer_Best', 'DenseNet', 'AlexNet']
        
        selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)

        # XAI Selection
        # XAI Selection
        xai_options = ['LIME', 'Grad-CAM', 'SHAP']
        if selected_model_name == 'Custom_PB':
            xai_options.remove('Grad-CAM')
            
        selected_xai_methods = st.sidebar.multiselect("Select XAI Techniques", xai_options, default=['LIME'])

        # Process Input
        if st.sidebar.button("Analyze"):
            with st.spinner("Processing..."):
                # Save file
                file_path = save_uploaded_file(uploaded_file)
                
                # Preprocessing
                input_data = None
                display_image = None
                
                if file_type == 'audio':
                    st.subheader("Audio Analysis")
                    st.audio(file_path)
                    st.write("Generating Spectrogram...")
                    display_image = create_spectrogram(file_path)
                    if display_image is not None:
                        st.image(display_image, caption="Mel-Spectrogram", width=300)
                        input_data = preprocess_image(display_image)
                
                elif file_type == 'image':
                    st.subheader("Image Analysis")
                    display_image, input_data = load_and_preprocess_image(file_path)
                    st.image(display_image, caption="Uploaded Image", width=300)

                if input_data is not None:
                    # Load Model
                    try:
                        model_interface = ModelFactory.get_model(file_type, selected_model_name)
                        
                        # Prediction
                        prediction_result = model_interface.predict(input_data)
                        
                        st.success(f"Prediction: **{prediction_result['label']}**")
                        st.write(f"Probability: {prediction_result['probability']:.4f}")
                        
                        # XAI
                        if selected_xai_methods:
                            # Robustness check: if prediction failed, likely model is bad
                            if prediction_result.get("label") == "Model Error":
                                st.warning("Cannot run XAI because the model failed to load or predict.")
                            else:
                                st.header("Explainability Analysis")
                                try:
                                    xai_engine = XAIEngine(model_interface)
                                    
                                    # Create columns for side-by-side comparison
                                    cols = st.columns(len(selected_xai_methods))
                                    
                                    for idx, method in enumerate(selected_xai_methods):
                                        with cols[idx]:
                                            st.subheader(method)
                                            if method == 'LIME':
                                                with st.spinner("Running LIME..."):
                                                    fig = xai_engine.lime_explain(np.array(display_image))
                                                    st.pyplot(fig)
                                            elif method == 'Grad-CAM':
                                                with st.spinner("Running Grad-CAM..."):
                                                    fig = xai_engine.grad_cam_explain(input_data)
                                                    if fig:
                                                        st.pyplot(fig)
                                                    else:
                                                        st.warning("Grad-CAM failed (layer not found?)")
                                            elif method == 'SHAP':
                                                with st.spinner("Running SHAP (this may take a while)..."):
                                                    fig = xai_engine.shap_explain(np.array(display_image))
                                                    st.pyplot(fig)
                                except ValueError as ve:
                                    st.error(f"XAI Error: {ve}")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during XAI: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    else:
        st.info("Please upload a file to begin.")

    # Comparison Tab (Implemented as a separate section below for simplicity in single-page app, 
    # or could be a separate Streamlit tab if using st.tabs)
    
    st.markdown("---")
    st.header("Comparison & Details")
    st.write("Use the sidebar to select multiple XAI methods to compare them side-by-side above.")

if __name__ == "__main__":
    main()
