# Lung Cancer Detection Experiment Sandbox

This folder contains a standalone environment for testing Lung Cancer detection and Explainable AI (XAI) models.

## Structure
- **`data/`**: Contains downloaded sample X-ray images (Benign vs Malignant).
- **`saved_model/`**: Stores the downloaded Keras model (`best_chest_xray_model.h5`).
- **`results/`**: Output folder for XAI visualizations (Grad-CAM, LIME).
- **`setup_data.py`**: Downloads sample images from public repositories.
- **`download_model.py`**: Downloads pre-trained Pneumonia/Abnormal Chest X-ray model from Hugging Face.
- **`test_xai.py`**: Loads the pre-trained model and performs inference + XAI on a test image.

## Usage
1.  **Setup Data**:
    ```bash
    python setup_data.py
    ```
2.  **Download Model**:
    ```bash
    python download_model.py
    ```
3.  **Run Test & XAI**:
    ```bash
    python test_xai.py
    ```
    Results (heatmaps) will be saved in `results/`.
