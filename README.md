# Unified Explainable AI Interface

## Project Overview
This project integrates two existing Explainable AI (XAI) systems into a single interactive platform capable of processing both audio and image data.
1.  **Deepfake Audio Detection**: Detects real vs. fake audio using neural networks and explains predictions using LIME, Grad-CAM, and SHAP.
2.  **Lung Cancer Detection**: Detects malignant tumors in chest X-rays (Demo mode using DenseNet/MobileNet) and provides visual explanations.

## Features
-   **Multi-modal Input**: Support for Audio (.wav) and Image (.jpg, .png).
-   **Model Selection**: Choose between compatible models for each input type.
-   **XAI Techniques**: Apply LIME, Grad-CAM, and SHAP to visualize model decisions.
-   **Comparison**: Side-by-side comparison of different XAI methods.
-   **Automatic Filtering**: Only relevant methods and models are shown based on input type.

## Setup and Installation

1.  **Clone the repository** (if not already done).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Generative AI Usage Statement
**Tools Used**: GitHub Copilot (Gemini 3 Pro)
**Purpose**:
-   **Code Refactoring**: Integrating code from original repositories into a modular structure (`src/utils.py`, `src/xai_engine.py`).
-   **Interface Design**: Generating the Streamlit application code (`app.py`).
-   **XAI Implementation**: Adapting LIME, Grad-CAM, and SHAP implementations for the unified interface.
-   **Documentation**: Drafting the README and code comments.

## Technical Report (Short)
### Design and Integration
The project uses a modular architecture:
-   `app.py`: The main Streamlit application handling UI and user interaction.
-   `src/model_loader.py`: Factory pattern to load appropriate models based on input type. Handles the loading of the pre-trained Audio model and the placeholder Image models.
-   `src/xai_engine.py`: Encapsulates the logic for LIME, Grad-CAM, and SHAP, providing a uniform API for the UI.
-   `src/utils.py`: Handles file processing, spectrogram generation, and image preprocessing.

### Selected Models and XAI Methods
-   **Audio**: Uses a pre-trained TensorFlow model (VGG16/MobileNet based) converted to detect Deepfakes from Mel-Spectrograms.
-   **Image**: Uses pre-trained DenseNet121/MobileNetV2 (ImageNet weights) as a demonstration for the Lung Cancer detection pipeline (due to missing original model weights).
-   **XAI**:
    -   **LIME**: Local Interpretable Model-agnostic Explanations.
    -   **Grad-CAM**: Gradient-weighted Class Activation Mapping (for CNNs).
    -   **SHAP**: SHapley Additive exPlanations (KernelExplainer).

### Improvements
-   Unified interface for multiple modalities.
-   Dynamic model and XAI method selection.
-   Side-by-side comparison of explanations.
