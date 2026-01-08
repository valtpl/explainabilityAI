import os
import tensorflow as tf
import numpy as np
import random


class ModelInterface:
    def predict(self, input_data):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def get_model(self):
        raise NotImplementedError("Subclasses must implement get_model method")



class Keras3Wrapper:
    """Wrapper for Keras 3 models (TFSMLayer) to behave like Keras 2 models for XAI."""
    def __init__(self, model):
        self.model = model
        
    def predict(self, x):
        pred = self.model.predict(x)
        if isinstance(pred, dict):
            # Return the first value (assuming single output model)
            return list(pred.values())[0]
        return pred
        
    @property
    def layers(self):
        # TFSMLayer hides layers, so Grad-CAM won't work easily.
        return []
        
    def __getattr__(self, name):
        return getattr(self.model, name)

class AudioModel(ModelInterface):
    def __init__(self, model_name='MobileNet'):
        print(f"Loading generic audio model: {model_name}...")
        self.class_names = ['real', 'fake']
        
        if model_name == 'Custom_PB':
             model_path = os.path.join(os.getcwd(), 'saved_model', 'model')
             if not os.path.exists(model_path):
                 model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_model', 'model')
             
             print("Loading Custom PB model via tf.saved_model.load...")
             try:
                self.raw_model = tf.saved_model.load(model_path)
                
                # Wrapper to make raw model behave like Keras model for predict
                class RawWrapper:
                    def __init__(self, raw_model):
                        self.raw_model = raw_model
                    
                    def predict(self, x):
                        # Default signature is usually what we want
                        infer = self.raw_model.signatures['serving_default']
                        # x needs to be constant tensor, float32
                        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                        out = infer(x_tensor)
                        # out is a dict {'dense_1': tensor, ...} or similar
                        # We return the first value found
                        return list(out.values())[0].numpy()
                        
                self.model = RawWrapper(self.raw_model)
                self.is_keras3 = False
                self.last_conv_layer_name = None # Cannot do gradcam on raw graph easily
                return
                
             except Exception as e:
                print(f"Loading failed: {e}")
                # Fallback to MobileNet
                print("Falling back to generic MobileNetV2 (Random Weights/ImageNet) because loading failed.")
                base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
                self.last_conv_layer_name = 'Conv_1'
        elif model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
            self.last_conv_layer_name = 'conv5_block3_out'
        else:
            # Default to MobileNetV2 if unknown or MobileNet requested
            base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
            self.last_conv_layer_name = 'Conv_1'
            
        # Add a custom head for 2 classes (Real vs Fake)
        # We perform this adaptation so the model structure is valid for the XAI tools (GradCAM needs a convolution layer)
        # and the output matches the app's expectation (2 classes).
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        self.is_keras3 = False # Standard Keras models are fine

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        
        if self.is_keras3 or isinstance(prediction, dict):
             if isinstance(prediction, dict):
                 prediction = list(prediction.values())[0] # Expecting {'output_0': probabilities}
        
        class_idx = np.argmax(prediction)
        return {
            "label": self.class_names[class_idx],
            "probability": float(np.max(prediction)),
            "predictions": prediction,
            "class_idx": class_idx,
            "class_names": self.class_names
        }
    
    def get_model(self):
        if hasattr(self, 'is_keras3') and self.is_keras3:
            return Keras3Wrapper(self.model)
        return self.model

class ImageModel(ModelInterface):
    def __init__(self, model_name):
        print(f"Loading image model {model_name}...")
        if model_name == 'DenseNet':
            self.model = tf.keras.applications.DenseNet121(weights='imagenet')
        elif model_name == 'AlexNet':
             self.model = tf.keras.applications.MobileNetV2(weights='imagenet')
        else:
             self.model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        self.class_names = [f"Class {i}" for i in range(1000)]

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        class_idx = np.argmax(prediction)
        return {
            "label": self.class_names[class_idx],
            "probability": float(np.max(prediction)),
            "predictions": prediction,
            "class_idx": class_idx,
            "class_names": self.class_names
        }

    def get_model(self):
        return self.model

class LungCancerModel(ModelInterface):
    def __init__(self, model_name='BestModel'):
        print(f"Loading Lung Cancer model: {model_name}...")
        self.class_names = ['Normal', 'Pneumonia/Abnormal']
        self.model = None
        self.last_conv_layer_name = None
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'notebooks', 'LungCancer', 'saved_model', 'best_chest_xray_model.h5')
        
        if not os.path.exists(model_path):
             print(f"Model not found at {model_path}. Please run download_model.py in lung_cancer_experiments.")
             return

        try:
            # We use compile=False to avoid issues with optimizer states in older models
            # and to prevent the 'compiled metrics' warning.
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully.")
            
            # Find conv layer
            for layer in reversed(self.model.layers):
                 if 'conv' in layer.name.lower():
                     self.last_conv_layer_name = layer.name
                     break
        except Exception as e:
            print(f"CRITICAL: Failed to load Lung Cancer model: {e}")
            self.model = None

    def predict(self, input_data):
        if self.model is None:
            # Return a safe dummy response so the UI doesn't crash
            return {
                "label": "Model Error",
                "probability": 0.0,
                "predictions": np.array([0]),
                "class_idx": -1,
                "class_names": self.class_names
            }
            
        prediction = self.model.predict(input_data)
        
        # Logic from test_xai.py
        if prediction.shape[-1] == 1:
            prob = prediction[0][0]
            label = self.class_names[1] if prob >= 0.5 else self.class_names[0]
            probability = prob if prob >= 0.5 else 1 - prob
            class_idx = 1 if prob >= 0.5 else 0
        else:
            class_idx = np.argmax(prediction)
            label = self.class_names[class_idx]
            probability = float(np.max(prediction))
            
        return {
            "label": label,
            "probability": float(probability),
            "predictions": prediction,
            "class_idx": class_idx,
            "class_names": self.class_names
        }

    def get_model(self):
        return self.model

class ModelFactory:
    @staticmethod
    def get_model(input_type, model_name):
        if input_type == 'audio':
            # We now use standard models selected by name, ignoring the file system .pb model
            return AudioModel(model_name)
        elif input_type == 'image':
            if model_name == 'LungCancer_Best':
                return LungCancerModel(model_name)
            return ImageModel(model_name)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
