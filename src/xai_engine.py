import numpy as np
import tensorflow as tf
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
import shap
from skimage.segmentation import slic
import warnings

class XAIEngine:
    def __init__(self, model_interface):
        self.model_interface = model_interface
        self.model = model_interface.get_model()

    def lime_explain(self, image_data, num_samples=100):
        # image_data is (224, 224, 3) or similar, not batch
        explainer = lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            if images.dtype == np.uint8:
                images = images.astype('float32') / 255.0
            elif np.max(images) > 1.0:
                 images = images / 255.0
            return self.model.predict(images)

        if len(image_data.shape) == 4:
            image_data = image_data[0]
            
        explanation = explainer.explain_instance(
            image_data.astype('double'), 
            predict_fn, 
            top_labels=5, 
            hide_color=0, 
            num_samples=num_samples
        )
        
        top_pred_class = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_pred_class, 
            positive_only=False, 
            num_features=10, 
            hide_rest=False
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(mark_boundaries(temp, mask))
        ax.set_title(f"LIME Explanation")
        ax.axis('off')
        return fig

    def grad_cam_explain(self, image_data, layer_name=None):
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
            
        if layer_name is None:
            # Try to find a suitable layer
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            return None

        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(layer_name).output, self.model.output]
            )
        except Exception as e:
            print(f"Error creating Grad-CAM model: {e}")
            return None

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_data)
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (image_data.shape[2], image_data.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        original_img = image_data[0]
        if original_img.max() <= 1.0:
            original_img = np.uint8(original_img * 255)
            
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(superimposed_img)
        ax.set_title(f"Grad-CAM (Layer: {layer_name})")
        ax.axis('off')
        return fig

    def shap_explain(self, image_data, num_samples=50):
        # Simplified SHAP
        if len(image_data.shape) == 4:
            image_data = image_data[0]
            
        def mask_image(zs, segmentation, image, background=None):
            if background is None:
                background = image.mean((0,1))
            out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
            for i in range(zs.shape[0]):
                out[i,:,:,:] = image
                for j in range(zs.shape[1]):
                    if zs[i,j] == 0:
                        out[i][segmentation == j,:] = background
            return out

        def f(z):
            # Preprocess for model
            imgs = mask_image(z, segments_slic, image_data, 0)
            # Normalize if needed
            if imgs.max() > 1.0:
                imgs = imgs / 255.0
            return self.model.predict(imgs)

        segments_slic = slic(image_data, n_segments=50, compactness=10, sigma=1, start_label=1)
        
        explainer = shap.KernelExplainer(f, np.zeros((1,50)))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(np.ones((1,50)), nsamples=num_samples)
            
        # Visualization
        # We will just show the top class explanation
        # shap_values is a list of arrays (one for each class)
        # We need to find which class is predicted
        preds = self.model.predict(np.expand_dims(image_data/255.0, axis=0))
        top_class = np.argmax(preds)
        
        # Get shap values for top class
        # shap_values[top_class] is (1, 50)
        
        # Create a colored mask
        from matplotlib.colors import LinearSegmentedColormap
        colors = []
        for l in np.linspace(1,0,100):
            colors.append((245/255,39/255,87/255,l))
        for l in np.linspace(0,1,100):
            colors.append((24/255,196/255,93/255,l))
        cm = LinearSegmentedColormap.from_list("shap", colors)
        
        def fill_segmentation(values, segmentation):
            out = np.zeros(segmentation.shape)
            for i in range(len(values)):
                out[segmentation == i+1] = values[i] # +1 because start_label=1
            return out
            
        m = fill_segmentation(shap_values[top_class][0], segments_slic)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image_data.astype(np.uint8))
        max_val = np.max(np.abs(shap_values[top_class][0]))
        ax.imshow(m, cmap=cm, vmin=-max_val, vmax=max_val, alpha=0.5)
        ax.set_title(f"SHAP Explanation")
        ax.axis('off')
        
        return fig
