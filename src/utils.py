import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

def get_file_type(filename):
    """
    Determines if the file is an audio or image file based on extension.
    Returns 'audio', 'image', or None.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.wav', '.mp3', '.flac']:
        return 'audio'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return 'image'
    return None

def save_uploaded_file(uploaded_file, save_dir='temp_files'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def create_spectrogram(audio_file_path, output_image_path='temp_spectrogram.png'):
    try:
        y, sr = librosa.load(audio_file_path)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        librosa.display.specshow(log_ms, sr=sr)
        plt.savefig(output_image_path)
        plt.close(fig)
        
        # Load the image to return it in a format suitable for the model
        # The original code loaded it with target_size=(224,224)
        image_data = load_img(output_image_path, target_size=(224, 224))
        return image_data
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def preprocess_image(image_data):
    img_array = img_to_array(image_data)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    # For direct image upload (Lung Cancer)
    image = load_img(image_file, target_size=target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return image, img_batch
