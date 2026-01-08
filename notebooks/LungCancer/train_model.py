import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def train_model():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_save_path = os.path.join(base_dir, 'saved_model', 'lung_cancer_mobilenet.h5')
    
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    # Hyperparameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    # Data Generators (Augmentation for small dataset)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Using 20% for validation
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Base Model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x) # 2 classes: benign, malignant

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Save
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save labels mapping
    class_indices = train_generator.class_indices
    # Invert mapping: {0: 'benign', 1: 'malignant'}
    labels = {v: k for k, v in class_indices.items()}
    print(f"Class Mappings: {labels}")

if __name__ == "__main__":
    train_model()
