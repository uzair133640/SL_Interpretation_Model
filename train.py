import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 36

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomFlip("horizontal"),
])

def create_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        
        # Conv Blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.45),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Data loading
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'split_data/train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'  # Should match sparse_categorical setup

    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'split_data/val',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Training
    model = create_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callback_list
    )
    
    # Final evaluation
    test_ds = tf.keras.utils.image_dataset_from_directory(
        'split_data/test',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    model.load_weights('best_model.keras')
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    
    model.save('asl_model.keras')

if __name__ == '__main__':
    main()