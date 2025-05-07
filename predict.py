import tensorflow as tf
import numpy as np
from PIL import Image

class ASLPredictor:
    def __init__(self, model_path='asl_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]  # 0-9 + a-z
    
    def predict(self, image_path):
        # Preprocess image
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        return self.class_names[np.argmax(predictions[0])]

# Usage example:
if __name__ == '__main__':
    predictor = ASLPredictor()
    print("Predicted sign:", predictor.predict('test_image.jpg'))