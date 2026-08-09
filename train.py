import io
import numpy as np
import tensorflow as tf
from PIL import Image

CLASSES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}
IMG_SIZE = (224, 224)


def build_dataset(feedbacks):
    images = []
    labels = []

    for fb in feedbacks:
        img = Image.open(io.BytesIO(fb.image)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        images.append(img_array)
        labels.append(CLASS_MAP[fb.correct_class])

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return x, y


def fine_tune_model(keras_model_path, feedbacks, epochs=5):
    """
    Returns (success: bool, new_model_paths: tuple or None, error: str or None)
    """
    try:
        x, y = build_dataset(feedbacks)

        if len(x) == 0:
            return False, None, "No usable feedback (correct_class didn't match any known class)"

        model = tf.keras.models.load_model(keras_model_path)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(x, y, epochs=epochs, batch_size=8)

        new_keras_path = 'new_model.keras'
        model.save(new_keras_path)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        new_tflite_path = 'new_model.tflite'
        with open(new_tflite_path, 'wb') as f:
            f.write(tflite_model)

        return True, (new_keras_path, new_tflite_path), None

    except Exception as e:
        return False, None, str(e)