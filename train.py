import io
import gc
import os

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

CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

IMG_SIZE = (224, 224)


def fine_tune_model(keras_model_path, feedbacks, epochs=1):

    model = None

    try:
        if not feedbacks:
            return False, None, "No feedback available"

        tf.keras.backend.clear_session()
        gc.collect()

        print("Loading model...")

        model = tf.keras.models.load_model(
            keras_model_path,
            compile=False
        )

        print("Model loaded")

        # Freeze everything
        for layer in model.layers:
            layer.trainable = False

        # Only train the final layer
        model.layers[-1].trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        for fb in feedbacks:

            if fb.correct_class not in CLASS_MAP:
                continue

            print("Processing feedback:", fb.id)

            img = Image.open(
                io.BytesIO(fb.image)
            ).convert('RGB')

            img = img.resize(IMG_SIZE)

            x = np.asarray(
                img,
                dtype=np.float32
            ) / 255.0

            x = np.expand_dims(x, axis=0)

            y = np.array(
                [CLASS_MAP[fb.correct_class]],
                dtype=np.int32
            )

            print("Training on one image...")

            model.fit(
                x,
                y,
                epochs=epochs,
                batch_size=1,
                verbose=1
            )

            del x
            del y
            del img

            gc.collect()

        print("Saving Keras model...")

        new_keras_path = "new_model.keras"

        model.save(
            new_keras_path
        )

        print("Converting to TFLite...")

        gc.collect()

        converter = tf.lite.TFLiteConverter.from_keras_model(
            model
        )

        tflite_model = converter.convert()

        new_tflite_path = "new_model.tflite"

        with open(
            new_tflite_path,
            "wb"
        ) as f:
            f.write(tflite_model)

        print("Fine tuning complete")

        return True, (
            new_keras_path,
            new_tflite_path
        ), None

    except Exception as e:

        print("Fine tuning error:", e)

        return False, None, str(e)

    finally:

        if model is not None:
            del model

        tf.keras.backend.clear_session()
        gc.collect()
