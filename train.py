import io
import gc

import numpy as np
import tensorflow as tf
from PIL import Image


CLASSES = [
    'Abyssinian',
    'Bengal',
    'Birman',
    'Bombay',
    'British_Shorthair',
    'Egyptian_Mau',
    'Maine_Coon',
    'Persian',
    'Ragdoll',
    'Russian_Blue',
    'Siamese',
    'Sphynx',
    'american_bulldog',
    'american_pit_bull_terrier',
    'basset_hound',
    'beagle',
    'boxer',
    'chihuahua',
    'english_cocker_spaniel',
    'english_setter',
    'german_shorthaired',
    'great_pyrenees',
    'havanese',
    'japanese_chin',
    'keeshond',
    'leonberger',
    'miniature_pinscher',
    'newfoundland',
    'pomeranian',
    'pug',
    'saint_bernard',
    'samoyed',
    'scottish_terrier',
    'shiba_inu',
    'staffordshire_bull_terrier',
    'wheaten_terrier',
    'yorkshire_terrier'
]


CLASS_MAP = {
    name: index
    for index, name in enumerate(CLASSES)
}

IMG_SIZE = (224, 224)


def fine_tune_model(keras_model_path, feedbacks):

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

        for layer in model.layers:
            layer.trainable = False

        model.layers[-1].trainable = True

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-5
        )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        trained = 0

        for fb in feedbacks:

            if fb.correct_class not in CLASS_MAP:
                print(
                    "Skipping unknown class:",
                    fb.correct_class
                )
                continue

            print(
                "Processing feedback:",
                fb.id
            )

            img = Image.open(
                io.BytesIO(fb.image)
            ).convert('RGB')

            img = img.resize(IMG_SIZE)

            x = np.asarray(
                img,
                dtype=np.float32
            ) / 255.0

            x = np.expand_dims(
                x,
                axis=0
            )

            y = tf.constant(
                [CLASS_MAP[fb.correct_class]],
                dtype=tf.int32
            )

            x = tf.convert_to_tensor(
                x,
                dtype=tf.float32
            )

            print("Training on one image...")

            with tf.GradientTape() as tape:

                predictions = model(
                    x,
                    training=True
                )

                loss = loss_fn(
                    y,
                    predictions
                )

            gradients = tape.gradient(
                loss,
                model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables
                )
            )

            print(
                "Feedback:",
                fb.id,
                "Loss:",
                float(loss.numpy())
            )

            trained += 1

            del x
            del y
            del img
            del predictions
            del loss
            del gradients

            gc.collect()

        if trained == 0:
            return (
                False,
                None,
                "No usable feedback found"
            )

        print("Training finished")

        new_keras_path = "new_model.keras"

        print("Saving Keras model...")

        model.save(
            new_keras_path
        )

        print("Converting to TFLite...")

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

        print("TFLite model saved")

        del tflite_model
        del converter

        gc.collect()

        return (
            True,
            (
                new_keras_path,
                new_tflite_path
            ),
            None
        )

    except Exception as e:

        print(
            "Fine-tuning error:",
            repr(e)
        )

        return (
            False,
            None,
            str(e)
        )

    finally:

        if model is not None:
            del model

        tf.keras.backend.clear_session()

        gc.collect()
