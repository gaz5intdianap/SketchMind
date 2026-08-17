import tensorflow as tf
import numpy as np
from PIL import Image
import os


class Predictor:

    def __init__(self):

        self.model = None
        self.labels = []

        self.load_labels()
        self.load_model()

    # -------------------------
    # Load Labels
    # -------------------------

    def load_labels(self):

        if os.path.exists("labels.txt"):

            with open("labels.txt", "r") as f:

                self.labels = [
                    line.strip().lower()
                    for line in f.readlines()
                    ]

        else:

            raise FileNotFoundError(
                "labels.txt not found."
            )

    # -------------------------
    # Load Model
    # -------------------------

    def load_model(self):

        model_path = "sketchmind_model.keras"

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                "sketchmind_model.keras not found."
            )

        self.model = tf.keras.models.load_model(
            model_path
        )

    # -------------------------
    # Image Preprocessing
    # -------------------------

    def preprocess(self, image):

    # Convert canvas to grayscale
        image = image.convert("L")
        img = np.array(image)

    # Find pixels belonging to the drawing
        coords = np.argwhere(img > 30)

    # Empty canvas
        if coords.size == 0:
            canvas = np.zeros((28, 28), dtype=np.float32)

        else:
            # Find bounding box of the drawing
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            cropped = img[y0:y1, x0:x1]

            h, w = cropped.shape

        # Make the drawing square
            size = max(h, w)

            square = np.zeros(
                (size, size),
                dtype=np.uint8
            )

        # Center drawing
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2

            square[
                y_offset:y_offset + h,
                x_offset:x_offset + w
            ] = cropped

        # Resize to model input
            resized = Image.fromarray(square).resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

            canvas = np.array(resized).astype(
                np.float32
            ) / 255.0

    # Add channel dimension
        canvas = np.expand_dims(canvas, axis=-1)

    # Add batch dimension
        canvas = np.expand_dims(canvas, axis=0)

        return canvas

    # -------------------------
    # Predict
    # -------------------------

    def predict(self, image):

        img = self.preprocess(image)

        prediction = self.model.predict(
            img,
            verbose=0
        )[0]

        index = np.argmax(prediction)

        confidence = float(prediction[index])

        label = self.labels[index].lower()

        return label, confidence