import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

IMG_SIZE = (128, 128)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model

model = build_model()
model.load_weights("scan_quality_weights.weights.h5")

def predict(image):
    image = image.convert("L")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = image.reshape(1, 128, 128, 1)

    prob_unclear = float(model.predict(image)[0][0])

    if prob_unclear > 0.65:
        result = "Unclear scan"
        confidence = "High"
        guidance = "Retake the scan. Adjust probe angle or gain."
    elif prob_unclear < 0.35:
        result = "Clear scan"
        confidence = "High"
        guidance = "Scan quality acceptable. Proceed with interpretation."
    else:
        result = "Uncertain scan quality"
        confidence = "Moderate"
        guidance = "Consider retaking the scan for confirmation."

    return {
        "Result": result,
        "Confidence": confidence,
        "Recommendation": guidance,
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Ultrasound Scan Quality MVP",
    description="Flags unclear ultrasound scans at capture time."
)


demo.launch()
