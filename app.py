from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)

# Charger le modèle entraîné
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_b64 = data.get("image")
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Adapter à la taille d'entrée de ton modèle
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        prediction = model.predict(np.expand_dims(img, axis=0))

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
