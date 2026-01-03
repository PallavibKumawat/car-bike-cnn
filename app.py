from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model/car_bike_cnn.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        if prediction >= 0.5:
            result = "Car 🚗"
            confidence = f"{prediction*100:.2f}%"
        else:
            result = "Bike 🚲"
            confidence = f"{(1-prediction)*100:.2f}%"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
