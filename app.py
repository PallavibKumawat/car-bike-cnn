from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model/car_bike_cnn.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224,224))
        img = image.img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]
        result = "Bike 🚲" if pred < 0.5 else "Car 🚗"
        image_path = filepath

    return render_template("index.html", result=result, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
