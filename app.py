from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Models
fracture_model = load_model("Models/best_fracture_model_integrated.keras")
fragility_model = load_model("Models/fragility_best_densenet.h5")


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size, color_mode="rgb")
    img = image.img_to_array(img)

    # Normalize
    img = img / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    return img


# ---------- INTERPRET FRACTURE ----------
def interpret_fracture(pred):
    # SIGMOID output => shape (1,)
    if isinstance(pred, np.ndarray) and pred.shape == (1,):
        prob = float(pred[0])

    # SOFTMAX output => shape (2,) assume index 1 = Fracture
    elif isinstance(pred, np.ndarray) and len(pred) == 2:
        prob = float(pred[1])

    else:
        prob = float(pred)

    # LOWER THRESHOLD (increases accuracy for borderline cases)
    if prob >= 0.35:
        return "Yes", prob
    else:
        return "No", prob


# ---------- INTERPRET FRAGILITY ----------
def interpret_fragility(pred_array):
    classes = ["Normal Bone", "Osteopenia", "Osteoporosis"]
    idx = np.argmax(pred_array)
    return classes[idx], float(pred_array[idx])


@app.route("/", methods=["GET", "POST"])
def index():
    fracture_result = None
    fragility_result = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        image_path = file_path

        # Preprocess image
        img_array = preprocess_image(file_path)

        # -------- Fracture Prediction --------
        raw_fracture = fracture_model.predict(img_array)[0]
        print("\n🔍 Raw Fracture Prediction:", raw_fracture)

        fracture_result, frac_conf = interpret_fracture(raw_fracture)

        # -------- If Fracture = Yes -> Skip Fragility --------
        if fracture_result == "Yes":
            fragility_result = "Not Applicable (Fracture Detected)"
            frag_conf = None

        else:
            # -------- Fragility Prediction --------
            raw_fragility = fragility_model.predict(img_array)[0]
            print("🦴 Raw Fragility Prediction:", raw_fragility)

            fragility_result, frag_conf = interpret_fragility(raw_fragility)

        return render_template("index.html",
                               file=filename,
                               fracture_label=fracture_result,
                               fracture_conf=round(frac_conf * 100, 2),
                               fragility_label=fragility_result,
                               fragility_conf=round(frag_conf * 100, 2) if fracture_result == "No" else None,
                               img=image_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
