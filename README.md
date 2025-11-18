# 🦴 FractureNet+ — Fracture Detection & Fragility Assessment

FractureNet+ is a deep-learning based medical imaging system that detects bone fracturesfrom X-ray images and, in the absence of a fracture, assesses bone fragility (osteoporosis risk).
The project uses a two-stage prediction approach to improve clinical relevance, assisting doctors, radiologists, and healthcare systems.

---

## 🚀 Features

✔ Detects fractures from X-ray images  
✔ Performs fragility (osteoporosis) assessment if no fracture is found  
✔ Flask-based web interface for image upload and prediction  
✔ Supports multiple image types: PNG, JPG, JPEG  
✔ Trained using CNN/Transfer Learning model  
✔ Deployed using GitHub & ready for API/server deployment

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras / CNN |
| Backend Framework | Flask |
| Image Processing | OpenCV / Pillow |
| Frontend | HTML, CSS, Bootstrap |
| Model Format | `.h5` |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

FractureNet+/
│-- static/
│-- templates/
│-- models/
│ ├── fracture_model.h5
│ └── fragility_model.h5
│-- app.py
│-- requirements.txt
│-- README.md
