# 🧠 Digit Recognizer AI

Draw a digit on the canvas, and this app will predict what you wrote using a Convolutional Neural Network trained on the MNIST dataset.

## 🎯 Features

- Interactive canvas to draw digits (0–9)
- Trained using TensorFlow on MNIST dataset
- Real-time predictions with confidence chart
- Streamlit-based interface

## 🛠️ How to Run Locally

1. Clone the repo or download the code
2. (Optional) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python digitrecognizer.py

streamlit run app.py
