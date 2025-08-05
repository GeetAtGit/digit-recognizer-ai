import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Digit Draw AI", layout="centered")

st.title("🖌️ Draw a Digit (0–9) and Let AI Guess!")

st.sidebar.markdown("## How to Use")
st.sidebar.write("""
- Use the white canvas below to draw a single digit.
- Make sure it's centered and takes up most of the canvas.
- Click **Predict** to see what AI thinks!
""")

# Load the model
model = load_model("digit_model.keras")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# When the user draws something
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Convert to grayscale, resize to 28x28
    img = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Show the preview
    st.image(img, width=140, caption="Processed Input")

    if st.button("🔮 Predict"):
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.success(f"✅ I think it's a **{predicted_digit}**")

        # Display prediction confidence as bar chart
        st.bar_chart(prediction[0])
