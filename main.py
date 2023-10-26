import numpy as np
import tkinter as tk
from tkinter import Canvas
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from PIL import Image, ImageDraw

# Load and preprocess data
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images / 255.0
train_labels = to_categorical(train_labels)

# Build and train a simple model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

def predict_digit(img):
    # Convert image to array and preprocess
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28)
    img = 255 - img  # Invert colors
    img = img / 255.0

    # Predict the digit
    pred = model.predict([img])
    return np.argmax(pred)

def on_submit():
    global canvas, draw
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    img.paste(Image.fromarray(np.array(canvas_image)))
    digit = predict_digit(img)
    label.config(text=f"Predicted Digit: {digit}")
    canvas.delete("all")
    canvas_image = Image.new("RGB", (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(canvas_image)

def paint(event):
    x, y = event.x, event.y
    canvas.create_oval((x - 10, y - 10, x + 10, y + 10), fill='black', width=20)
    draw.line([x, y, x, y], fill='black', width=20)

root = tk.Tk()
root.title("Digit Recognizer")

canvas = Canvas(root, bg='white', width=500, height=500)
canvas.pack(pady=20)

canvas.bind("<B1-Motion>", paint)

canvas_image = Image.new("RGB", (500, 500), (255, 255, 255))
draw = ImageDraw.Draw(canvas_image)

submit_button = tk.Button(root, text="Predict", command=on_submit)
submit_button.pack(pady=20)

label = tk.Label(root, text="Draw a digit and click on Predict!")
label.pack(pady=20)

root.mainloop()
