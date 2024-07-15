import os
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Image Data Generators for loading and preprocessing images
def create_image_generators(train_dir, val_dir, test_dir, target_size=(256, 256), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator, test_generator

# Build the detection model
def build_detection_model(input_shape=(256, 256, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to predict if a given image is pixelated
def predict_pixelation(image_path, model, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Measure inference time
    start_time = time.time()
    prediction = model.predict(img_array)
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1 / inference_time

    if prediction < 0.5:
        result = "The given image is not pixelated."
    else:
        result = "The given image is pixelated."
    result += f"\nPrediction score: {prediction[0][0]:.4f}"
    result += f"\nInference Time: {inference_time:.4f} seconds"
    result += f"\nFrames Per Second (FPS): {fps:.2f}"
    return result

# Tkinter GUI for selecting directories and loading images
def select_directory(title):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    directory_path = filedialog.askdirectory(title=title)
    if not directory_path:
        messagebox.showwarning("Warning", f"{title} not selected. Exiting.")
        exit()
    return directory_path

# Tkinter GUI to load and display images
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_pixelation(file_path, detection_model)
        display_image(file_path, result)

def display_image(file_path, result):
    img = Image.open(file_path)
    img.thumbnail((256, 256))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img  # Keep a reference to avoid garbage collection
    result_label.config(text=result)

# Prompt the user to select the dataset directories
train_dir = select_directory("Select Training Directory")
val_dir = select_directory("Select Validation Directory")
test_dir = select_directory("Select Testing Directory")

# Create the image generators
train_generator, val_generator, test_generator = create_image_generators(train_dir, val_dir, test_dir)

# Train the model if it does not exist, otherwise load the existing model
model_path = 'pixelation_detection_model.h5'
if not os.path.exists(model_path):
    detection_model = build_detection_model()
    detection_model.summary()

    # Train the model
    detection_model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Evaluate the model
    loss, accuracy = detection_model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    detection_model.save(model_path)
else:
    detection_model = load_model(model_path)
    print("Model loaded from disk.")

# Create the Tkinter GUI for image selection and display
root = tk.Tk()
root.title("Pixelation Detection")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

panel = tk.Label(frame)
panel.pack()

btn = tk.Button(frame, text="Load Image", command=load_image)
btn.pack(pady=10)

result_label = tk.Label(frame, text="")
result_label.pack()

root.mainloop()
