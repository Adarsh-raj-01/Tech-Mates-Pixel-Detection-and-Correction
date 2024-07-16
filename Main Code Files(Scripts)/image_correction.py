# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:35:19 2024

@author: adars
"""

import torch
from PIL import Image, ImageFilter
import os
from RealESRGAN import RealESRGAN
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
def load_model(scale=4):
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    return model

# Function to apply post-processing
def post_process(image):
    return image.filter(ImageFilter.SHARPEN)

# Function to enhance and save the image
def enhance_image(input_image_path, output_directory, scale=4):
    model = load_model(scale)
    image = Image.open(input_image_path).convert('RGB')
    sr_image = model.predict(image)
    
    # Post-process the image
    sr_image = post_process(sr_image)
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Extract the filename and extension from the input path
    filename = os.path.basename(input_image_path)
    name, ext = os.path.splitext(filename)
    
    # Create the output path
    output_image_path = os.path.join(output_directory, f"{name}_enhanced_x{scale}{ext}")
    
    # Save the enhanced image
    sr_image.save(output_image_path)
    print(f'Enhanced image saved as {output_image_path}')
    
    return image, sr_image

# Function to prompt user to select an input image file
def select_input_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title='Select an Image', filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

# Function to plot original and enhanced images
def plot_images(original, enhanced_images, scales):
    plt.figure(figsize=(18, 6))
    
    # Plot original image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')
    
    # Plot enhanced images for each scale
    for i, (enhanced, scale) in enumerate(zip(enhanced_images, scales)):
        plt.subplot(1, 4, i + 2)
        plt.title(f"Enhanced Image (x{scale})")
        plt.imshow(enhanced)
        plt.axis('off')
    
    plt.show()

# Main function
def main():
    # Prompt the user to select an input image file
    input_image_path = select_input_image()
    
    if not input_image_path:
        print("No image selected. Exiting.")
        return
    
    output_directory = 'D:/Project 9 Detect Pixelated Image and Correct it/Main/Correction/Real-ESRGAN-main/Real-ESRGAN-main/results'  # Update this path to your output directory

    scales = [2, 4, 8]
    enhanced_images = []
    
    # Enhance image with different scales and store results
    for scale in scales:
        _, enhanced = enhance_image(input_image_path, output_directory, scale)
        enhanced_images.append(enhanced)
    
    # Load original image
    original_image = Image.open(input_image_path).convert('RGB')
    
    # Plot original and enhanced images
    plot_images(original_image, enhanced_images, scales)

# Run the main function
if __name__ == "__main__":
    main()
