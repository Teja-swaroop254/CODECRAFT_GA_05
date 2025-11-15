# CODECRAFT_GA_05
Neural Style Transfer using PyTorch and VGG-19 to blend the content of one image with the artistic style of another. Includes preprocessing, feature extraction, Gram matrix style loss, LBFGS optimization, and visualization of content, style, and output images.

# 🎨 Neural Style Transfer using PyTorch

This project implements **Neural Style Transfer (NST)** using a pretrained **VGG-19** network in PyTorch.  
It blends the **content** of one image with the **style** of another image to generate a stylized output similar to artistic paintings.

This repository is created as part of an **internship project**, with the aim of demonstrating clean code structure, deep learning understanding, and clear documentation.

---

# 📘 Table of Contents

- [Introduction](#introduction)
- [How Neural Style Transfer Works](#how-neural-style-transfer-works)
- [Architecture Diagram](#architecture-diagram)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Explanation](#code-explanation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

# 📖 Introduction

Neural Style Transfer (NST), introduced by **Gatys et al. (2015)**, recreates content images in the artistic style of a reference style image using deep neural networks.

NST extracts:

- **Content** → shapes, structure  
- **Style** → textures, colors, brush strokes  

This implementation uses **VGG-19**, a powerful pretrained CNN model, and the **LBFGS optimizer**, which produces high-quality stylized results.

---

# 🧠 How Neural Style Transfer Works

NST works through the following three steps:

### 1️⃣ Content Representation  
Extract from a deeper VGG layer (e.g., `conv_4`) because deeper layers encode object shapes.

### 2️⃣ Style Representation  
Extract multiple layers (`conv_1` to `conv_5`) and compute **Gram matrices**, which capture textures and patterns.

### 3️⃣ Pixel Optimization  
Instead of training a model, NST directly optimizes the **pixels of the target image**:

---

# Architecture Diagram

###### <img width="1024" height="1536" alt="ChatGPT Image Nov 15, 2025, 11_30_55 PM" src="https://github.com/user-attachments/assets/e02dc1fa-d586-4b77-9dd6-9f70891a28a0" />

---

## 📌 Features
- Neural Style Transfer implemented using PyTorch and VGG-19  
- Combines content and style images to generate stylized output  
- LBFGS optimizer for high-quality results  
- Extracts multi-layer content and style features  
- Gram matrix used to capture texture and style information  
- Displays content, style, and output images side by side  
- GPU acceleration supported (CUDA auto-detected)  
- Clean, well-structured, and internship-ready code  

---

## 🔧 Installation
Install required packages:

```bash
pip install torch torchvision pillow matplotlib

---

## 🚀 Usage

**Update the paths in the code:**
```python
content_img = load_image("/path/to/content.png")
style_img   = load_image("/path/to/style.png")

**Run the script:**
```bash
python style_transfer.py

**Output will be saved as:**
output_LBFGS_like_first.png

---

## 📂 Project Structure

CODECRAFT_GA_05/
│── style_transfer.py
│── content_image.png
│── style_image.png
│── output_LBFGS_like_first.png
│── README.md

---

## 🧩 Code Explanation

**1. Image Preprocessing**  
- Loads content and style images  
- Resizes them  
- Converts to tensors  
- Normalizes using ImageNet mean and standard deviation  

**2. VGG-19 Model Setup**  
- Loads pretrained VGG-19 in evaluation mode  
- Replaces in-place ReLU layers to avoid gradient issues  
- Selects specific layers for content and style extraction  

**3. Feature Extraction**  
- Content features are taken from `conv_4`  
- Style features are taken from layers `conv_1` to `conv_5`  
- Gram matrices are computed for style representation  

**4. Loss Functions**  
- **Content Loss:** Difference between content and target features  
- **Style Loss:** Difference between Gram matrices of style and target features  
- Weighted combination creates the final loss  

**5. Optimization (LBFGS)**  
- The target image (initialized as the content image) is optimized  
- LBFGS minimizes the total loss for high-quality stylization  

**6. Output Generation**  
- The target tensor is denormalized  
- Converted back to an image  
- Saved and displayed along with content and style images  



