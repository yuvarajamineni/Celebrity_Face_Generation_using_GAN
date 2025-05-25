# 🎭 Celebrity Face Generation using GANs

This project implements a **Generative Adversarial Network (GAN)** using TensorFlow/Keras to generate grayscale celebrity face images at 28x28 resolution. The GAN is trained on a custom dataset of celebrity faces, and the code includes full model definitions, training loops, and periodic output generation.

---

## 📌 Features

- ✅ TensorFlow 2.x implementation
- ✅ Custom Generator and Discriminator models
- ✅ Trained on grayscale celebrity faces
- ✅ Periodic image generation and saving
- ✅ Simple and extensible codebase

---

## 🧠 Model Architecture

### 🎨 Generator
- Input: 100-dimensional noise vector
- Layers:
  - Dense → LeakyReLU → BatchNorm
  - Dense → LeakyReLU → BatchNorm
  - Dense → LeakyReLU → BatchNorm
  - Dense (784 units, `tanh`) → Reshape to (28, 28, 1)

### 🕵️ Discriminator
- Input: (28, 28, 1) grayscale image
- Layers:
  - Flatten → Dense → LeakyReLU
  - Dense → LeakyReLU
  - Dense (1 unit, `sigmoid`)

---

## 🗂 Dataset

- Directory: `./content/data/`
- Requirements:
  - Grayscale images
  - Size: 28x28 pixels (auto-resized in code)
- Normalization: Pixel values scaled to `[-1, 1]`

---

## 🚀 Getting Started

### ✅ Requirements

```bash
pip install tensorflow numpy matplotlib opencv-python
