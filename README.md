Celebrity Face Generation using GANs
This project implements a Generative Adversarial Network (GAN) using TensorFlow/Keras to generate grayscale celebrity face images at 28x28 resolution. The model is trained on a custom dataset of celebrity faces and includes code to define, train, and evaluate both the generator and discriminator models.

📌 Project Highlights
TensorFlow-based implementation of a DCGAN-style model

Custom-trained on a grayscale celebrity face dataset

Generator and Discriminator implemented from scratch

Periodic image generation for visualizing progress

📁 Project Structure
bash
Copy
Edit
.
├── gan_face_generator.py   # Main training script
├── gan_generated_image_epoch_X.png   # Generated outputs (saved periodically)
├── /content/data/          # Directory containing celebrity face images
└── README.md
🧠 Model Architecture
Generator
Input: 100-dimensional noise vector

Fully connected layers with batch normalization and LeakyReLU activation

Output: 28x28x1 grayscale image (with tanh activation)

Discriminator
Input: 28x28x1 grayscale image

Fully connected layers with LeakyReLU activation

Output: Binary classification (real or fake)

🗂 Dataset
Input: Grayscale celebrity face images resized to 28x28 pixels.

Normalization: All images are normalized to the range [-1, 1].

Directory: Place images inside /content/data/.

🚀 How to Run
Install Dependencies

bash
Copy
Edit
pip install tensorflow matplotlib opencv-python
Prepare Dataset

Place grayscale face images in: /content/data/

Ensure images are readable and correctly sized/resized (handled in code).

Run the Script

bash
Copy
Edit
python gan_face_generator.py
Monitor Outputs

Every 1000 epochs, a generated image grid will be saved as gan_generated_image_epoch_X.png.

⚙️ Hyperparameters
latent_dim = 100

epochs = 10000

batch_size = 64

save_interval = 1000

🧪 Example Outputs
After training, the model generates synthetic celebrity-like faces such as:


⚠️ Limitations
Image resolution is limited to 28x28 for training speed and simplicity.

Faces are in grayscale.

Basic GAN, no advanced stabilization techniques used (e.g., WGAN, spectral normalization).

🛠 Future Improvements
Train on higher resolution images (e.g., 64x64, 128x128).

Add color support (RGB).

Implement advanced GAN variants for better quality and training stability.

Use augmentation to improve dataset variability.
