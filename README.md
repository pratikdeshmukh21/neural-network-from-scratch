🧠 Neural Network From Scratch (MNIST)

A fully implemented neural network from scratch using NumPy only, trained on the MNIST handwritten digits dataset.
This project focuses on understanding core neural network concepts such as forward propagation, backpropagation, softmax, and gradient descent — without using high-level ML frameworks like TensorFlow or PyTorch.

📌 This project was first implemented as a Kaggle notebook and later converted into a reusable Python library.

🚀 Project Highlights

✅ Implemented from scratch using NumPy

❌ No TensorFlow / PyTorch / Keras

📊 Trained on MNIST (28×28 handwritten digits)

🧮 Manual implementation of:

Forward propagation

Backpropagation

ReLU activation

Softmax + Cross-Entropy loss

Gradient Descent

📦 Converted into a Python library

🎯 Achieved ~89–90% accuracy with a simple 2-layer network

🧠 Model Architecture
Input Layer   : 784 neurons (28 × 28 pixels)
Hidden Layer  : 10 neurons (ReLU)
Output Layer  : 10 neurons (Softmax)


This intentionally simple architecture was chosen to prioritize clarity and correctness over maximum accuracy.

📈 Results
Metric	Value
Dataset	MNIST
Training Type	Full-batch Gradient Descent
Epochs	~600
Accuracy	~89–90%

⚠️ Higher accuracy on MNIST typically requires deeper networks or convolutional layers.
This project focuses on core understanding, not optimization tricks.

📂 Project Structure
simple_nn/
│
├── simplenn/
│   ├── __init__.py
│   └── model.py        # Neural Network
│
├── examples/
│   └── train_mnist.py  # Example training script
│
├── README.md
├── setup.py
└── requirements.txt

⚙️ Installation

Clone the repository and install dependencies:

pip install -r requirements.txt


Or install locally as a package:

pip install .

▶️ Usage Example
from simplenn import NeuralNetwork
import numpy as np
import pandas as pd

# Load MNIST CSV
data = pd.read_csv("train.csv").values
X = data[:, 1:].T / 255.0
Y = data[:, 0]

# Train model
model = NeuralNetwork(lr=0.1, epochs=600)
model.fit(X, Y)

# Evaluate
accuracy = model.evaluate(X, Y)
print("Accuracy:", accuracy)

🧪 Key Concepts Implemented

Dense layers (matrix multiplication)

ReLU activation and derivative

Softmax normalization (numerically stable)

One-hot encoding

Cross-entropy loss

Backpropagation via chain rule

Gradient descent optimization

🎯 Why Build This From Scratch?

Most ML projects rely on high-level libraries that hide internal workings.
This project was built to:

Gain deep understanding of neural networks

Learn how gradients flow through layers

Debug real numerical issues (softmax, saturation, plateaus)

Understand why models fail or plateau

Build a strong ML foundation before using frameworks

🔮 Future Improvements

Add another hidden layer

Implement mini-batch gradient descent

Add model save / load functionality

Add CNN version (separate project)

Publish on PyPI

🧑‍💻 Author

Pratik Deshmukh
Computer Science Student | Machine Learning Enthusiast

Kaggle Notebook: (https://www.kaggle.com/code/pratikdeshmukh212121/neural-network-from-scratch)

GitHub: (your GitHub profile)

📌 Project Status

✅ Completed
📦 Converted to library
📚 Well-documented
🚀 Ready to showcase

🔥 This project demonstrates understanding, not shortcuts.🧠 Simple Neural Network from Scratch (MNIST)