import os
import pandas as pd
import numpy as np
from simplenn import NeuralNetwork

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")

data = pd.read_csv(DATA_PATH).values

X = data[:, 1:].T / 255.0
Y = data[:, 0]

model = NeuralNetwork(lr=0.1, epochs=600)
model.fit(X, Y)

print("Final accuracy:", model.evaluate(X, Y))
