import numpy as np

class NeuralNetwork:
    def __init__(self, lr=0.1, epochs=600):
        self.lr = lr
        self.epochs = epochs
        self._init_parameters()

    # ===== YOUR ORIGINAL FUNCTIONS =====

    def _init_parameters(self):
        self.w1 = np.random.randn(10, 784) * 0.01
        self.b1 = np.zeros((10, 1))
        self.w2 = np.random.randn(10, 10) * 0.01
        self.b2 = np.zeros((10, 1))

    def ReLU(self, z):
        return np.maximum(0, z)

    def dReLU(self, z):
        return (z > 0).astype(float)

    def SoftMax(self, z):
        exp = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def one_hot(self, Y):
        one_hot_Y = np.zeros((10, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    def forward_prop(self, X):
        z1 = self.w1.dot(X) + self.b1
        a1 = self.ReLU(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.SoftMax(z2)
        return z1, a1, z2, a2

    def back_prop(self, z1, a1, a2, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y)

        dz2 = a2 - one_hot_Y
        dw2 = (1/m) * dz2.dot(a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = self.w2.T.dot(dz2) * self.dReLU(z1)
        dw1 = (1/m) * dz1.dot(X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        return dw1, db1, dw2, db2

    def update_parameters(self, dw1, db1, dw2, db2):
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    def get_predictions(self, a2):
        return np.argmax(a2, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    # ===== TRAIN / PREDICT API =====

    def fit(self, X, Y):
        for i in range(self.epochs):
            z1, a1, z2, a2 = self.forward_prop(X)
            dw1, db1, dw2, db2 = self.back_prop(z1, a1, a2, X, Y)
            self.update_parameters(dw1, db1, dw2, db2)

            if i % 10 == 0:
                preds = self.get_predictions(a2)
                acc = self.get_accuracy(preds, Y)
                print(f"Epoch {i} | Accuracy: {acc:.4f}")

    def predict(self, X):
        _, _, _, a2 = self.forward_prop(X)
        return self.get_predictions(a2)

    def evaluate(self, X, Y):
        preds = self.predict(X)
        return self.get_accuracy(preds, Y)
