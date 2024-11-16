from choiceParameters import ChoiceParameters
from transformer import TransformerModel, train, evaluate
import numpy as np
from sklearn.model_selection import train_test_split

class TransformerParametersSequential(ChoiceParameters):
    def __init__(self):
        self.model_size = 128
        self.model = TransformerModel(d_model=128, nhead=8, num_layers=2)
        self.state = None
        self.separator = 110901

    def __call__(self, embedding):
        if self.state is None:
            self.state = embedding.flatten()
        else:
            np.concatenate((self.state, np.array([self.separator]), embedding.flatten()))

        return evaluate(self.model, [self.state])[0]
    
    def prepare_data(self, X):
        newX = []

        for sequence in X:
            joined = None

            for embedding in sequence:
                if joined is None:
                    joined = embedding.flatten()
                else:
                    np.concatenate((joined, np.array([self.separator]), embedding.flatten()))

            newX.append(joined)

        return newX

    def train(self, X, Y):
        X = self.prepare_data(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        train(self.model, X_train, Y_train)

        return evaluate(self.model, X_test, Y_test)
        