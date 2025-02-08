from ..choiceParameters import ChoiceParameters
from .transformer import TransformerModel, train, evaluate
import numpy as np
from sklearn.model_selection import train_test_split

class TransformerParameters(ChoiceParameters):
    def __init__(self, num_output, dmodel=128, nhead=8, num_layers=2, remember_previous_internal_state=True):
        super().__init__()
        self.model_size = 128
        self.model = TransformerModel(num_output, d_model=dmodel, nhead=nhead, num_layers=num_layers)
        self.remember_previous_internal_state = remember_previous_internal_state

    def get_state_after_embedding(self, embedding):
        pass

    def get_state_for_list(self, list_embedding):
        pass

    def __call__(self, embedding):
        return evaluate(self.model, [self.get_state_after_embedding(embedding).flatten()])[0]

    def prepare_data(self, X):
        newX = []

        for sequence in X:
            newX.append(self.get_state_for_list(sequence).flatten())

        return newX
    
    def train(self, X, Y, epochs = 10):
        X = self.prepare_data(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        train(self.model, X_train, Y_train, epochs)

        return evaluate(self.model, X_test, Y_test)

class TransformerParametersSequential(TransformerParameters):
    def __init__(self, num_output, dmodel=128, nhead=8, num_layers=2, remember_previous_internal_state=True):
        super().__init__(num_output, dmodel, nhead, num_layers, remember_previous_internal_state)
        self.state = None
        self.separator = 110901

    def reset_state(self):
        self.state = None

    def get_state_after_embedding(self, embedding):
        if self.state is None or not self.remember_previous_internal_state:
            self.state = embedding.flatten()
        else:
            self.state = np.concatenate((self.state, np.array([self.separator]), embedding.flatten()))

        return self.state

    def get_state_for_list(self, list_embedding):
        joined = None

        for embedding in list_embedding:
            if joined is None:
                joined = embedding.flatten()
            else:
                joined = np.concatenate((joined, np.array([self.separator]), embedding.flatten()))

        return joined
            

class TransformerParametersSVD(TransformerParameters):
    def __init__(self, num_output, n_componentes=None, dmodel=128, nhead=8, num_layers=2, remember_previous_internal_state=True):
        super().__init__(num_output, dmodel, nhead, num_layers, remember_previous_internal_state)
        self.state = []
        self.n_components = n_componentes

    def reset_state(self):
        self.state = []

    def get_state_for_list(self, list_embedding):
        return self.reduced_embeddings([e.flatten() for e in list_embedding])
        
    def get_state_after_embedding(self, embedding):
        if self.remember_previous_internal_state:
            self.state.append(embedding.flatten())
        else:
            self.state = [embedding.flatten()]
        return self.reduced_embeddings(self.state)


    def reduced_embeddings(self, flatten_embeddings):
        min_len = min(arr.size for arr in flatten_embeddings)

        trunk_rows = [arr[:min_len] for arr in flatten_embeddings]
        M = np.stack(trunk_rows, axis=0)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        n_components = self.n_components
        if n_components is None or n_components > S.size:
            n_components = S.size

        top_k_vectors = Vt[:n_components, :]
        
        result_vector = top_k_vectors.flatten()
        return result_vector