from mnist.loader import MNIST
import numpy as np



class MNISTLoader:

    def __init__(self, data_path):
        # Inizializza l'istanza del caricatore MNIST con il percorso dei dati

        self._mnist_data = MNIST(data_path)
        # Carica i dati MNIST
        self._load_data()

    def _load_data(self):
        # Carica dati e etichette dal set di addestramento MNIST
        data, labels = self._mnist_data.load_training()
        # Trasponi i dati per avere un formato più comodo (feature come colonne)
        data = np.array(data).T
        # Codifica le etichette in formato one-hot
        labels = self.encode(np.array(labels))
        # Assegna i dati e le etichette all'istanza della classe
        self.data = data
        self.labels = labels

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        # Codifica le etichette in formato one-hot
        encoded_labels = np.zeros(shape=(10, labels.shape[0]))
        for n in range(labels.shape[0]):
            encoded_labels[labels[n]][n] = 1
        return encoded_labels

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        # Decodifica le etichette one-hot restituendo gli indici degli elementi massimi
        return np.array([
            np.argmax(labels[:, n]) for n in range(labels.shape[1])
        ])
