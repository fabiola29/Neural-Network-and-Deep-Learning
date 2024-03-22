import os
import numpy as np
from loader import MNISTLoader
from functions import NeuralNetwork

# Caricamento il dataset MNIST
data_path = os.getcwd() + "/Mnist"  #cambiare a seconda del percorso in cui si trova la cartella Mnist
print(data_path)
mnist_loader = MNISTLoader(data_path)

# Caricamento il dataset
X, y_onehot = mnist_loader.data.T, mnist_loader.labels.T

# Normalizzazione delle immagini
X = X / 255.0

# Impostazione le dimensioni del training set e del test set  ( va distribuito bene e deve essere minore di 60000 la somma train + validation)
train_size = 40000
test_size = 8000
validation_size = 12000

# Dividisione del dataset in training e test set
X_train, X_valid, X_test = X[:train_size], X[train_size:train_size+validation_size], X[train_size+validation_size:]
y_train, y_valid, y_test = y_onehot[:train_size], y_onehot[train_size:train_size+validation_size], y_onehot[train_size+validation_size:]


y_train = np.array(y_train, dtype=np.int64)
y_valid = np.array(y_valid, dtype=np.int64)
y_test = np.array(y_test, dtype=np.int64)


# Lista dei numeri di nodi interni da testare
hidden_sizes = [128]  # per doc far variare #64 128 256 512 1024

# Impostazione degli iperparametri
epochs = 100   # per doc far variare

# Creazione di un'istanza della classe NeuralNetwork
neural_net = NeuralNetwork(neurons_per_layer=[X_train.shape[1]] + hidden_sizes + [y_train.shape[1]])

# Impostazione delle funzioni di attivazione e di errore
activation_functions = ["relu", "identity"]
neural_net.set_activation_functions(activation_functions)
neural_net.set_error_function("cross_entropy", softmax_post_processing=True)

# Addestramento della rete con RProp
best_parameters = neural_net.train(X_train.T, np.argmax(y_train, axis=1), X_valid.T, np.argmax(y_valid, axis=1), type_update_parameter="rprop",
                                   epochs = epochs)

# Propagazione in avanti sulla rete per il test set
forward_test = neural_net.forward_propagation(X_test.T, best_parameters)

print("Dimensione del set di test:", y_test.shape[0])
print("Dimensione del validation set:", y_valid.shape[0])
print("Dimensione del trainging set:", y_train.shape[0])


# Calcolo dell'accuratezza sul test set
y_one_hot = neural_net.one_hot(np.argmax(y_train, axis=1))
test_accuracy = neural_net.accuracy(X_test.T, neural_net.one_hot(np.argmax(y_test, axis=1)), best_parameters)
print("Test Accuracy: {:.2f}%".format(test_accuracy))


# Calcolo della loss sul test set
test_loss = neural_net.cross_entropy(forward_test["output"], np.argmax(y_test, axis=1))
print("Loss on Validation Dataset:", test_loss)
