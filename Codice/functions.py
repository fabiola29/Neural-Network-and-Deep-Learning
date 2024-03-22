import numpy as np
import sys
import matplotlib.pyplot as pl
from matplotlib import pyplot

np.set_printoptions(threshold=sys.maxsize)


class NeuralNetwork:

    def __init__(self, neurons_per_layer):
        # Inizializza il numero di neuroni in ciascuno strato
        self.layers = self.inizialize_numebr_of_neurons_per_layers(np.array(neurons_per_layer))
        # Dizionari per le funzioni di attivazione e di errore
        self.dict_activation_functions = self.set_dict_activation_functions()
        self.dict_error_functions = self.set_dict_error_functions()

    def inizialize_numebr_of_neurons_per_layers(self, sizes):
        # Inizializza il numero di neuroni in ciascun strato
        layers = {}
        for i in range(sizes.shape[0]):
            layers["layer" + str(i)] = sizes[i]
        return layers

    def get_number_of_layers(self):
        # Restituisce il numero di strati nella rete neurale
        return len(self.layers) - 1

    def set_dict_activation_functions(self):
        # Definisce un dizionario contenente tutte le funzioni di attivazione e le loro derivate
        return {
            "tanh": [self.tanh, self.derivative_tanh],
            "relu": [self.relu, self.derivative_relu],
            "leaky_relu": [self.leaky_relu, self.derivative_leaky_relu],
            "sigmoid": [self.sigmoid, self.derivative_sigmoid],
            "identity": [self.identity, self.derivative_identity]
        }

    def set_dict_error_functions(self):
        # Definisce un dizionario contenente le funzioni di errore e le loro derivate
        return {
            "cross_entropy": [self.cross_entropy, self.derivative_cross_entropy],
            "sum_of_squares": [self.sum_of_squares, self.derivative_sum_of_squares]
        }

    def set_activation_functions(self, activation_function):
        # Imposta le funzioni di attivazione
        self.activation_functions = activation_function

    def set_error_function(self, error_function, softmax_post_processing=True):
        # Imposta la funzione di errore e il post-processing softmax
        self.error_function = error_function
        self.softmax_post = softmax_post_processing

    def eprint(self, *args, **kwargs):
        # Funzione di stampa di errore e uscita dal programma
        print(*args, file=sys.stderr, **kwargs)
        sys.exit()

    def check_iper_parameters(self):
        # Controllo degli iper-parametri
        if len(self.layers) < 3:
            self.eprint("Number of layers have to be at least 3")
        for nodes in self.layers.values():
            if nodes < 1:
                self.eprint("The minimum number of neurons for each layer is 1")
        if not hasattr(self, "activation_functions"):
            self.eprint("You have to pass activation functions")
        if not hasattr(self, "error_function"):
            self.eprint("You have to pass error function")
        if self.error_function not in self.dict_error_functions.keys():
            self.eprint("Error function can be only cross-entropy or sum-of-squares")
        if len(self.activation_functions) != self.get_number_of_layers():
            self.eprint("Number of activation function different from number of layer")
        for af in self.activation_functions:
            if (af not in self.dict_activation_functions.keys()):
                self.eprint("Activation function not available")
        if self.error_function == "cross_entropy":
            if not hasattr(self, "softmax_post"):
                self.eprint("You have to set post-processing true or false")
            if self.softmax_post != True and self.softmax_post != False:
                self.eprint("Softmax post-processing can to be only True or False")
            if self.softmax_post == True and self.layers["layer" + str(self.get_number_of_layers())] == 1:
                self.eprint("When have 1 node on output layer you can't use softmax")
            if self.softmax_post == True and self.activation_functions[self.get_number_of_layers() - 1] != "identity":
                self.eprint(
                    "When softmax post-processing is True the activation function of the last layer have to be identity")
        elif self.error_function == "sum-of-squares":
            if self.neurons_per_layer[self.get_number_of_layers()] != 1:
                self.eprint("When use sum-of-squares you have to be only 1 node as output layer")

    # ------------------------------------ Initialization weights and biases ----------------------------
    def initialize_weights_and_biases(self):
        # Inizializza i pesi e i bias della rete neurale
        parameters = {}

        for i in range(len(self.layers) - 1):
            np.random.seed(15)
            # Inizializza i pesi con una distribuzione casuale
            parameters["W" + str(i + 1)] = (np.random.randn(self.layers.get("layer" + str(i + 1)),
                                                           self.layers.get("layer" + str(i))) *
                                            np.sqrt(1. / self.layers.get("layer" + str(i))))
            # Inizializza i bias a zero
            parameters["b" + str(i + 1)] = np.zeros((self.layers.get("layer" + str(i + 1)), 1))

        return parameters

    # ------------------------------------ Calcolate one-hot codification -------------------------------
    def one_hot(self, Y):
        # Converte le etichette in codifica one-hot
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    # ----------------------------------- Activation function --------------------------------------------
    def tanh(self, x):
        # funzione di attivazione tangente iperbolica
        '''
                    x    -x
                   e  - e
        tanh(x) = ────────
                    x    -x
                  e  + e
        '''
        return np.tanh(x)

    def relu(self, x):
        # Funzione di attivazione ReLU
        # f(x) = max(0, x)
         return np.maximum(x, 0)

    def leaky_relu(self, x):
        # Funzione di attivazione Leaky ReLU
        # f(x) = max(0.01 * x, x).
        return np.where(x > 0, x, 0.01 * x)

    def sigmoid(self, x):
        # Funzione di attivazione sigmoide
        '''
                          1
                f(x) = ───────
                             -x
                        1 + e

         '''
        return np.where(x < 0, np.exp(x) / (1.0 + np.exp(x)), 1.0 / (1.0 + np.exp(-x)))  # avoid NaN

    def identity(self, x):
        # Funzione di attivazione identita'
        # f(x) = x
        return x

    # --------------------------------- Derived activation function ----------------------------------------

    def derivative_tanh(self, x):
        # Derivata della tangente iperbolica
        return (1 - np.power(np.tanh(x), 2))

    def derivative_relu(self, x):
        # Derivata di ReLU
        return np.where(x > 0, 1, 0)

    def derivative_leaky_relu(self, x):
        # Derivata di Leaky ReLU
        return np.where(x > 0, 1, 0.01)

    def derivative_sigmoid(self, x):
        # Derivata della sigmoide
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def derivative_identity(self, x):
        # Derivata dell'identita'
        return 1

    # ------------------------------------ Post-processing----------------------------------------------------
    def softmax(self, x):
        # Funzione softmax per la post-elaborazione
        '''
        Il problema comune che può verificarsi durante l’applicazione di softmax è il problema di stabilità numerica,
        il che significa che ∑j e^(z_j) può diventare molto grande a causa dell’errore esponenziale e di overflow
        che può verificarsi. Questo errore di overflow può essere risolto sottraendo ogni valore dell’array
        con il suo valore massimo.
        '''
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] - np.max(x[:, i])

        expX = np.exp(x)
        return expX / np.sum(expX, axis=0)

    def post_processing(self, z):
        # Applica la funzione softmax per la post-elaborazione
        return self.softmax(z)

    # ------------------------------------- Error functions ---------------------------------------------

    def cross_entropy(self, y_pred, y_true):
        # Calcola la cross entropy come funzione di errore
        y_true_one_hot_vec = (y_true[:, np.newaxis] == np.arange(10))
        loss_sample = (np.log(y_pred.T, where=y_pred.T != 0) * y_true_one_hot_vec).sum(axis=1)
        return -loss_sample.sum(axis=0)

    def sum_of_squares(self, out, t):
        # Calcola la somma dei quadrati come funzione di errore
        return (np.sum((out - t) ** 2)) / 2

    # ------------------------------------- Derivative error functions ---------------------------------------------

    def derivative_cross_entropy(self, out, t):
        # Calcola derivata della cross entropy
        if (self.softmax_post == False):
            return (out - t) / (out * (1 - out))
        else:
            return (out - t)

    def derivative_sum_of_squares(self, out, t):
        # Calcola derivata della somma dei quadrati
        return (out - t)

    # ------------------------------------- Forward proppagation ---------------------------------------------
    def forward_propagation(self, x, parameters):
        # Propagazione in avanti attraverso la rete neurale
        forward_value = {}

        for i in range(self.get_number_of_layers()):
            activation_function = self.dict_activation_functions.get(self.activation_functions[i])[0]
            if i == 0:
                # Calcolo dell'output del primo strato nascosto
                forward_value["a" + str(i + 1)] = np.dot(parameters.get("W" + str(i + 1)), x) + parameters.get(
                    "b" + str(i + 1))
                forward_value["z" + str(i + 1)] = activation_function(forward_value.get("a" + str(i + 1)))
            elif i == self.get_number_of_layers() - 1:
                # Calcolo dell'output del layer di output
                forward_value["a" + str(i + 1)] = np.dot(parameters.get("W" + str(i + 1)),
                                                         forward_value.get("z" + str(i))) + parameters.get(
                    "b" + str(i + 1))
                forward_value["z" + str(i + 1)] = activation_function(forward_value.get("a" + str(i + 1)))
                if self.softmax_post == True:
                    # Applica la post-elaborazione softmax se necessario
                    forward_value["output"] = self.post_processing(forward_value.get("z" + str(i + 1)))
                else:
                    forward_value["output"] = forward_value["z" + str(i + 1)]
            else:
                # Calcolo dell'output degli strati nascosti intermedi
                forward_value["a" + str(i + 1)] = np.dot(parameters.get("W" + str(i + 1)),
                                                         forward_value.get("z" + str(i))) + parameters.get(
                    "b" + str(i + 1))
                forward_value["z" + str(i + 1)] = activation_function(forward_value.get("a" + str(i + 1)))

        return forward_value

    # ------------------------------------- Back proppagation ---------------------------------------------
    def backward_prop(self, x, y, parameters, forward_value):
        # Calcola i gradienti durante la retro-propagazione
        max_layer = self.get_number_of_layers()
        delta = {}
        m = x.shape[1]
        derivative_error_function = self.dict_error_functions.get(self.error_function)[1]

        for i in range(max_layer, 0, -1):
            derivated_activation_function = self.dict_activation_functions.get(self.activation_functions[i - 1])[1]
            if i == max_layer:
                delta["dz" + str(i)] = (derivative_error_function(forward_value.get("output"),y)
                                        * derivated_activation_function(forward_value.get("a" + str(i))))
                # Istruzione che va fatta solo per calcolare il dz dell'output layer, per quetso ho messo if specifico
                delta["dw" + str(i)] = (1 / m) * np.dot(delta.get("dz" + str(i)), forward_value.get("z" + str(i - 1)).T)
            elif i == 1:
                delta["dz" + str(i)] = (np.dot(parameters.get("W" + str(i + 1)).T,delta.get("dz" + str(i + 1)))
                                        * derivated_activation_function(forward_value.get("a" + str(i))))
                delta["dw" + str(i)] = (1 / m) * np.dot(delta.get("dz" + str(i)),
                                                        x.T)
                # Istruzione che va fatta solo per calcolare il dw tra input layer e primo hidden layer, per quetso ho messo if specifico
            else:
                delta["dz" + str(i)] = (np.dot(parameters.get("W" + str(i + 1)).T, delta.get("dz" + str(i + 1)))
                                        * derivated_activation_function(forward_value.get("a" + str(i))))
                delta["dw" + str(i)] = (1 / m) * np.dot(delta.get("dz" + str(i)), forward_value.get("z" + str(i - 1)).T)
            delta["db" + str(i)] = (1 / m) * np.sum(delta.get("dz" + str(i)), axis=1, keepdims=True)

        return delta

    # ------------------------------------- Update parameters gradient descent ---------------------------------------------
    def update_parameters(self, parameters, gradients, learning_rate):
        # Aggiorna i pesi e i bias utilizzando la discesa del gradiente

        num_itr = self.get_number_of_layers()
        for i in range(1, (num_itr + 1), 1):
            parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * gradients["dw" + str(i)]
            parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * gradients["db" + str(i)]

        return parameters

    # ------------------------------------- Update parameters RPROP ---------------------------------------------
    def rprop(self, etaP, etaM, dict_matrix_prec_deltas, dict_matrix_prec_grad, dict_array_prec_bias_deltas,
              dict_array_prec_bias_grad, weights, gradients):
        # Aggiorna i pesi e i bias utilizzando l'algoritmo RPROP
        etaPlus = etaP
        etaMinus = etaM
        delta = 0.0
        deltaMax = 50.0
        deltaMin = 1e-6

        for l in range(self.get_number_of_layers()):

            # Aggiornamento di W
            matrix_prec_grad = dict_matrix_prec_grad[l + 1]
            matrix_prec_deltas = dict_matrix_prec_deltas[l + 1]
            matrix_act_grad = gradients["dw" + str(l + 1)]
            for i in range(matrix_act_grad.shape[0]):
                for j in range(matrix_act_grad.shape[1]):
                    if (matrix_act_grad[i][j] * matrix_prec_grad[i][j] >= 0):
                        delta = matrix_prec_deltas[i][j] * etaPlus
                        if (delta > deltaMax): delta = deltaMax
                    elif (matrix_act_grad[i][j] * matrix_prec_grad[i][j] < 0):
                        delta = matrix_prec_deltas[i][j] * etaMinus
                        if (delta < deltaMin):
                            delta = deltaMin

                    weights["W" + str(l + 1)][i][j] = weights["W" + str(l + 1)][i][j] - (
                                np.sign(matrix_act_grad[i][j]) * delta)
                    matrix_prec_grad[i][j] = matrix_act_grad[i][j]
                    matrix_prec_deltas[i][j] = delta

            # Aggiornamento di b
            matrix_prec_bias_grad = dict_array_prec_bias_grad[l + 1]
            matrix_prec_bias_deltas = dict_array_prec_bias_deltas[l + 1]
            matrix_act_bias_grad = gradients["db" + str(l + 1)]
            for i in range(matrix_act_bias_grad.shape[0]):
                if (matrix_act_bias_grad[i][0] * matrix_prec_bias_grad[i][0] >= 0):
                    delta = matrix_prec_bias_deltas[i][0] * etaPlus
                    if (delta > deltaMax): delta = deltaMax
                elif (matrix_act_bias_grad[i][0] * matrix_prec_bias_grad[i][0] < 0):
                    delta = matrix_prec_bias_deltas[i][0] * etaMinus
                    if (delta < deltaMin):
                        delta = deltaMin
                weights["b" + str(l + 1)][i][0] = weights["b" + str(l + 1)][i][0] - (
                            np.sign(matrix_act_bias_grad[i][0]) * delta)
                matrix_prec_bias_grad[i][0] = matrix_act_bias_grad[i][0]
                matrix_prec_bias_deltas[i][0] = delta

        return weights

    def initialize_rprop_matrix(self):

        # Inizializza le matrici e i vettori necessari per l'algoritmo RPROP.
        # Ritorna:
        # dict_matrix_prec_deltas: Dizionario contenente le matrici di pesi precedenti per ciascuno strato
        # dict_matrix_prec_grad: Dizionario contenente le matrici di gradienti precedenti per ciascuno strato
        # dict_array_prec_bias_deltas: Dizionario contenente i vettori dei bias precedenti per ciascuno strato
        # dict_array_prec_bias_grad: Dizionario contenente i vettori di gradienti di bias precedenti per ciascuno strato

        dict_matrix_prec_deltas = {}
        dict_matrix_prec_grad = {}
        dict_array_prec_bias_deltas = {}
        dict_array_prec_bias_grad = {}
        for i in range(self.get_number_of_layers()):
            dict_matrix_prec_deltas[i + 1] = np.zeros(
                (self.layers["layer" + str(i + 1)], self.layers["layer" + str(i)]))
            dict_matrix_prec_grad[i + 1] = np.zeros((self.layers["layer" + str(i + 1)], self.layers["layer" + str(i)]))
            dict_matrix_prec_deltas[i + 1] += 0.01
            dict_array_prec_bias_deltas[i + 1] = np.zeros((self.layers["layer" + str(i + 1)], 1))
            dict_array_prec_bias_grad[i + 1] = np.zeros((self.layers["layer" + str(i + 1)], 1))
            dict_array_prec_bias_deltas[i + 1] += 0.01
        return dict_matrix_prec_deltas, dict_matrix_prec_grad, dict_array_prec_bias_deltas, dict_array_prec_bias_grad

    # ------------------------------------- Gradient descent with RPROP ---------------------------------------------

    def train_rprop(self, X_train, Y_train, X_val, Y_val, epochs, etaPlus, etaMinus, parameters, Y_train_one_hot,
                    error_function):
        # Addrestramento della rete neurale utilizzando l'algoritmo RPROP

        dict_matrix_prec_deltas, dict_matrix_prec_grad, dict_array_prec_bias_deltas, dict_array_prec_bias_grad = self.initialize_rprop_matrix()
        best_parameters = {}

        for i in range(epochs):
            forward_cache = self.forward_propagation(X_train, parameters)

            gradients = self.backward_prop(X_train, Y_train_one_hot, parameters, forward_cache)

            parameters = self.rprop(etaPlus, etaMinus, dict_matrix_prec_deltas, dict_matrix_prec_grad,
                                    dict_array_prec_bias_deltas, dict_array_prec_bias_grad, parameters, gradients)

            # Validation check
            forward_cache_val_set = self.forward_propagation(X_val, parameters)
            val_err = error_function(forward_cache_val_set["output"], Y_val.T)

            if val_err > 0.0 and error_function(forward_cache["output"], Y_train.T) > 0.0:
                self.train_errors.append(error_function(forward_cache["output"], Y_train.T))
                self.validation_errors.append(val_err)

            if i == 0:
                min_err = val_err
                for j in parameters.keys():
                    best_parameters[j] = np.copy(parameters.get(j))
                    best_parameters[j] = np.copy(parameters.get(j))
                self.best_epoch = i

            if val_err < min_err and val_err > 0.0:
                min_err = val_err
                self.best_epoch = i
                for j in parameters.keys():
                    best_parameters[j] = np.copy(parameters.get(j))
                    best_parameters[j] = np.copy(parameters.get(j))

            # End validation check

            if (i % (epochs / 10) == 0):
                print("Accuracy of Train Dataset after", i, "iterations: ",
                      self.accuracy(X_train, Y_train_one_hot, parameters), "%")
                print("Loss of Train Dataset after", i, "iterations: ",
                      error_function(forward_cache["output"], Y_train.T))

        return best_parameters

    # ------------------------------------- Gradient descent standard ---------------------------------------------
    def train_standard(self, X_train, Y_train, epochs, learning_rate, X_val, Y_val, parameters, Y_train_one_hot,
                       error_function):

        # Allenamento della rete neurale utilizzando il gradiente discendente standard
        for i in range(epochs):

            forward_cache = self.forward_propagation(X_train, parameters)

            gradients = self.backward_prop(X_train, Y_train_one_hot, parameters, forward_cache)

            parameters = self.update_parameters(parameters, gradients, learning_rate)

            # Validation check
            forward_cache_val_set = self.forward_propagation(X_val, parameters)
            val_err = error_function(forward_cache_val_set["output"], Y_val.T)

            self.train_errors.append(error_function(forward_cache["output"], Y_train.T))
            self.validation_errors.append(val_err)

            if i == 0:
                min_err = val_err
                best_parameters = parameters
                self.best_epoch = i

            if val_err < min_err:
                min_err = val_err
                best_parameters = parameters
                self.best_epoch = i

            # End validation check

            if (i % (epochs / 10) == 0):
                print("Accuracy of Train Dataset after", i, "iterations: ",
                      self.accuracy(X_train, Y_train_one_hot, parameters), "%")
                print("Loss of Train Dataset after", i, "iterations: ",
                      error_function(forward_cache["output"], Y_train.T))

        return best_parameters

    def train(self, X_train, Y_train, X_val, Y_val, type_update_parameter, learning_rate=0.02, etaPlus=1.2,
              etaMinus=0.5, epochs=50):
        # Funzione principale per l'allenamento della rete neurale

        parameters = {}
        self.check_iper_parameters()

        y_one_hot = self.one_hot(Y_train)
        parameters = self.initialize_weights_and_biases()
        error_function = self.dict_error_functions.get(self.error_function)[0]

        self.train_errors = []
        self.validation_errors = []

        if (type_update_parameter == "rprop"):
            best_parameters = self.train_rprop(X_train, Y_train, X_val, Y_val, epochs, etaPlus, etaMinus, parameters,
                                               y_one_hot, error_function)
        elif (type_update_parameter == "standard"):
            best_parameters = self.train_standard(X_train, Y_train, epochs, learning_rate, X_val, Y_val, parameters,
                                                  y_one_hot, error_function)
        else:
            self.eprint("The type of update parameter have to be standard or rprop")

        self.plot_error_on_epochs()

        return best_parameters

    def accuracy(self, inp, labels, parameters):
        # Calcola l'accuratezza della rete neurale rispetto ai dati di input e alle etichette fornite.

        forward_cache = self.forward_propagation(inp, parameters)
        a_out = forward_cache['output']  # contiene le probabilità con forma (10, 1)
        a_out = np.argmax(a_out, 0)  # 0 rappresenta l'asse delle righe
        labels = np.argmax(labels, 0)
        acc = np.mean(a_out == labels) * 100

        return acc

    def plot_error_on_epochs(self):
        # Plotta l'andamento degli errori durante l'allenamento della rete neurale.

        pyplot.plot(self.train_errors, label="Training set", color="red")
        pyplot.plot(self.validation_errors, label="Validation set", color="blue")
        ax = pl.gca()
        ylim = ax.get_ylim()
        pyplot.vlines(self.best_epoch, ylim[0], ylim[1], label="Minimo", color="green")
        pyplot.xlabel("Epoche")
        pyplot.ylabel("Errore")
        pyplot.legend()
        pyplot.show()
