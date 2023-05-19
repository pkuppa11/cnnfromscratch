# I'm implementing a Convolutional Neural Network rather than a regular Neural Network because CNNs are awesome :)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def one_hot_encode(inds, num_classes):
    arr = np.zeros((len(inds), num_classes), dtype=np.float32)
    arr[np.arange(len(inds)), inds] = 1
    return arr

def zero_pad(X, padding):
    return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), "constant", constant_values=(0, 0))

class Conv2D:
    def __init__(self, filters, filter_size, input_channels=3, padding=0, stride=1,lr=0.001, optimizer=None):
        self.filters = filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.padding = padding
        self.stride = stride
        self.lr = lr
        self.optimizer = optimizer
        self.cache = None               #save time during backprop
        self.initialized = False
        
    def relu(self, Z):
        A = np.maximum(0,Z)
        cache = Z 
        return A, cache
    
    def relu_backward(self, dA, activation_cache):
        Z = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
        
    def conv_single_step(self, a_slice_prev, W, b):
        s = np.multiply(a_slice_prev, W)
        Z = np.sum(s)
        Z = Z + float(b)
        return Z

    def ff(self, A_prev):
        activation_caches = []
        if self.initialized == False:
            np.random.seed(0)
            self.W = np.random.randn(self.filter_size, self.filter_size, A_prev.shape[-1], self.filters) 
            self.b = np.random.randn(1, 1, 1, self.filters)
            self.initialized = True
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = self.W.shape

        n_H = int((n_H_prev - f + (2 * self.padding)) / self.stride) + 1
        n_W = int((n_W_prev - f + (2 * self.padding)) / self.stride) + 1
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = zero_pad(A_prev, self.padding)
        for i in range(m):
            a_prev_pad = A_prev_pad[i]  
            for h in range(n_H): 
                vert_start = h * self.stride
                vert_end = vert_start + f
                for w in range(n_W):  
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):  
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = self.W[:, :, :, c]
                        biases = self.b[:, :, :, c] 
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, weights, biases)
                Z[i], activation_cache = self.relu(Z[i])
                activation_caches.append(activation_cache)
        self.cache = (A_prev, np.array(activation_caches))
    
        return Z

    def bp(self, dZ):
        A_prev, activation_cache = self.cache
        W, b = self.W, self.b
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape

        stride = self.stride
        pad = self.padding
        
        (m, n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        self.dW = np.zeros((f, f, n_C_prev, n_C))
        self.db = np.zeros((1, 1, 1, n_C))

        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
        for i in range(m):
            dZ[i] = self.relu_backward(dZ[i], activation_cache[i])
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(n_H):                   
                vert_start = h * stride
                vert_end = vert_start + f
                for w in range(n_W):               
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):          
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        self.dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        self.db[:,:,:,c] += dZ[i, h, w, c]
                        
            if pad:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i, :, :, :] = dA_prev[i, :, :, :]
    
        
        self.update_parameters(self.optimizer)
        return dA_prev

    def Adam(self, beta1=0.9, beta2=0.999):
        self.epsilon = 1e-8
        self.v_dW = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)
        self.s_dW = np.zeros(self.W.shape)
        self.s_db = np.zeros(self.b.shape)
        self.t = 1

        self.v_dW = beta1 * self.v_dW + (1-beta1) * self.dW
        self.v_db = beta1 * self.v_db + (1-beta1) * self.db
        self.v_dW_corrected = self.v_dW / (1-beta1**self.t)
        self.v_db_corrected = self.v_db / (1-beta1**self.t)

        self.s_dW = beta2 * self.s_dW + (1-beta2) * np.square(self.dW)
        self.s_db = beta2 * self.s_db + (1-beta2) * np.square(self.db)
        self.s_dW_corrected = self.s_dW / (1-beta2**self.t)
        self.s_db_corrected = self.s_db / (1-beta2**self.t)

        self.t += 1

        self.W = self.W - self.lr * (self.v_dW_corrected / (np.sqrt(self.s_dW_corrected) + self.epsilon))
        self.b = self.b - self.lr * (self.v_db_corrected / (np.sqrt(self.s_db_corrected) + self.epsilon))

    def update_parameters(self,optimizer=None):
        if optimizer == 'adam':
            self.Adam()
        else:   
            self.W = self.W - self.lr * self.dW 
            self.b = self.b - self.lr * self.db 

class Pooling2D:
    def __init__(self, filter_size, stride, mode='max'):
        self.filter_size = filter_size 
        self.stride = stride
        self.mode = mode

    def ff(self, A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        f = self.filter_size
        stride = self.stride

        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev 

        A = np.zeros((m, n_H, n_W, n_C))              
        for i in range(m):
            for h in range(n_H):                     
                vert_start = h * stride
                vert_end = vert_start + f
                for w in range(n_W):                 
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range (n_C):        
                        a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] =  np.mean(a_prev_slice)
        self.cache = (A_prev)
        return A
      
    def create_mask_from_window(self, x):
        mask = (x == x.max())
        return mask

    def distribute_value(self, dz, shape):  
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones((n_H, n_W)) * average
  
        return a
                      
    def bp(self, dA):
        A_prev = self.cache

        stride = self.stride
        f = self.filter_size

        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        dA_prev = np.zeros((A_prev.shape))

        for i in range(m): 
            a_prev = A_prev[i, :, :, :]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):          
                        vert_start = h * stride
                        vert_end = vert_start + f 
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        if self.mode == "max":
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += (mask * dA[i, h, w, c])
                        elif self.mode == "average":
                            da = dA[i, h, w, c]
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, (f, f))
        return dA_prev

class Flatten:
    def __init__(self):
        self.input_shape = None
    
    def ff(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def bp(self, X):
        return X.reshape(self.input_shape)
    
class Dense:
    def __init__(self, units, activation="relu", optimizer=None, lr=0.001):
        self.units = units
        self.W = None
        self.b = None 
        self.activation = activation 
        self.input_shape = None  
        self.optimizer = optimizer 
        self.lr = lr

    def ff(self, A):
        if self.W is None:
            self.initialize_weights(A.shape[1])
        self.A = A
        out = np.dot(A, self.W) + self.b
        if self.activation == "relu":
            out = np.maximum(0, out)
        elif self.activation == "softmax":
            out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)

        return out

    def initialize_weights(self, input_shape):
        np.random.seed(0)
        self.input_shape = input_shape
        self.W = np.random.randn(input_shape, self.units) * 0.01
        self.b = np.zeros((1, self.units))
    
    def cross_entropy_loss(self, y_true, y_pred):
        loss = - np.mean(y_true * np.log(y_pred + 1e-7))  
        return loss

    def bp(self, dout):
        dA = np.dot(dout, self.W.T)
        self.dW = np.dot(self.A.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        self.update_parameters(self.optimizer)
        return dA

    def update_parameters(self, optimizer=None):
        if self.optimizer == 'adam':
            self.Adam()
        else:
            self.W -= self.lr * self.dW
            self.b -= self.lr * self.db

    def Adam(self, beta1=0.9, beta2=0.999):
        self.v_dW = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)
        self.s_dW = np.zeros(self.W.shape)
        self.s_db = np.zeros(self.b.shape)
    
        self.v_dW = beta1 * self.v_dW + (1 - beta1) * self.dW
        self.v_db = beta1 * self.v_db + (1 - beta1) * self.db
    
        self.v_dW = self.v_dW / (1 - beta1 ** 2)
        self.v_db = self.v_db / (1 - beta1 ** 2)
    
        self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
        self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)
    
        self.s_dW = self.s_dW / (1 - beta2 ** 2)
        self.s_db = self.s_db / (1 - beta2 ** 2)
    
        self.W = self.W - self.lr * self.dW
        self.b = self.b - self.lr * self.db

class CNN:
    def __init__(self,layers, lr=0.001, optimizer=None):
        self.layers = layers
        self.lr = lr
        self.optimizer = optimizer
        self.initialize_network()

    def initialize_network(self):
        for layer in self.layers:
            if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                layer.lr  = self.lr
                layer.optimizer = self.optimizer

    def forward(self, inputs):
        inputs = self.layers[0].ff(inputs)
        for layer in self.layers[1:]:
            inputs = layer.ff(inputs)
        return inputs

    def backward(self, inputs):
        inputs = self.layers[-1].bp(inputs)
        for layer in reversed(self.layers[:-1]):
            inputs = layer.bp(inputs)

    def compute_cost(self, y_true, y_pred):
        if isinstance(self.layers[-1], Dense):
            cost = self.layers[-1].cross_entropy_loss(y_true, y_pred)
            return cost
        else:
            raise ValueError("The last layer in the layers list should be a Dense layer.")

    def step_decay(self, epoch, lr, decay_rate=0.1, decay_step=10, lowest_lr=1e-05):
        if lr > lowest_lr:
            new_lr = lr * (decay_rate ** (epoch // decay_step))
        else:
            new_lr = lr
        return new_lr

    def fit(self, X, y, epochs=10, decay_rate=0.2, print_cost=True, plot_accuracy=False):
        costs = []
        accuracies = []
        for i in range(epochs):
            self.lr = self.step_decay(i, self.lr, decay_rate)
            predictions = self.forward(X)
            cost = self.compute_cost(y, predictions)
            accuracy = (np.argmax(predictions, axis=1) == np.argmax(y, axis=1)).mean()
            dout = predictions - y
            gradients = self.backward(dout)
            costs.append(cost)
            accuracies.append(accuracy)
            if print_cost:
                print(f"Epoch {i}: Cost {cost}, Accuracy: {str(accuracy*100)}%")
        if plot_accuracy:
            fig = px.line(y=np.squeeze(accuracies),title='Accuracy',template="plotly_dark")
            fig.update_layout(
                title_font_color="#f6abb6", 
                xaxis=dict(color="#f6abb6"), 
                yaxis=dict(color="#f6abb6") 
            )
            fig.show()
  
    def predict(self, X):
        predictions = self.forward(X)
        return predictions

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss = self.compute_cost(y, y_pred)
        accuracy = ((np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).mean()) * 100
        return f'{accuracy}%'

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    num_classes = train.label.nunique()
    X = train.iloc[:,1:].values
    X = X.reshape(-1, 28, 28, 1) / 255
    y = one_hot_encode(train.iloc[:,0].values, num_classes)

    X_train, X_test, y_train, y_test = X[:np.int32(len(X)*0.01)], X[np.int32(len(X)*0.01):np.int32(len(X)*0.02)], y[:np.int32(len(X)*0.01)], y[np.int32(len(X)*0.01):np.int32(len(X)*0.02)]

    layers = [
        Conv2D(8, 3, padding=0, lr=0.0001),
        Pooling2D(2, 2, "max"),
        Flatten(),
        #Dense(754, activation="relu"),
        #Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ]

    model = CNN(layers, lr=0.0001, optimizer="adam")
    model.fit(X_train, y_train, 200, 0.5, True, True)
    print("="*100)
    model.evaluate(X_test, y_test)

if __name__ == '__main__': main()
