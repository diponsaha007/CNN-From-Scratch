import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import copy
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import pickle
import csv
import seaborn as sns


class ConvolutionalLayer:
    def __init__(self, num_output_channel, kernel_size, stride=1, padding=0):
        self.num_output_channel = num_output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None
        self.cache = None

    def getWindows(
        self, input, output_size, kernel_size, padding=0, stride=1, dilate=0
    ):
        working_input = input
        working_pad = padding
        if dilate != 0:
            working_input = np.insert(
                working_input, range(1, input.shape[2]), 0, axis=2
            )
            working_input = np.insert(
                working_input, range(1, input.shape[3]), 0, axis=3
            )

        if working_pad != 0:
            working_input = np.pad(
                working_input,
                pad_width=((0,), (0,), (working_pad,), (working_pad,)),
                mode="constant",
                constant_values=(0.0,),
            )

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = input.shape
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        return np.lib.stride_tricks.as_strided(
            working_input,
            (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
            (
                batch_str,
                channel_str,
                stride * kern_h_str,
                stride * kern_w_str,
                kern_h_str,
                kern_w_str,
            ),
        )

    def forward(self, X):
        X_tmp = X.copy()
        if self.weights is None:
            self.weights = np.random.randn(
                self.num_output_channel, X.shape[3], self.kernel_size, self.kernel_size
            ) * math.sqrt(2 / (self.kernel_size * self.kernel_size * X.shape[3]))
        if self.bias is None:
            self.bias = np.zeros(self.num_output_channel)
        X_tmp = np.transpose(X_tmp, (0, 3, 1, 2))
        n, c, h, w = X_tmp.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = self.getWindows(
            X_tmp, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride
        )

        out = np.einsum("bihwkl,oikl->bohw", windows, self.weights)

        out += self.bias[None, :, None, None]

        self.cache = X_tmp, windows
        out = np.transpose(out, (0, 2, 3, 1))

        return out

    def backward(self, delta, learning_rate):
        delta_tmp = delta.copy()
        delta_tmp = np.transpose(delta_tmp, (0, 3, 1, 2))
        x, windows = self.cache
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding
        dout_windows = self.getWindows(
            delta_tmp,
            x.shape,
            self.kernel_size,
            padding=padding,
            stride=1,
            dilate=self.stride - 1,
        )
        rot_kern = np.rot90(self.weights, 2, axes=(2, 3))
        db = np.sum(delta_tmp, axis=(0, 2, 3))
        dw = np.einsum("bihwkl,bohw->oikl", windows, delta_tmp)
        dx = np.einsum("bohwkl,oikl->bihw", dout_windows, rot_kern)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        dx = np.transpose(dx, (0, 2, 3, 1))
        return dx

    def clear_data(self):
        self.cache = None


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        X[X < 0] = 0
        return X

    def backward(self, delta, learning_rate):
        delta[self.X < 0] = 0
        return delta

    def clear_data(self):
        self.X = None


class MaxPoolingLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        x = np.transpose(X, (0, 3, 1, 2))
        n_batch, ch_x, h_x, w_x = x.shape

        out_h = int((h_x - self.kernel_size) / self.stride) + 1
        out_w = int((w_x - self.kernel_size) / self.stride) + 1

        windows = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_batch, ch_x, out_h, out_w, self.kernel_size, self.kernel_size),
            strides=(
                x.strides[0],
                x.strides[1],
                self.stride * x.strides[2],
                self.stride * x.strides[3],
                x.strides[2],
                x.strides[3],
            ),
        )
        out = np.max(windows, axis=(4, 5))

        maxs = out.repeat(2, axis=2).repeat(2, axis=3)
        x_window = x[:, :, : out_h * self.stride, : out_w * self.stride]
        mask = np.equal(x_window, maxs).astype(int)

        self.cache = (x, mask)

        out = np.transpose(out, (0, 2, 3, 1))
        return out

    def backward(self, delta, learning_rate):
        dA_prev = np.transpose(delta, (0, 3, 1, 2))
        (x, mask) = self.cache

        dA = dA_prev.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
        dA = np.multiply(dA, mask)
        pad = np.zeros(x.shape)
        pad[:, :, : dA.shape[2], : dA.shape[3]] = dA
        pad = np.transpose(pad, (0, 2, 3, 1))
        return pad

    def clear_data(self):
        self.cache = None


class FlatteningLayer:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        ret = X.copy()
        ret = ret.reshape(-1, np.prod(ret.shape[1:]))
        ret = ret.T
        return ret

    def backward(self, delta, learning_rate):
        ret = np.copy(delta)
        ret = ret.T
        ret = ret.reshape(self.X_shape)
        return ret

    def clear_data(self):
        pass


class FullyConnectedLayer:
    def __init__(self, output_size):
        self.output_size = output_size
        self.weights = None
        self.bias = None
        self.X = None

    def forward(self, X):
        self.X = X.copy()

        if self.weights is None:
            self.weights = np.random.randn(self.output_size, X.shape[0]) * math.sqrt(
                2 / X.shape[0]
            )

        if self.bias is None:
            self.bias = np.zeros((self.output_size, 1))

        ret = np.dot(self.weights, X) + self.bias
        return ret

    def backward(self, delta, learning_rate):
        dZ = delta.copy()
        dW = np.dot(dZ, self.X.T) / dZ.shape[1]
        db = np.reshape(np.mean(dZ, axis=1), (dZ.shape[0], 1))
        dX = np.dot(self.weights.T, dZ)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return dX

    def clear_data(self):
        self.X = None


class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, X):
        X = X - np.max(X, axis=0, keepdims=True)
        ret = np.exp(X)
        ret = ret / np.sum(ret, axis=0, keepdims=True)
        return ret

    def backward(self, delta, learning_rate):
        return delta

    def clear_data(self):
        pass


class Model:
    def __init__(self):
        self.layers = []
        self.build_Lenet()

    def build_Lenet(self):
        self.layers.append(ConvolutionalLayer(6, 5, 1, 2))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(2, 2))
        self.layers.append(ConvolutionalLayer(16, 5, 1, 0))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(2, 2))
        self.layers.append(FlatteningLayer())
        self.layers.append(FullyConnectedLayer(120))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(84))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(10))
        self.layers.append(SoftmaxLayer())

    def train(self, X, y_true, learning_rate):
        X_tmp = X.copy()
        for i in range(len(self.layers)):
            X_tmp = self.layers[i].forward(X_tmp)

        delta = X_tmp - y_true
        for i in range(len(self.layers) - 1, -1, -1):
            delta = self.layers[i].backward(delta, learning_rate)

    def predict(self, X):
        X_tmp = X.copy()
        for i in range(len(self.layers)):
            X_tmp = self.layers[i].forward(X_tmp)
        return X_tmp

    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear_data()


if __name__ == "__main__":
    num_class = 10
    best_model = Model()
    with open("model.pkl", "rb") as f:
        best_model = pickle.load(f)

    best_model.clear()

    with open("model2.pkl", "wb") as f:
        pickle.dump(best_model, f)
