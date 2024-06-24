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


def cal_cross_entropy_loss(y_true, y_pred):
    return np.sum(-1 * np.sum(y_true * np.log(y_pred), axis=0))


def cal_accuracy(y_true, y_pred):
    y_pred2 = np.argmax(y_pred, axis=0)
    y_true2 = np.argmax(y_true, axis=0)
    return np.sum(y_pred2 == y_true2) / y_true2.shape[0]


def cal_f1_score(y_true, y_pred):
    y_pred2 = np.argmax(y_pred, axis=0)
    y_true2 = np.argmax(y_true, axis=0)
    return f1_score(y_true2, y_pred2, average="macro")


def load_data(image_dirs, csv_files):
    images = []
    labels = []
    for image_dir, csv_file in zip(image_dirs, csv_files):
        f = pd.read_csv(csv_file)

        for files in os.listdir(image_dir):
            if not files.endswith(".png"):
                continue
            # if len(images) >= 50:
            #     break
            path = os.path.join(image_dir, files)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            image = 255 - image
            image = cv2.resize(image, (28, 28))
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)

            image = image.astype(np.float32) / 255
            image = image.reshape(28, 28, 1)
            image = np.array(image)
            images.append(image)
            labels.append(f.loc[f["filename"] == files, "digit"].values[0])
    images = np.array(images)
    labels = np.array(labels)
    labels = labels.reshape(-1, 1)
    return images, labels


def split_data(images, labels, split_ratio):
    split = int(len(images) * split_ratio)
    train_images = images[:split]
    train_labels = labels[:split]
    validation_images = images[split:]
    validation_labels = labels[split:]
    return train_images, train_labels, validation_images, validation_labels


def train(
    train_images,
    train_labels,
    validation_images,
    validation_labels,
    num_class,
    learning_rate,
    epochs,
    batch_size,
):
    x_axis = []
    y_axis_train_loss = []
    y_axis_train_accuracy = []
    y_axis_train_f1_score = []
    y_axis_validation_loss = []
    y_axis_validation_accuracy = []
    y_axis_validation_f1_score = []

    model = Model()
    best_model = copy.deepcopy(model)
    best_f1_score = -1e100

    y_train_true = np.eye(num_class)[train_labels.reshape(-1)]
    y_train_true = y_train_true.T

    y_true_validation = np.eye(num_class)[validation_labels.reshape(-1)]
    y_true_validation = y_true_validation.T

    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        for i in range(0, len(train_images), batch_size):
            X = None
            y_true = None
            if i + batch_size < len(train_images):
                X = train_images[i : i + batch_size]
                y_true = train_labels[i : i + batch_size]
            else:
                X = train_images[i:]
                y_true = train_labels[i:]
            y_true = np.eye(num_class)[y_true.reshape(-1)]
            y_true = y_true.T
            model.train(X, y_true, learning_rate)

        y_pred = model.predict(train_images)
        train_loss = cal_cross_entropy_loss(y_train_true, y_pred)
        train_acc = cal_accuracy(y_train_true, y_pred)
        train_f1 = cal_f1_score(y_train_true, y_pred)
        print("Train loss:", train_loss)
        print("Train accuracy:", train_acc)
        print("Train f1 score:", train_f1)
        print()

        y_pred_validation = model.predict(validation_images)
        validation_loss = cal_cross_entropy_loss(y_true_validation, y_pred_validation)
        validation_acc = cal_accuracy(y_true_validation, y_pred_validation)
        validation_f1 = cal_f1_score(y_true_validation, y_pred_validation)
        print("Validation loss:", validation_loss)
        print("Validation accuracy:", validation_acc)
        print("Validation f1 score:", validation_f1)
        print()
        if validation_f1 > best_f1_score:
            best_f1_score = validation_f1
            best_model = copy.deepcopy(model)

        x_axis.append(epoch + 1)
        y_axis_train_loss.append(train_loss)
        y_axis_train_accuracy.append(train_acc * 100)
        y_axis_train_f1_score.append(train_f1)
        y_axis_validation_loss.append(validation_loss)
        y_axis_validation_accuracy.append(validation_acc * 100)
        y_axis_validation_f1_score.append(validation_f1)

    plt.clf()
    plt.plot(x_axis, y_axis_train_loss, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch for Learning Rate = " + str(learning_rate))
    plt.legend()
    plt.savefig("Figures/train_loss_lr_" + str(learning_rate) + ".png")

    plt.clf()
    plt.plot(x_axis, y_axis_validation_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch for Learning Rate = " + str(learning_rate))
    plt.legend()
    plt.savefig("Figures/validation_loss_lr_" + str(learning_rate) + ".png")

    plt.clf()
    plt.plot(x_axis, y_axis_validation_accuracy, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy(%)")
    plt.title("Validation Accuracy vs Epoch for Learning Rate = " + str(learning_rate))
    plt.legend()
    plt.savefig("Figures/validation_accuracy_lr_" + str(learning_rate) + ".png")

    plt.clf()
    plt.plot(x_axis, y_axis_validation_f1_score, label="Validation f1 score")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Score")
    plt.title("Validation F1 Score vs Epoch for Learning Rate = " + str(learning_rate))
    plt.legend()
    plt.savefig("Figures/validation_f1_score_lr_" + str(learning_rate) + ".png")

    with open(
        "OutputCSV/learning_rate_" + str(learning_rate) + ".csv", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Training Accuracy",
                "Training F1 Score",
                "Validation Loss",
                "Validation Accuracy",
                "Validation F1 Score",
            ]
        )
        for i in range(len(x_axis)):
            writer.writerow(
                [
                    x_axis[i],
                    y_axis_train_loss[i],
                    y_axis_train_accuracy[i],
                    y_axis_train_f1_score[i],
                    y_axis_validation_loss[i],
                    y_axis_validation_accuracy[i],
                    y_axis_validation_f1_score[i],
                ]
            )

    y_pred = best_model.predict(validation_images)
    y_pred = np.argmax(y_pred, axis=0)
    y_true = validation_labels.reshape(-1)
    cm = confusion_matrix(y_true, y_pred)
    plt.clf()
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Learning Rate = " + str(learning_rate))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("Figures/confusion_matrix_lr_" + str(learning_rate) + ".png")
    return best_f1_score, best_model


def test(model, test_images, test_labels, num_class):
    y_pred = model.predict(test_images)
    y_true = np.eye(num_class)[test_labels.reshape(-1)]
    y_true = y_true.T
    test_loss = cal_cross_entropy_loss(y_true, y_pred)
    test_acc = cal_accuracy(y_true, y_pred)
    test_f1 = cal_f1_score(y_true, y_pred)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test f1 score:", test_f1)
    print()


if __name__ == "__main__":
    learning_rates = [0.001, 0.005, 0.01]
    epochs = 25
    batch_size = 32
    num_class = 10

    images, labels = load_data(
        ["Dataset/training-a", "Dataset/training-b", "Dataset/training-c"],
        ["Dataset/training-a.csv", "Dataset/training-b.csv", "Dataset/training-c.csv"],
    )

    train_images, train_labels, validation_images, validation_labels = split_data(
        images, labels, 0.8
    )
    print(
        "Train images shape:",
        train_images.shape,
        "\nTrain labels shape:",
        train_labels.shape,
    )
    print(
        "Validation images shape:",
        validation_images.shape,
        "\nValidation labels shape:",
        validation_labels.shape,
    )

    best_validation_f1_score = -1e100
    best_model = None
    best_learning_rate = None
    for learning_rate in learning_rates:
        print("Learning rate: ", learning_rate)
        cur_f1_score, model = train(
            train_images,
            train_labels,
            validation_images,
            validation_labels,
            num_class,
            learning_rate,
            epochs,
            batch_size,
        )
        if cur_f1_score > best_validation_f1_score:
            best_validation_f1_score = cur_f1_score
            best_model = copy.deepcopy(model)
            best_learning_rate = learning_rate

    print("Best learning rate: ", best_learning_rate)
    print("Best f1 score: ", best_validation_f1_score)

    best_model.clear()
    with open("model.pickle", "wb") as f:
        pickle.dump(best_model, f)

    test_images, test_labels = load_data(
        ["Dataset/training-d"], ["Dataset/training-d.csv"]
    )
    print(
        "Test images shape:",
        test_images.shape,
        "\nTest labels shape:",
        test_labels.shape,
    )
    test(best_model, test_images, test_labels, num_class)
