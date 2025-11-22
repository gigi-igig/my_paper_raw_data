import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

class CNNClassifier2:
    def __init__(self, input_shape,
                 conv_filters=[64, 128, 256],
                 kernel_size=3,
                 dropout_rate=0.3):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        # ===== Conv Layer 1 =====
        model.add(Conv1D(self.conv_filters[0], self.kernel_size,
                         padding="same", strides=1,
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # ===== Conv Layer 2 =====
        model.add(Conv1D(self.conv_filters[1], self.kernel_size,
                         padding="same", strides=2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # ===== Conv Layer 3 =====
        model.add(Conv1D(self.conv_filters[2], self.kernel_size,
                         padding="same", strides=2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # ===== Dense Head =====
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation="sigmoid"))

        return model


class CNNClassifier2:
    def __init__(self, input_shape, conv1_filters=64, conv2_filters=128, kernel_size=3):
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.kernel_size = kernel_size
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(self.conv1_filters, self.kernel_size, activation='relu', input_shape=self.input_shape, padding = "same"))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(self.conv2_filters, self.kernel_size, activation='relu', padding = "same"))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

class CNNClassifier4:
    def __init__(self, input_shape,
                 conv1_filters=64, conv2_filters=128,
                 conv3_filters=128, conv4_filters=64,
                 kernel_size=3):
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        self.conv4_filters = conv4_filters
        self.kernel_size = kernel_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # 四層 Conv1D, padding='same'
        model.add(Conv1D(self.conv1_filters, self.kernel_size, activation='relu',
                         padding='same', input_shape=self.input_shape))
        model.add(Conv1D(self.conv2_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv3_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(Conv1D(self.conv4_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(GlobalAveragePooling1D())
        # 二分類
        model.add(Dense(1, activation='sigmoid'))
        return model