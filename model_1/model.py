import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

class CNNClassifier:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential()

        model.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(32, 3, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(16, 3, padding="same", activation="relu"))
        model.add(GlobalAveragePooling1D())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        return model


class CNNClassifier2:
    def __init__(self, input_shape, conv1_filters=64, conv2_filters=32, kernel_size=3):
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.kernel_size = kernel_size
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(self.conv1_filters, self.kernel_size, activation='relu', input_shape=self.input_shape, padding = "same"))
        model.add(MaxPooling1D(2))
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
        model.add(MaxPooling1D(3))
        model.add(Conv1D(self.conv2_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv3_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv4_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(GlobalAveragePooling1D())
        # 二分類
        model.add(Dense(1, activation='sigmoid'))
        return model