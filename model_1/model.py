import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense

class CNNClassifier:
    def __init__(self, input_shape, conv1_filters=64, conv2_filters=128, kernel_size=3):
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = 128
        self.conv4_filters = 64
        self.kernel_size = kernel_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(self.conv1_filters, self.kernel_size, activation='relu',
                         padding='same', input_shape=self.input_shape))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv2_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv3_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.conv4_filters, self.kernel_size, activation='relu', padding='same'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model