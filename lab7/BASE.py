import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Dense , Conv2D, concatenate, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.layers import Softmax, Dropout, Flatten,  ReLU, concatenate, BatchNormalization
from tensorflow.keras import Model as Model_ , Sequential
from utils import block_conv , Encoder_map, Decoder



class BASE(Model_):
    def __init__(self, num_classes=10, drop=True):

        super(BASE, self).__init__()
        filter = 64
        self.drop = drop
        self.down_conv1 = block_conv(filter)
        self.maxpool1   = MaxPooling2D((2,2),strides=2)
        self.maxpool2   = MaxPooling2D((2,2),strides=2)
        self.maxpool3   = MaxPooling2D((2,2),strides=2)
        self.down_conv2 = block_conv(2*filter)
        self.down_conv3 = block_conv(4*filter)
        self.z          = block_conv(8*filter)


        # global pooling
        self.avg_pool = tf.keras.layers.GlobalMaxPool2D()
        # self.flat      = tf.keras.layers.Flatten()
        self.classfier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.dropout1   = Dropout(0.2)
        self.dropout2   = Dropout(0.3)
        self.dropout3   = Dropout(0.4)

    def call(self, input_x):
        dowconv1 = self.down_conv1(input_x)
        max1     = self.maxpool1(dowconv1)
        max1     = self.dropout1(max1)
        dowconv2 = self.down_conv2(max1)
        max2     = self.maxpool2(dowconv2)
        dowconv3 = self.down_conv3(max2)
        max3     = self.maxpool3(dowconv3)
        lattent   = self.z(max3)

        if self.drop:
          self.dropout2(lattent)
        avg = self.avg_pool(lattent)
        avg = self.dropout3(avg)
        # x = self.flat(avg)
        cat_out = self.classfier(avg)

        return cat_out

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train/255.
    x_test  = x_test / 255.
    Model = BASE(num_classes=10)
    Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cifat10_loss = Model.fit(x_train, y_train, batch_size=43, epochs=30, validation_data=(x_test, y_test))
