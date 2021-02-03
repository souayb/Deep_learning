import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Dense , Conv2D, concatenate, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.layers import Softmax, Dropout, Flatten,  ReLU, concatenate, BatchNormalization
from tensorflow.keras import Model as Model_ , Sequential

from utils import block_conv , Encoder_map, Decoder



class Full(Model_):
    def __init__(self, num_classes=10, drop=True):

        super(Full, self).__init__()
        filter = 64
        self.drop = drop
        self.down_conv1 = block_conv(64)
        self.maxpool1   = MaxPooling2D((2,2),strides=2)
        self.maxpool2   = MaxPooling2D((2,2),strides=2)
        self.maxpool3   = MaxPooling2D((2,2),strides=2)
        self.down_conv2 = block_conv(2*64)
        self.down_conv3 = block_conv(4*64)
        self.z          = block_conv(8*filter)



        self.upsample1   = UpSampling2D((2,2))
        self.upsample2   = UpSampling2D((2,2))
        self.upsample3   = UpSampling2D((2,2))
        self.up_conv1   = block_conv(4*filter)
        self.up_conv2   = block_conv(2*filter)
        self.up_conv3   = block_conv(filter)


        self.down_conv1_ = block_conv(64)
        self.maxpool1_   = MaxPooling2D((2,2),strides=2)
        self.maxpool2_   = MaxPooling2D((2,2),strides=2)
        self.maxpool3_   = MaxPooling2D((2,2),strides=2)
        self.down_conv2_ = block_conv(2*64)
        self.down_conv3_ = block_conv(4*64)
        self.z_          = block_conv(8*filter)


        self.encoder_map = Encoder_map()
        self.decoder    = Decoder()
        # global pooling
        self.avg_pool = tf.keras.layers.GlobalMaxPool2D()
        # self.flat      = tf.keras.layers.Flatten()
        self.classfier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.dropout1   = Dropout(0.2)
        self.dropout2   = Dropout(0.3)
        self.dropout3   = Dropout(0.4)
        self.dropout4   = Dropout(0.4)

    def call(self, input_x):
        dowconv1 = self.down_conv1(input_x)
        max1     = self.maxpool1(dowconv1 )
        max1     = self.dropout1(max1)
        dowconv2 = self.down_conv2(max1)
        max2     = self.maxpool2(dowconv2 )
        dowconv3 = self.down_conv3(max2)
        max3     = self.maxpool3(dowconv3 )
        lattent   = self.z(max3)

        upsamp1  = self.upsample1(lattent)
        upconv1  = self.up_conv1(concatenate([upsamp1, dowconv3], axis=-1))
        upsamp2  = self.upsample2(upconv1)
        upconv2  = self.up_conv2(concatenate([upsamp2, dowconv2],axis=-1))
        upsamp3  = self.upsample3(upconv2)
        upsamp3  = self.dropout2(upsamp3)
        upconv3  = self.up_conv3(concatenate([upsamp3, dowconv1],axis=-1))


        dowconv1_ = self.down_conv1_(upconv3)
        max1_     = self.maxpool1_(dowconv1_+dowconv1)
        dowconv2_ = self.down_conv2(max1_)
        max2_     = self.maxpool2_(dowconv2_+dowconv2)
        dowconv3_ = self.down_conv3(max2_)
        max3_     = self.maxpool3_(dowconv3_+dowconv3)
        max3_     = self.dropout3(max3_)
        lattent_   = self.z_(max3_)


        if self.drop:
         self.dropout2(lattent_+lattent)
        avg = self.avg_pool(lattent_)
        avg = self.dropout4(avg)
        # x = self.flat(avg)
        cat_out = self.classfier(avg)

        return cat_out


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train/255.
    x_test  = x_test / 255.
    Classifer_Dec10 = Full(num_classes=10)
    y_valids = {
            "output_1": x_test,
            "output_2": y_test,
    }

    Classifer_Dec10.compile(optimizer='adam', loss={'output_1': 'mean_absolute_error',
                        'output_2': 'sparse_categorical_crossentropy'}, metrics={
                            'output_1': 'mse',
                            'output_2': tf.keras.metrics.SparseCategoricalAccuracy(name='acc')})
    hist10 = Classifer_Dec10.fit(x_train, {'output_1': x_train , 'output_2': y_train}, batch_size=32, epochs=30, validation_data=(x_test, y_valids) )
