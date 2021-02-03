import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense , Conv2D, concatenate, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.layers import Softmax, Dropout, Flatten,  ReLU, concatenate, BatchNormalization
from tensorflow.keras import Model as Model_ , Sequential

class block_conv(tf.keras.layers.Layer):
  def __init__(self, outchannel):
    super(block_conv, self).__init__()
    self.conv1 = Conv2D(outchannel,3, padding='same')
    self.BN1  = BatchNormalization()
    self.relu1 = ReLU()
    self.conv2 = Conv2D(outchannel,5, padding='same')
    self.BN2  = BatchNormalization()
    self.relu2 = ReLU()
  def call(self, x):
    x = self.conv1(x)
    x = self.relu1(self.BN1(x))
    x = self.conv2(x)
    x = self.relu2(self.BN2(x))
    return  x

class Decoder(Model_):
    def __init__(self):

        super(Decoder, self).__init__()
        filter = 64
        self.upsample1   = UpSampling2D((2,2))
        self.upsample2   = UpSampling2D((2,2))
        self.upsample3   = UpSampling2D((2,2))
        self.up_conv1   = block_conv(4*filter)
        self.up_conv2   = block_conv(2*filter)
        self.up_conv3   = block_conv(filter)
        self.out     = Conv2D(1,kernel_size=1,padding='same')

    def call(self, dowconv1, dowconv2, dowconv3, lattent):
        upsamp1  = self.upsample1(lattent)
        concat1  = concatenate([dowconv3,upsamp1],axis=3)
        upconv1  = self.up_conv1(concat1)
        upsamp2  = self.upsample2(upconv1)
        concat2  = concatenate([dowconv2,upsamp2],axis=3)
        upconv2  = self.up_conv2(concat2)
        upsamp3  = self.upsample3(upconv2)
        concat3  = concatenate([dowconv1,upsamp3],axis=3)
        upconv3  = self.up_conv3(concat3)
        out      = self.out(upconv3)
        return upconv1, upconv2, upconv3, out



class Encoder_map(Model_):

    """
    Encoder part of the model,


    """
    def __init__(self, num_classes=10):

        super(Encoder_map, self).__init__()
        filter = 64
        self.down_conv1 = block_conv(filter)
        self.maxpool = MaxPooling2D((2,2),strides=2)
        self.down_conv2 = block_conv(2*filter)
        self.down_conv3 = block_conv(4*filter)
        self.z          = block_conv(8*filter)

    def call(self, input_x):

        dowconv1 = self.down_conv1(input_x)
        max1     = self.maxpool(dowconv1)
        dowconv2 = self.down_conv2(max1)
        max2     = self.maxpool(dowconv2)
        dowconv3 = self.down_conv3(max2)
        max3     = self.maxpool(dowconv3)
        lattent   = self.z(max3)

        return dowconv1, dowconv2, dowconv3, lattent
