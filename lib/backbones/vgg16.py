"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import tensorflow as tf


class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(32,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(32,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(64,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(64,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(128,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(128,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(256,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(256,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512,
                                              3,
                                              padding='same',
                                              activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512,
                                              3,
                                              padding='same',
                                              activation='relu')

    def call(self, x, training=None):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_2(self.conv3_1(x)))
        x = self.pool4(self.conv4_2(self.conv4_1(x)))
        x = self.conv5_3(self.conv5_2(self.conv5_1(x)))
        return x


if __name__ == "__main__":
    model = VGG16()
    model.build((None, 224, 224, 3))
    model.summary()