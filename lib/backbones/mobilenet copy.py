"""
"""
import tensorflow as tf


def conv_block(
        inputs,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(6.0)(x)


def depthwise_conv_block(
        inputs,
        pointwise_conv_filters,
        strides=(1, 1)
):
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU(6.0)(x)


def mobilenet_v1(
        inputs,
        classes
):
    x = conv_block(inputs, 32, strides=(2, 2))
    x = depthwise_conv_block(x, 64)
    x = depthwise_conv_block(x, 128, strides=(2, 2))
    x = depthwise_conv_block(x, 128)
    x = depthwise_conv_block(x, 256, strides=(2, 2))
    x = depthwise_conv_block(x, 256)
    x = depthwise_conv_block(x, 512, strides=(2, 2))
    x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 1024, strides=(2, 2))
    x = depthwise_conv_block(x, 1024)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    return x





if __name__ == "__main__":

    inputs = tf.keras.Input(shape=(224, 224, 3))
    model = tf.keras.Model(inputs=inputs, outputs=mobilenet_v1(inputs, 10))
    model.summary()
