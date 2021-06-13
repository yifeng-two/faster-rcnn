"""
"""
import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, filters, strides, downsample=None):
        super(BasicBlock, self).__init__()
        # 第1个部分
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3),
                                            strides=strides,
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        # 第2个部分
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.downsample = downsample

    def __call__(self, inputs, training=None):
        # residual等于输入值本身，即residual=x
        residual = inputs
        # 将输入通过卷积、BN层、激活层，计算F(x)
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        out = self.relu(tf.keras.layers.add([out, residual]))
        return out


class BottleNeck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, strides, downsample=None):
        super(BottleNeck, self).__init__()
        # 第1个部分
        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 第2个部分
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3),
                                            strides=strides,
                                            padding='same',
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 第3个部分
        self.conv3 = tf.keras.layers.Conv2D(filters * self.expansion, (1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation('relu')

        # 对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        self.downsample = downsample

        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, inputs, training=None):

        residual = inputs
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out = self.relu(tf.keras.layers.add([out, residual]))

        return out


def get_layer_params(num_layers):
    layer_params = []
    if num_layers == 18:
        block = BasicBlock
        layer_params = [2, 2, 2, 2]
    elif num_layers == 34:
        block = BasicBlock
        layer_params = [3, 4, 6, 3]
    elif num_layers == 50:
        block = BottleNeck
        layer_params = [3, 4, 6, 3]
    elif num_layers == 101:
        block = BottleNeck
        layer_params = [3, 4, 23, 3]
    elif num_layers == 152:
        block = BottleNeck
        layer_params = [3, 8, 36, 3]
    return block, layer_params


class ResNetV1(tf.keras.Model):
    def __init__(self, num_layers):
        super(ResNetV1, self).__init__()
        self.block, self.layer_params = get_layer_params(num_layers)
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = self._make_layer(block=self.block,
                                       first_block=True,
                                       filter_num=64,
                                       blocks=self.layer_params[0],
                                       strides=1,
                                       name="block1")
        self.layer2 = self._make_layer(block=self.block,
                                       first_block=False,
                                       filter_num=128,
                                       blocks=self.layer_params[1],
                                       strides=2,
                                       name="block2")
        self.layer3 = self._make_layer(block=self.block,
                                       first_block=False,
                                       filter_num=256,
                                       blocks=self.layer_params[2],
                                       strides=2,
                                       name="block3")
        self.layer4 = self._make_layer(block=self.block,
                                       first_block=False,
                                       filter_num=512,
                                       blocks=self.layer_params[3],
                                       strides=2,
                                       name="block4")
        self.relu = tf.keras.layers.Activation('relu')

        # self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.fc = tf.keras.layers.Dense(units=21, activation=tf.keras.activations.softmax)

    def _make_layer(self, block, first_block, filter_num, blocks, strides=1, name=None):

        downsample = None
        # if strides != 1 or filters != filters * block.expansion:
        if strides != 1 or first_block is True:
            downsample = tf.keras.Sequential()
            downsample.add(
                tf.keras.layers.Conv2D(filter_num * block.expansion, (1, 1),
                                       strides=strides,
                                       use_bias=False))
            downsample.add(tf.keras.layers.BatchNormalization())

        res_block = tf.keras.Sequential(name=name)
        res_block.add(block(filter_num, strides, downsample))
        for _ in range(1, blocks):
            res_block.add(block(filter_num, strides=1))
        return res_block

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        output = self.layer4(x, training=training)
        # x = self.avgpool(x)
        # output = self.fc(x)

        return output


if __name__ == "__main__":

    model = ResNetV1(18)
    # model = ResNetV1(BottleNeck, 50)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

