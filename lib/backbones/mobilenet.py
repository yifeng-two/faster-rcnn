"""
"""
import tensorflow as tf


# for MobileNetV1
class depthwise_conv_block(tf.keras.layers.Layer):
    def __init__(self, pointwise_conv_filters, strides=1):
        super(depthwise_conv_block, self).__init__()
        self.depth_wise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                               strides=strides,
                                                               padding='SAME',
                                                               use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filters=pointwise_conv_filters,
                                           kernel_size=(1, 1),
                                           padding='SAME',
                                           use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def __call__(self, inputs, training=None, mask=None):

        out = self.depth_wise_conv(inputs)
        out = self.bn1(out, training=training)
        out = tf.keras.activations.relu(out, 6)
        out = self.conv(out)
        out = self.bn2(out, training=training)
        out = tf.keras.activations.relu(out, 6)

        return out


class MobileNetV1(tf.keras.Model):
    def __init__(self, alpha):
        super(MobileNetV1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32 * alpha,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding='SAME',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.depthwise_conv_block1 = self._make_depthwise_conv_block(pointwise_conv_filters=64 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block2 = self._make_depthwise_conv_block(pointwise_conv_filters=128 *
                                                                     alpha,
                                                                     strides=2)
        self.depthwise_conv_block3 = self._make_depthwise_conv_block(pointwise_conv_filters=128 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block4 = self._make_depthwise_conv_block(pointwise_conv_filters=256 *
                                                                     alpha,
                                                                     strides=2)
        self.depthwise_conv_block5 = self._make_depthwise_conv_block(pointwise_conv_filters=256 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block6 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                     alpha,
                                                                     strides=2)

        self.depthwise_conv_block7 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block8 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block9 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                     alpha,
                                                                     strides=1)
        self.depthwise_conv_block10 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                      alpha,
                                                                      strides=1)
        self.depthwise_conv_block11 = self._make_depthwise_conv_block(pointwise_conv_filters=512 *
                                                                      alpha,
                                                                      strides=1)

        self.depthwise_conv_block12 = self._make_depthwise_conv_block(pointwise_conv_filters=1024 *
                                                                      alpha,
                                                                      strides=2)
        self.depthwise_conv_block13 = self._make_depthwise_conv_block(pointwise_conv_filters=1024 *
                                                                      alpha,
                                                                      strides=1)

        self.GAPool = tf.keras.layers.GlobalAveragePooling2D()
        # pred = tf.keras.layers.Dense(classes, activation='softmax')(x)

    def _make_depthwise_conv_block(self, pointwise_conv_filters, strides):
        block = tf.keras.Sequential()
        block.add(
            depthwise_conv_block(pointwise_conv_filters=pointwise_conv_filters, strides=strides))
        return block

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.keras.activations.relu(out, 6)
        out = self.depthwise_conv_block1(out)
        out = self.depthwise_conv_block2(out)
        out = self.depthwise_conv_block3(out)
        out = self.depthwise_conv_block4(out)
        out = self.depthwise_conv_block5(out)
        out = self.depthwise_conv_block6(out)
        out = self.depthwise_conv_block7(out)
        out = self.depthwise_conv_block8(out)
        out = self.depthwise_conv_block9(out)
        out = self.depthwise_conv_block10(out)
        out = self.depthwise_conv_block11(out)
        out = self.depthwise_conv_block12(out)
        out = self.depthwise_conv_block13(out)
        out = self.GAPool(out)

        return out


# for MobileNetV2
class depthwise_res_block(tf.keras.layers.Layer):
    # expansion = 4

    def __init__(self, input_filters, out_filters, strides, expansion, resdiual=False):
        super(depthwise_res_block, self).__init__()
        # 第1个部分
        self.conv1 = tf.keras.layers.Conv2D(input_filters * expansion, (1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 第2个部分
        self.depth_wise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                               strides=strides,
                                                               padding='SAME',
                                                               use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 第3个部分
        self.conv3 = tf.keras.layers.Conv2D(out_filters, (1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation('relu')

        # 对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        self.resdiual = resdiual

    def __call__(self, inputs, training=None):

        residual = inputs
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.keras.activations.relu(out, 6)

        out = self.depth_wise_conv(out)
        out = self.bn2(out, training=training)
        out = tf.keras.activations.relu(out, 6)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.resdiual:
            out = tf.keras.layers.add([out, residual])

        return out


class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # 每个inverted_residual_layers输入参数,input_filte,out_filters,expansion,n_repeats,strides
        # [32,16,1,1,1]
        self.layers_params = [
            [32, 16, 1, 1, 1],
            [16, 24, 6, 2, 2],
            [24, 32, 6, 3, 2],
            [32, 64, 6, 4, 2],
            [64, 96, 6, 3, 1],
            [96, 160, 6, 3, 2],
            [160, 320, 6, 1, 1],
        ]
        # 第1个部分
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.depth_wise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                               strides=1,
                                                               padding='SAME',
                                                               use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # start inverted_residual_layers
        self.layer1 = self._make_inverted_residual_layers(input_filters=self.layers_params[0][0],
                                                          out_filters=self.layers_params[0][1],
                                                          expansion=self.layers_params[0][2],
                                                          n_repeats=self.layers_params[0][3],
                                                          strides=self.layers_params[0][4],
                                                          name='block1')
        self.layer2 = self._make_inverted_residual_layers(input_filters=self.layers_params[1][0],
                                                          out_filters=self.layers_params[1][1],
                                                          expansion=self.layers_params[1][2],
                                                          n_repeats=self.layers_params[1][3],
                                                          strides=self.layers_params[1][4],
                                                          name='block2')
        self.layer3 = self._make_inverted_residual_layers(input_filters=self.layers_params[2][0],
                                                          out_filters=self.layers_params[2][1],
                                                          expansion=self.layers_params[2][2],
                                                          n_repeats=self.layers_params[2][3],
                                                          strides=self.layers_params[2][4],
                                                          name='block3')
        self.layer4 = self._make_inverted_residual_layers(input_filters=self.layers_params[3][0],
                                                          out_filters=self.layers_params[3][1],
                                                          expansion=self.layers_params[3][2],
                                                          n_repeats=self.layers_params[3][3],
                                                          strides=self.layers_params[3][4],
                                                          name='block4')
        self.layer5 = self._make_inverted_residual_layers(input_filters=self.layers_params[4][0],
                                                          out_filters=self.layers_params[4][1],
                                                          expansion=self.layers_params[4][2],
                                                          n_repeats=self.layers_params[4][3],
                                                          strides=self.layers_params[4][4],
                                                          name='block5')
        self.layer6 = self._make_inverted_residual_layers(input_filters=self.layers_params[5][0],
                                                          out_filters=self.layers_params[5][1],
                                                          expansion=self.layers_params[5][2],
                                                          n_repeats=self.layers_params[5][3],
                                                          strides=self.layers_params[5][4],
                                                          name='block6')

        self.conv7 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.GAPool = tf.keras.layers.GlobalAveragePooling2D()

    def _make_inverted_residual_layers(self,
                                       input_filters,
                                       out_filters,
                                       expansion,
                                       n_repeats,
                                       strides,
                                       name=None):

        inverted_res_block = tf.keras.Sequential(name=name)
        inverted_res_block.add(
            depthwise_res_block(input_filters=input_filters,
                                out_filters=out_filters,
                                strides=strides,
                                expansion=expansion,
                                resdiual=False))

        for _ in range(1, n_repeats):
            inverted_res_block.add(
                depthwise_res_block(input_filters=out_filters,
                                    out_filters=out_filters,
                                    strides=1,
                                    expansion=expansion,
                                    resdiual=True))

        return inverted_res_block

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.depth_wise_conv(out)
        out = self.bn2(out, training=training)
        out = tf.keras.activations.relu(out, 6)
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        out = self.layer5(out, training=training)
        out = self.layer6(out, training=training)
        out = self.conv7(out)
        out = self.bn7(out, training=training)
        out = self.GAPool(out)
        return out


# for mobilenetV3
def h_sigmoid(x):

    output = tf.keras.layers.Activation('hard_sigmoid')(x)
    return output


def h_swish(x):

    output = x * h_sigmoid(x)
    return output


class SE_layer(tf.keras.layers.Layer):
    def __init__(self, filters, gama=16):
        super(SE_layer, self).__init__()
        self.filter_sq = filters // gama
        self.GApool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(self.filter_sq)
        self.relu = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(filters)
        self.reshape = tf.keras.layers.Reshape((1, 1, filters))

    def __call__(self, inputs, training=None):
        out = self.GApool(inputs)
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dense2(out)
        out = h_sigmoid(out)
        out = self.reshape(out)
        out = out * inputs
        return out


class depthwise_bottle_neck(tf.keras.layers.Layer):
    def __init__(self, inpput_filters, exp_filters, out_filters, kernel_size, strides, SE_flag,
                 NL_flag):
        super(depthwise_bottle_neck, self).__init__()
        self.input_filters = inpput_filters
        self.output_filters = out_filters
        self.strides = strides
        self.SE_flag = SE_flag
        self.NL_flag = NL_flag
        self.conv1 = tf.keras.layers.Conv2D(filters=exp_filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.depth_wise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                                strides=strides,
                                                                padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se_layer = SE_layer(filters=exp_filters)
        self.conv3 = tf.keras.layers.Conv2D(filters=out_filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.linear3 = tf.keras.layers.Activation('linear')

    def __call__(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        if self.NL_flag == "HS":
            out = h_swish(out)
        elif self.NL_flag == "RE":
            out = tf.keras.activations.relu(out, 6)
        if self.SE_flag:
            out = self.se_layer(out)
        out = self.depth_wise_conv2(out)
        out = self.bn2(out, training=training)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        out = self.linear3(out)
        if self.strides == 1 and self.input_filters == self.output_filters:
            out = tf.keras.layers.add([out, inputs])
        return out


class MobileNetV3(tf.keras.Model):
    def __init__(self, type='large'):
        super(MobileNetV3, self).__init__()
        self.layer_params = self._get_layer_param(type)
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.bottle_layers = tf.keras.Sequential()
        for i in range(len(self.layer_params)):
            self.bottle_layers.add(
                depthwise_bottle_neck(inpput_filters=self.layer_params[i][0],
                                      exp_filters=self.layer_params[i][1],
                                      out_filters=self.layer_params[i][2],
                                      kernel_size=self.layer_params[i][3],
                                      strides=self.layer_params[i][4],
                                      SE_flag=self.layer_params[i][5],
                                      NL_flag=self.layer_params[i][6]))

        self.conv2 = tf.keras.layers.Conv2D(filters=960,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv2 = tf.keras.layers.BatchNormalization()
        # x = h_swish(x)
        self.AVpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        # x = h_swish(x)

    def _get_layer_param(self, type):
        # 参数依次代表：input_filters, exp_filters, output_filters,kernel_size,strides,SE_flag,NL_flag
        # {
        #         'input_filters': 16,
        #         'exp_filters': 16,
        #         'output_filters': 16,
        #         'kernel_size': 3,
        #         'strides': 1,
        #         'SE_flag': False,
        #         'NL_flag': 'RE'
        #     }
        if type == 'large':
            layer_params = [
                [16, 16, 16, 3, 1, False, 'RE'],
                [16, 64, 24, 3, 2, False, 'RE'],
                [24, 72, 24, 3, 1, False, 'RE'],
                [24, 72, 40, 5, 2, True, 'RE'],
                [40, 120, 40, 5, 1, True, 'RE'],
                [40, 120, 40, 5, 1, True, 'RE'],
                [40, 240, 80, 3, 2, False, 'HS'],
                [80, 200, 80, 3, 1, False, 'HS'],
                [80, 184, 80, 3, 1, False, 'HS'],
                [80, 184, 80, 3, 1, False, 'HS'],
                [80, 480, 112, 3, 1, True, 'HS'],
                [112, 672, 112, 3, 1, True, 'HS'],
                [112, 672, 160, 5, 2, True, 'HS'],
                [160, 960, 160, 5, 1, True, 'HS'],
                [160, 960, 160, 5, 1, True, 'HS'],
            ]
        elif type == 'small':
            layer_params = [
                [16, 16, 16, 3, 2, True, 'RE'],
                [16, 72, 24, 3, 2, False, 'RE'],
                [24, 88, 24, 5, 1, False, 'RE'],
                [24, 96, 40, 5, 2, True, 'HS'],
                [40, 240, 40, 3, 1, True, 'HS'],
                [40, 240, 40, 3, 1, True, 'HS'],
                [40, 120, 48, 3, 1, True, 'HS'],
                [48, 144, 48, 3, 1, True, 'HS'],
                [48, 288, 96, 3, 2, True, 'HS'],
                [96, 576, 96, 3, 1, True, 'HS'],
                [96, 576, 96, 5, 1, True, 'HS'],
            ]
        return layer_params

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = h_swish(out)
        out = self.bottle_layers(out, training=training)
        out = self.conv2(out)
        out = h_swish(out)
        out = self.AVpool(out)
        out = self.conv3(out)
        out = h_swish(out)
        return out


if __name__ == "__main__":

    # input = tf.keras.Input(shape=(224, 224, 3))
    model = MobileNetV3(type='large')
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
