import tensorflow as tf

dropout = .2


def conv2d(x, W, b, activation, strides=1, padding='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)

    if activation == 'relu':
        return tf.nn.relu(x)
    else:
        return tf.nn.leaky_relu(x, alpha=.05)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def BN(x):
    return tf.layers.batch_normalization(x, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                         beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(),
                                         moving_variance_initializer=tf.ones_initializer(), training=True, trainable=True, renorm=False, renorm_momentum=0.99)


def conv_net(x, W, b, activation, n_classes, prob):
    # Layer 1
    conv1 = conv2d(x, W['wc1'], b['bc1'], activation, 1)
    conv1 = BN(conv1)
    conv2 = conv2d(conv1, W['wc2'], b['bc2'], activation, 1)
    conv2 = BN(conv2)
    pool3 = maxpool2d(conv2, 2)
    # Layer 2
    conv4 = conv2d(pool3, W['wc3'], b['bc3'], activation, 1)
    conv4 = BN(conv4)
    conv5 = conv2d(conv4, W['wc4'], b['bc4'], activation, 1)
    conv5 = BN(conv5)
    pool6 = maxpool2d(conv5, 2)
    # Layer 3
    conv7 = conv2d(pool6, W['wc5'], b['bc5'], activation, 1)
    conv7 = BN(conv7)
    pool8 = maxpool2d(conv7, 2)
    conv9 = conv2d(pool8, W['wc6'], b['bc6'], activation, 1, padding='VALID')
    conv9 = BN(conv9)
    conv9_reshape = tf.reshape(conv9, shape=[-1, 6 * 6 * 512])

    fc_layer1 = tf.add(tf.matmul(conv9_reshape, W['wd1']), b['bd1'])

    if activation == 'relu':
        fc_layer1_relu = tf.nn.relu(fc_layer1)
    else:
        fc_layer1_relu = tf.nn.leaky_relu(fc_layer1, alpha=.05)

    BN1 = BN(fc_layer1_relu)

    if dropout != 1:
        BN1 = tf.nn.dropout(BN1, keep_prob=1 - prob)

    fc_layer2 = tf.add(tf.matmul(BN1, W['wd2']), b['bd2'])

    if activation == 'relu':
        fc_layer2_relu = tf.nn.relu(fc_layer2)
    else:
        fc_layer2_relu = tf.nn.leaky_relu(fc_layer2, alpha=.05)

    BN2 = BN(fc_layer2_relu)

    if dropout != 1:
        BN2 = tf.nn.dropout(BN2, keep_prob=1 - prob)

    fc_out = tf.add(tf.matmul(BN2, W['out']), b['out'])
    return [conv9_reshape, fc_out]
