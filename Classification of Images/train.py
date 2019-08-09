# Import Libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random

from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops
from data_agumentation import (rotate_fine, rotate_90, translate_images,
                               central_scale_images, flip_images)
from network_functions import conv_net

parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--init")
parser.add_argument("--batch_size")
parser.add_argument("--epochs")
parser.add_argument("--save_dir")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--test")
parser.add_argument("--dataAugment")
args = parser.parse_args()

seed1 = 448000
seed2 = 25580

# seed1 = -1
# seed2 = -1
if seed1 == -1:
    seed1 = int(random.random() * 1000000)
if seed2 == -1:
    seed2 = int(random.random() * 1000000)
print("Seed1:" + str(seed1))
print("Seed2:" + str(seed2))

np.random.seed(seed1)
tf.random.set_random_seed(seed2)

# HyperParameters
batch_size = 64
if args.batch_size:
    batch_size = int(batch_size)
learning_rate = 0.0005
if args.lr:
    learning_rate = float(args.lr)
epochs = 20
if args.epochs:
    epochs = int(args.epochs)
n_classes = 20
BN = True  # Accounts for Batch Normalization
# Leaky ReLU activation is used everywhere
activation = 'l_relu'   # Type of Activation Function
dropout = 0.2
early_stopping = False
early_stop_parameter = 5  # Early Stopping parameter
rotate_90_bool = False
rotate_fine_bool = True
flip = True
translate = False
scale = True

directory = "./"
if args.save_dir:
    directory = directory + args.save_dir

train = pd.read_csv(args.train)
valid = pd.read_csv(args.val)
test = pd.read_csv(args.test)

# Import training data
train_x = train.drop(['label', 'id'], axis=1)
train_x = train_x.values.astype('float32')
train_y = train['label'].values

max_min = np.array([255.0 for x in range(train_x.shape[1])])

# Training Standardization between [0,1]
for i in range(train_x.shape[0]):
    train_x[i] = train_x[i] / max_min

train_y_OH = tf.one_hot(train_y, 20)

# Import validation data
valid_y = valid['label'].values
valid_x = valid.drop(['label', 'id'], axis=1)
valid_x = valid_x.values.astype('float32')

# Validation Standardization between [0,1]
for i in range(valid_x.shape[0]):
    valid_x[i] = valid_x[i] / max_min

valid_y_OH = tf.one_hot(valid_y, 20)

with tf.Session() as sess:
    train_y_OH = sess.run(train_y_OH)
    valid_y_OH = sess.run(valid_y_OH)

train_x_reshaped = []

for i in range(train_x.shape[0]):
    train_x_reshaped.append(train_x[i].reshape([64, 64, 3]))
train_x_reshaped = np.array(train_x_reshaped)

train_x = np.copy(train_x_reshaped)

train_x_orig = np.copy(train_x)
train_y_OH_orig = np.copy(train_y_OH)

valid_x_reshaped = []

for i in range(valid_x.shape[0]):
    valid_x_reshaped.append(valid_x[i].reshape([64, 64, 3]))
valid_x_reshaped = np.array(valid_x_reshaped)
valid_x = np.copy(valid_x_reshaped)

# Import test data
test_x = test.drop(['id'], axis=1)
test_x = test_x.values.astype('float32')

# test Standardization between [0,1]
for i in range(test_x.shape[0]):
    test_x[i] = test_x[i] / max_min

test_x_reshaped = []

for i in range(test_x.shape[0]):
    test_x_reshaped.append(test_x[i].reshape([64, 64, 3]))
test_x_reshaped = np.array(test_x_reshaped)
test_x = np.copy(test_x_reshaped)

if args.dataAugment is "1":

    # Rotation by 90 deg
    train_x_rotate_90 = np.empty(shape=(0, 64, 64, 3))
    train_y_OH_rotate_90 = np.empty(shape=(0, 20))
    if rotate_90_bool:
        train_x_rotate_90, train_y_OH_rotate_90 = rotate_90(
            train_x_orig, train_y_OH_orig)
    train_x = np.concatenate([train_x, train_x_rotate_90], axis=0)
    train_y_OH = np.concatenate([train_y_OH, train_y_OH_rotate_90], axis=0)

    # Rotation by finer angles
    train_x_rotate_fine = np.empty(shape=(0, 64, 64, 3))
    train_y_OH_rotate_fine = np.empty(shape=(0, 20))
    if rotate_fine_bool:
        train_x_rotate_fine = rotate_fine(train_x_orig, -80, 80)
        train_y_OH_rotate_fine = np.copy(train_y_OH_orig)
    train_x = np.concatenate([train_x, train_x_rotate_fine], axis=0)
    train_y_OH = np.concatenate([train_y_OH, train_y_OH_rotate_fine], axis=0)
    print(train_x.shape)
    print(train_y_OH.shape)

    train_x_rotate_fine = np.empty(shape=(0, 64, 64, 3))
    train_y_OH_rotate_fine = np.empty(shape=(0, 20))
    if rotate_fine_bool:
        train_x_rotate_fine = rotate_fine(train_x_orig, -30, 30)
        train_y_OH_rotate_fine = np.copy(train_y_OH_orig)
    train_x = np.concatenate([train_x, train_x_rotate_fine], axis=0)
    train_y_OH = np.concatenate([train_y_OH, train_y_OH_rotate_fine], axis=0)
    print(train_x.shape)
    print(train_y_OH.shape)

    # Flip the images left to right
    train_x_flip = np.empty(shape=(0, 64, 64, 3))
    train_y_OH_flip = np.empty(shape=(0, 20))
    if flip:
        train_x_flip = flip_images(train_x_orig)
        train_y_OH_flip = np.copy(train_y_OH_orig)
    train_x = np.concatenate([train_x, train_x_flip], 0)
    train_y_OH = np.concatenate([train_y_OH, train_y_OH_flip], 0)
    print(train_x.shape)
    print(train_y_OH.shape)

    # Scaling of image
    train_x_scale = np.empty(shape=(0, 64, 64, 3))
    train_y_OH_scale = np.empty(shape=(0, 20))
    if scale:
        train_x_scale = central_scale_images(train_x_orig, [0.9])
        train_y_OH_scale = np.copy(train_y_OH_orig)
    train_x = np.concatenate([train_x, train_x_scale], 0)
    train_y_OH = np.concatenate([train_y_OH, train_y_OH_scale], 0)
    print(train_x.shape)
    print(train_y_OH.shape)

    # Translate images
    if translate:
        IMAGE_SIZE = 64
        translated_imgs = translate_images(train_x_orig, IMAGE_SIZE)
        train_x = np.concatenate([train_x, translated_imgs], axis=0)
        train_y_trans = []
        for i in range(train_y_OH_orig.shape[0]):
            for j in range(4):
                train_y_trans.append(train_y_OH_orig[i])
        train_y_trans = np.array(train_y_trans)
        train_y_OH = np.concatenate([train_y_OH, train_y_trans])
        print(train_x.shape)
        print(train_y_OH.shape)

permutation = np.random.permutation(train_x.shape[0])

train_x = train_x[permutation]
train_y_OH = train_y_OH[permutation]

print("Permuated the whole data")

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
prob = tf.placeholder_with_default(0.0, shape=())

# Functions to be used for initialisations
w_init_functions = {
    "2": tf.contrib.layers.variance_scaling_initializer,
    "1": tf.contrib.layers.xavier_initializer
}

init_number = args.init

weights = {
    'wc1': tf.get_variable('W0', shape=(7, 7, 3, 128), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wc2': tf.get_variable('W1', shape=(7, 7, 128, 192), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wc3': tf.get_variable('W2', shape=(5, 5, 192, 256), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wc4': tf.get_variable('W3', shape=(5, 5, 256, 384), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wc5': tf.get_variable('W4', shape=(3, 3, 384, 512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wc6': tf.get_variable('W5', shape=(3, 3, 512, 512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wd1': tf.get_variable('W6', shape=(6 * 6 * 512, 4096), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'wd2': tf.get_variable('W7', shape=(4096, 512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'out': tf.get_variable('W8', shape=(512, n_classes), initializer=w_init_functions[init_number](dtype=tf.float32)),
}

biases = {
    'bc1': tf.get_variable('B0', shape=(128), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bc2': tf.get_variable('B1', shape=(192), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bc3': tf.get_variable('B2', shape=(256), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bc4': tf.get_variable('B3', shape=(384), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bc5': tf.get_variable('B4', shape=(512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bc6': tf.get_variable('B5', shape=(512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bd1': tf.get_variable('B6', shape=(4096), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'bd2': tf.get_variable('B7', shape=(512), initializer=w_init_functions[init_number](dtype=tf.float32)),
    'out': tf.get_variable('B8', shape=(n_classes), initializer=w_init_functions[init_number](dtype=tf.float32)),
}

acti = conv_net(x, weights, biases, n_classes, activation, prob)
pred = acti[1]

y_pred = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)

cost = tf.reduce_mean(y_pred)

norms = []

for weight in weights.values():
    if "6" in weight.name or "7" in weight.name or "8" in weight.name:
        norms.append(tf.nn.l2_loss(weight))

for weight in biases.values():
    if "6" in weight.name or "7" in weight.name or "8" in weight.name:
        norms.append(tf.nn.l2_loss(weight))

loss_L2 = tf.add_n(norms) * .05

cost = cost

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost + loss_L2)

correct_pred = tf.equal(tf.math.argmax(input=pred, axis=1),
                        tf.argmax(input=y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=epochs, save_relative_paths=True)

with tf.Session() as sess:
    sess.run(init)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    min_train_loss = 20000000000
    min_valid_loss = 20000000000

    min_train_index = -1
    min_valid_index = -1
    for epoch in range(epochs):
        train_loss.append(0)
        train_acc.append(0)

        for batch in range(train_x.shape[0] // batch_size):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, train_x.shape[0])

            batch_x = train_x[start:end]
            batch_y = train_y_OH[start:end]

            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            t_loss, t_acc = sess.run([cost, accuracy], feed_dict={
                                     x: batch_x, y: batch_y, prob: dropout})

            train_loss[-1] += t_loss * (end - start) / train_x.shape[0]
            train_acc[-1] += t_acc * (end - start) / train_x.shape[0]

        print("Iter " + str(epoch) + ", Loss= " +
              "{:.6f}".format(train_loss[-1]) + ", Training Accuracy= " +
              "{:.5f}".format(100 * train_acc[-1]))

        v_loss, v_acc = sess.run([cost, accuracy], feed_dict={
                                 x: valid_x, y: valid_y_OH})

        valid_loss.append(v_loss)
        valid_acc.append(v_acc)

        print("Validation: Loss", "{:.5f}".format(
            valid_loss[-1]), "Accuracy:", "{:.5f}".format(valid_acc[-1] * 100))

        saved_path = saver.save(
            sess, directory + 'models/DL-PA2', global_step=epoch)

        print()

        if min_train_loss > train_loss[-1]:
            min_train_loss = train_loss[-1]
            min_train_index = epoch

        if min_valid_loss > valid_loss[-1]:
            min_valid_loss = valid_loss[-1]
            min_valid_index = epoch

        if early_stopping:
            if epoch - min_valid_index >= early_stop_parameter:
                break
            if epoch - min_train_index >= early_stop_parameter:
                break

        y_pred_test = sess.run(pred, feed_dict={x: test_x})

        test_labels = []

        for i in range(y_pred_test.shape[0]):
            test_labels.append(np.argmax(y_pred_test[i]))
        id_list = [x for x in range(len(test_labels))]

        df = pd.DataFrame(data={"id": id_list, "label": test_labels})
        df.to_csv(directory + "CE15B100_ME15B130_" +
                  str(epoch) + ".csv", sep=',', index=False)


logs = pd.DataFrame(data={'train_loss': train_loss, "train_acc": train_acc,
                          "valid_loss": valid_loss, "valid_acc": valid_acc})
logs.to_csv(directory + "logs.csv", sep=',', index=False)


guided_bp = None
while guided_bp is None or ("Y" is not guided_bp and "N" is not guided_bp):
    guided_bp = input(
        "Do you want to run guided back propagation on any 10 iamges from training data set? (Y/N)")
    if ("Y" is not guided_bp and "N" is not guided_bp):
        print("Plz add proper valid input")

if guided_bp is "Y":
    @ops.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

    best_iter = int(
        input("Plz insert Epoch number which you think would give best results:"))
    permutation = np.random.permutation(6 * 6 * 512)

    batch_size = 32
    with tf.Session() as sess:
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            saver.restore(sess, directory + "models/DL-PA2-" + str(best_iter))

        train_x_imgs = []
        img_index = np.random.permutation(train_x_orig.shape[0])[:10]
        features = []
        for index in img_index:
            firing = sess.run(acti[0], feed_dict={x: train_x[index][None]})

            max_index = np.argmax(firing)

            f = sess.run(tf.gradients(acti[0][:, max_index], x), feed_dict={
                         x: train_x[index][None]})
            f[0] = (f[0] - f[0].min()) / (f[0].max() - f[0].min())
            features.append(f[0])

    plt.figure(figsize=(64, 64))

    for i in range(len(features)):
        plt.figure(i + 1)
        plt.imshow(np.reshape(features[i], [64, 64, 3]), interpolation='none')
        plt.title('Image: {}'.format(i + 1))
        plt.savefig(directory + 'guided_back_prop_' +
                    str(img_index[i]) + '.png')
        plt.clf()

    for i in range(len(features)):
        plt.figure(i + 1)
        plt.imshow(np.reshape(train_x[img_index[i]], [
                   64, 64, 3]), interpolation='none')
        plt.title('Image: {}'.format(i + 1))
        plt.savefig(directory + 'original_image_' + str(img_index[i]) + '.png')
        plt.clf()

with tf.Session() as sess:
	saver.restore(sess, directory + "models/DL-PA2-" + str(best_iter))
	y_pred_test = sess.run(pred, feed_dict={x: test_x})

        test_labels = []

        for i in range(y_pred_test.shape[0]):
            test_labels.append(np.argmax(y_pred_test[i]))
        id_list = [x for x in range(len(test_labels))]

        df = pd.DataFrame(data={"id": id_list, "label": test_labels})
        df.to_csv(directory + "CE15B100_ME15B130_" +
                  str(epoch) + ".csv", sep=',', index=False)