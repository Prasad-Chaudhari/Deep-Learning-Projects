import numpy as np
import tensorflow as tf
from math import pi, ceil, floor

def rotate_90(imgs, ys):
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    k = tf.placeholder(tf.int32)

    rotated_img = tf.image.rot90(X, k=k)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ans = np.empty(shape=(0, 64, 64, 3))
        ys_ = np.empty(shape=(0, 20))
        for i in [1, 3]:
            ans = np.concatenate(
                [ans, sess.run(rotated_img, feed_dict={X: imgs, k: i})], axis=0)
            ys_ = np.concatenate([ys_, ys], axis=0)
    return ans, ys_


def rotate_fine(X_imgs, start_angle, end_angle):
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    radian = tf.placeholder(tf.float32, shape=(X_imgs.shape[0]))
    tf_img = tf.contrib.image.rotate(X, radian)

    angles = np.random.uniform(start_angle, end_angle, size=(X_imgs.shape[0]))
    angles = angles * pi / 180

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tf_img, feed_dict={X: X_imgs, radian: angles})
    return result

def get_translate_parameters(index, IMAGE_SIZE):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.1], dtype=np.float32)
        size = np.array([64, ceil(0.9 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.9 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.1], dtype=np.float32)
        size = np.array(
            [IMAGE_SIZE, ceil(0.9 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.9) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.1, 0.0], dtype=np.float32)
        size = np.array(
            [ceil(0.9 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.9 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.1, 0.0], dtype=np.float32)
        size = np.array(
            [ceil(0.9 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.9) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), 64, 64, 3),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(
                i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0],
                         w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr


def central_scale_images(X_imgs, scales):
    tf.reset_default_graph()

    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)

    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([64, 64], dtype=np.int32)

    X = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
    # Define Tensorflow operation for all scales but only one base image at
    # a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = []
        for img in X_imgs:
            result.extend(sess.run(tf_img, feed_dict={
                          X: np.expand_dims(img, axis=0)}))
    return np.array(result)


def flip_images(X_imgs):
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    img_flip = tf.image.flip_left_right(X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        result = sess.run(img_flip, feed_dict={X: X_imgs})

    return result
