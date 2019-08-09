`# Import Libraries
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import argparse

tf.reset_default_graph()
tf.random.set_random_seed(213123)
sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--init")
parser.add_argument("--batch_size")
parser.add_argument("--epochs")
parser.add_argument("--save_dir")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--dropout_prob")
parser.add_argument("--decode_method")
parser.add_argument("--beam_width")
parser.add_argument("--direction")

args = parser.parse_args()

directory = args.save_dir

train = pd.read_csv(args.train + 'train.csv')
valid = pd.read_csv(args.val + 'valid.csv')
test = pd.read_csv(directory + "test_final.csv")

train_x = train['ENG'].tolist()
train_y = train['HIN'].tolist()

for i in range(len(train_x)):
    train_x[i] = train_x[i].split(' ')
    train_y[i] = train_y[i].split(' ')

if not os.path.isdir(directory + "DL_PA3"):
    os.mkdir(directory + "DL_PA3")

exp_num = 0
while True:
    exists = os.path.isdir(directory + "DL_PA3/" + str(exp_num))
    if not exists:
        os.mkdir(directory + "DL_PA3/" + str(exp_num))
        directory = directory + "DL_PA3/" + str(exp_num)
        break
    else:
        exp_num += 1

batch_size = 128
if args.batch_size:
    batch_size = int(batch_size)

learning_rate = 0.002
if args.lr:
    learning_rate = float(args.lr)

epochs = 40
if args.epochs:
    epochs = int(args.epochs)

dropout_rate_out = .1
if args.dropout_prob:
    dropout_rate_out = float(args.dropout_prob)

beam_width = 3
if args.beam_width:
    beam_width = int(args.beam_width)

if beam_width == 0:
    b_or_g_str = "GREEDY"
    beam_search = False
else:
    b_or_g_str = "BEAM_SEARCH"
    beam_search = True

max_length = 70
encoder_input_embed_size = 256
encoder_rnn_size = 256
decoder_input_embed_size = 256
decoder_rnn_size = 512
decoder_rnn_layers = 2

clipping = 0

early_stopping = False
stop_by_loss = True
early_stopping_parameter = 5

use_attention = True

direction = "bidirectional"
if direction == "unidirectional":
    decoder_input_embed_size = 256

init = args.init


def indexed_array(data, vocab):
    data_indexed = []
    for d in data:
        d_indexed = []
        for char in d:
            if vocab.index(char) != -1:
                d_indexed.append(vocab.index(char))
            else:
                d_indexed.append(vocab.index("<UNKNOWN>"))
        data_indexed.append(d_indexed)
    return data_indexed


def pad_till_max_length(data, max_length, cond):
    if cond == 1:
        for d in data:
            d.pop(0)
    for d in data:
        l = len(d)
        while l < max_length:
            d.append("<PAD>")
            l += 1
    return data


eng_vocab = ["<PAD>", "<UNKNOWN>", "<STOP>"]
train_x_lens = []

for x in train_x:
    for char in x:
        eng_vocab.append(char)
    x.append("<STOP>")
    train_x_lens.append(max_length)

eng_vocab = list(set(eng_vocab))
input_vocab_size = len(eng_vocab)

train_x = np.array(indexed_array(
    pad_till_max_length(train_x, max_length, 0), eng_vocab))
train_x_lens = np.array(train_x_lens)

valid_x = valid["ENG"].tolist()
valid_x_lens = []
for i in range(len(valid_x)):
    valid_x[i] = valid_x[i].split(' ')
    valid_x[i].append("<STOP>")
    valid_x_lens.append(max_length)

valid_x = np.array(indexed_array(
    pad_till_max_length(valid_x, max_length, 0), eng_vocab))
valid_x_lens = np.array(valid_x_lens)

test_x = test["ENG"].tolist()
test_x_lens = []
for i in range(len(test_x)):
    test_x[i] = test_x[i].split(' ')
    test_x[i].append("<STOP>")
    test_x_lens.append(max_length)

test_x = np.array(indexed_array(
    pad_till_max_length(test_x, max_length, 0), eng_vocab))
test_x_lens = np.array(test_x_lens)

hindi_vocab = ["<PAD>", "<START>", "<STOP>", "<UNKNOWN>"]

train_y_lens = []
for y in train_y:
    for char in y:
        hindi_vocab.append(char)
    y.append("<STOP>")
    y.insert(0, "<START>")
    train_y_lens.append(max_length)

hindi_vocab = list(set(hindi_vocab))
output_vocab_size = len(hindi_vocab)

train_y_decoder_input = np.array(indexed_array(
    pad_till_max_length(train_y, max_length, 0), hindi_vocab))
train_y_decoder_output = np.array(indexed_array(
    pad_till_max_length(train_y, max_length, 1), hindi_vocab))

valid_y = valid["HIN"].tolist()
valid_y_lens = []
for i in range(len(valid_y)):
    valid_y[i] = valid_y[i].split(' ')
    valid_y[i].append("<STOP>")
    valid_y[i].insert(0, "<START>")
    valid_y_lens.append(max_length)

valid_y_decoder_input = np.array(indexed_array(
    pad_till_max_length(valid_y, max_length, 0), hindi_vocab))
valid_y_decoder_output = np.array(indexed_array(
    pad_till_max_length(valid_y, max_length, 1), hindi_vocab))
print(hindi_vocab.index("<PAD>"))

x = tf.placeholder(tf.int32, shape=[None, None])
x_lengths = tf.placeholder(tf.int32, shape=[None])
dropout_rate_out_tf = tf.placeholder(tf.float32)

print(x_lengths)
if init == 1:
    init_f = tf.contrib.layers.xavier_initializer
else:
    init_f = tf.contrib.layers.variance_scaling_initializer

encoder_input_embedding_mat = tf.get_variable("encoder_input_embedding_weight", [
                                              input_vocab_size, encoder_input_embed_size], initializer=init_f(dtype=tf.float32))

encoder_embedding = tf.nn.embedding_lookup(
    encoder_input_embedding_mat, x, name="encoder_input")

if direction == 'bidirectional':
    encoder_LSTMCell_fw = tf.nn.rnn_cell.LSTMCell(encoder_rnn_size)
    encoder_LSTMCell_fw = tf.nn.rnn_cell.DropoutWrapper(
        encoder_LSTMCell_fw, output_keep_prob=1 - dropout_rate_out_tf)

    encoder_LSTMCell_bw = tf.nn.rnn_cell.LSTMCell(encoder_rnn_size)
    encoder_LSTMCelL_bw = tf.nn.rnn_cell.DropoutWrapper(
        encoder_LSTMCell_bw, output_keep_prob=1 - dropout_rate_out_tf)

    (encoder_output_fw, encoder_output_bw), (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_LSTMCell_fw, cell_bw=encoder_LSTMCell_bw, inputs=encoder_embedding, sequence_length=x_lengths, dtype=tf.float32)
    encoder_output = tf.concat((encoder_output_fw, encoder_output_bw), -1)

    state_c = tf.concat((encoder_state_fw.c, encoder_state_bw.c), -1)
    state_h = tf.concat((encoder_state_fw.h, encoder_state_bw.h), -1)
    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h)
    encoder_final_state = tuple([lstm_state for _ in range(decoder_rnn_layers)])
else:
    encoder_LSTMCell = tf.nn.rnn_cell.LSTMCell(encoder_rnn_size)
    encoder_LSTMCell = tf.nn.rnn_cell.DropoutWrapper(
        encoder_LSTMCell, output_keep_prob=1 - dropout_rate_out_tf)

    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=encoder_LSTMCell, inputs=encoder_embedding, sequence_length=x_lengths, dtype=tf.float32)
    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(
        c=encoder_state.c, h=encoder_state.h)
    encoder_final_state = tuple([lstm_state for _ in range(decoder_rnn_layers)])

y_true_input = tf.placeholder(tf.int32, shape=[None, None])
y_true_output = tf.placeholder(tf.int32, shape=[None, None])
y_lens = tf.placeholder(tf.int32, shape=[None])
b_or_g_tf = tf.placeholder(tf.string)
b_or_g = tf.constant("BEAM_SEARCH")

decoder_input_embedding_mat = tf.get_variable("decoder_input_embedding_weight", [
                                              output_vocab_size, decoder_input_embed_size], initializer=init_f(dtype=tf.float32))

decoder_embedding = tf.nn.embedding_lookup(
    decoder_input_embedding_mat, y_true_input, name="encoder_input")

decoder_LSTMCell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(decoder_rnn_size) for _ in range(decoder_rnn_layers)])
decoder_LSTMCell = tf.nn.rnn_cell.DropoutWrapper(
    decoder_LSTMCell, output_keep_prob=1 - dropout_rate_out_tf)

decoder_output_to_hindi_vocab = tf.layers.Dense(
    output_vocab_size, name="decoder_output_to_hindi_vocab")

batch_sizzzz = tf.shape(x)[0]


def beam_search_edits():
    print("Using beam search")
    encoder_output_ = tf.contrib.seq2seq.tile_batch(
        encoder_output, multiplier=beam_width)
    encoder_final_state_ = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    x_lengths_ = tf.contrib.seq2seq.tile_batch(x_lengths, multiplier=beam_width)
    batch_sizzzz_ = batch_sizzzz * beam_width
    return encoder_output_, encoder_final_state_, x_lengths_, batch_sizzzz_


encoder_output_, encoder_final_state_, x_lengths_, batch_sizzzz_ = tf.cond(tf.math.equal(b_or_g_tf, b_or_g),
                                                                           lambda: beam_search_edits(),
                                                                           lambda: (encoder_output, encoder_final_state, x_lengths, batch_sizzzz))

if use_attention:
    print("Using attention mechanism")
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=decoder_rnn_size, memory=encoder_output_, memory_sequence_length=x_lengths_)
    decoder_LSTMCell = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_LSTMCell, attention_mechanism=attention_mechanism, attention_layer_size=None, output_attention=False, alignment_history=True)
    decoder_initial_state = decoder_LSTMCell.zero_state(
        batch_sizzzz_, tf.float32).clone(cell_state=encoder_final_state_)
else:
    decoder_initial_state = encoder_final_state_


# TRAINING
training_helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_embedding, y_lens, time_major=False)
training_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_LSTMCell, training_helper, decoder_initial_state, output_layer=decoder_output_to_hindi_vocab)
training_decoder_outputs, training_decode_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
    training_decoder, output_time_major=False, maximum_iterations=max_length, swap_memory=False, impute_finished=True)
training_logits = training_decoder_outputs.rnn_output

softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y_true_output, logits=training_logits)

# l2_loss = 0.01 * tf.add_n([tf.nn.l2_loss(v)
#                            for v in tf.trainable_variables() if 'bias' not in v.name])
# loss = tf.reduce_mean(softmax_cross_entropy) + l2_loss
loss = tf.reduce_mean(softmax_cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate)

if clipping > 0:
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, trainable_variables), clipping)
    optimizer = optimizer.apply_gradients(zip(grads, trainable_variables))
else:
    optimizer = optimizer.minimize(loss)

if beam_search:
    y_start_tokens = tf.fill([batch_sizzzz], hindi_vocab.index("<START>"))
    y_end_token = hindi_vocab.index("<STOP>")

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        decoder_LSTMCell, decoder_input_embedding_mat, y_start_tokens, y_end_token, decoder_initial_state, beam_width, decoder_output_to_hindi_vocab)
    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, output_time_major=False, maximum_iterations=max_length, swap_memory=False, impute_finished=False)
    inference_sample_id = inference_decoder_outputs.predicted_ids
    inference_scores = inference_decoder_outputs.beam_search_decoder_output.scores
else:
    y_start_tokens = tf.fill([batch_sizzzz_], hindi_vocab.index("<START>"))
    y_end_token = hindi_vocab.index("<STOP>")

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        decoder_input_embedding_mat, y_start_tokens, y_end_token)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_LSTMCell, inference_helper, decoder_initial_state, output_layer=decoder_output_to_hindi_vocab)
    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, output_time_major=False, maximum_iterations=max_length, swap_memory=False, impute_finished=False)
    inference_sample_id = inference_decoder_outputs.sample_id

train_data_size = train_x.shape[0]
valid_data_size = valid_x.shape[0]

valid_ys = valid["HIN"].tolist()
train_ys = train["HIN"].tolist()

saver = tf.train.Saver(max_to_keep=epochs, save_relative_paths=True)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    train_loss_track_2 = []
    valid_loss_track_2 = []
    train_acc_track_2 = []
    valid_acc_track_2 = []

    for epoch in range(0, 0 + epochs):

        print(epoch)
        training_loss = 0

        for batch in range((train_data_size // batch_size) + 1):

            start = batch * batch_size
            end = min((batch + 1) * batch_size, train_data_size)
            batch_x = train_x[start:end]
            batch_x_lens = train_x_lens[start:end]

            batch_y_input = train_y_decoder_input[start:end]
            batch_y_output = train_y_decoder_output[start:end]
            batch_y_lens = train_y_lens[start: end]

            t_loss, _ = sess.run([loss, optimizer], feed_dict={x: batch_x,
                                                               x_lengths: batch_x_lens,
                                                               y_true_input: batch_y_input,
                                                               y_true_output: batch_y_output,
                                                               y_lens: batch_y_lens,
                                                               dropout_rate_out_tf: dropout_rate_out,
                                                               b_or_g_tf: "asdas"})

            training_loss += t_loss * (end - start) / train_data_size
        train_loss_track_2.append(training_loss)

        train_acc = 0
        train_pred = []

        for batch in range((train_data_size // batch_size) + 1):

            start = batch * batch_size
            end = min(start + batch_size, train_data_size)
            batch_x = train_x[start:end]
            batch_x_lens = train_x_lens[start:end]

            t_sample_ids = sess.run(inference_sample_id, feed_dict={x: batch_x,
                                                                    x_lengths: batch_x_lens,
                                                                    dropout_rate_out_tf: 0,
                                                                    b_or_g_tf: b_or_g_str})
            if beam_search:
                t_sample_ids = np.transpose(t_sample_ids, axes=[0, 2, 1])
                for i in range(t_sample_ids.shape[0]):
                    s = ""
                    for j in range(t_sample_ids.shape[2]):
                        if t_sample_ids[i][0][j] == hindi_vocab.index("<STOP>"):
                            break
                        else:
                            s += hindi_vocab[t_sample_ids[i][0][j]] + " "
                    train_pred.append(s[:-1])
            else:
                for i in range(t_sample_ids.shape[0]):
                    s = ""
                    for j in range(t_sample_ids.shape[1]):
                        if t_sample_ids[i][j] == hindi_vocab.index("<STOP>"):
                            break
                        else:
                            s += hindi_vocab[t_sample_ids[i][j]] + " "
                    train_pred.append(s[:-1])

        for i in range(train_data_size):
            if train_pred[i] == train_ys[i]:
                train_acc += 1

        print("training loss", round(training_loss, 5),
              "training acc", round(train_acc / train_data_size, 5))

        train_acc_track_2.append(train_acc * 100 / train_data_size)

        validing_loss = 0

        for batch in range((valid_data_size // batch_size) + 1):

            start = batch * batch_size
            end = min((batch + 1) * batch_size, valid_data_size)
            batch_x = valid_x[start:end]
            batch_x_lens = valid_x_lens[start:end]

            batch_y_input = valid_y_decoder_input[start:end]
            batch_y_output = valid_y_decoder_output[start:end]
            batch_y_lens = valid_y_lens[start: end]
            t_loss = sess.run(loss, feed_dict={x: batch_x,
                                               x_lengths: batch_x_lens,
                                               y_true_input: batch_y_input,
                                               y_true_output: batch_y_output,
                                               y_lens: batch_y_lens,
                                               dropout_rate_out_tf: dropout_rate_out,
                                               b_or_g_tf: "asdas"})

            validing_loss += t_loss * (end - start) / valid_data_size

        valid_loss_track_2.append(validing_loss)
        valid_acc = 0
        valid_pred = []

        for batch in range((valid_data_size // batch_size) + 1):

            start = batch * batch_size
            end = min(start + batch_size, valid_data_size)
            batch_x = valid_x[start:end]
            batch_x_lens = valid_x_lens[start:end]

            t_sample_ids = sess.run(inference_sample_id, feed_dict={x: batch_x,
                                                                    x_lengths: batch_x_lens,
                                                                    dropout_rate_out_tf: 0,
                                                                    b_or_g_tf: b_or_g_str})
            if beam_search:
                t_sample_ids = np.transpose(t_sample_ids, axes=[0, 2, 1])
                for i in range(t_sample_ids.shape[0]):
                    s = ""
                    for j in range(t_sample_ids.shape[2]):
                        if t_sample_ids[i][0][j] == hindi_vocab.index("<STOP>"):
                            break
                        else:
                            s += hindi_vocab[t_sample_ids[i][0][j]] + " "
                    valid_pred.append(s[:-1])
            else:
                for i in range(t_sample_ids.shape[0]):
                    s = ""
                    for j in range(t_sample_ids.shape[1]):
                        if t_sample_ids[i][j] == hindi_vocab.index("<STOP>"):
                            break
                        else:
                            s += hindi_vocab[t_sample_ids[i][j]] + " "
                    valid_pred.append(s[:-1])

        for i in range(valid_data_size):
            if valid_pred[i] == valid_ys[i]:
                valid_acc += 1

        valid_acc_track_2.append(valid_acc * 100 / valid_data_size)
        print("validing loss", round(validing_loss, 5),
              "validing acc", round(valid_acc / valid_data_size, 5))

        test_pred = []

        test_sample_ids = sess.run(inference_sample_id, feed_dict={x: test_x,
                                                                   x_lengths: test_x_lens,
                                                                   dropout_rate_out_tf: 0,
                                                                   b_or_g_tf: b_or_g_str})
        test_sample_ids = np.transpose(test_sample_ids, axes=[0, 2, 1])
        for i in range(test_sample_ids.shape[0]):
            s = ""
            for j in range(test_sample_ids.shape[2]):
                if test_sample_ids[i][0][j] == hindi_vocab.index("<STOP>"):
                    break
                else:
                    s += hindi_vocab[test_sample_ids[i][0][j]] + " "
            test_pred.append(s[:-1])

        df = pd.DataFrame(data={'id': np.arange(
            test_x.shape[0]).tolist(), 'HIN': test_pred})
        df.to_csv(directory + "/CE15B100_ME15B130_" +
                  str(epoch) + ".csv", sep=',', index=False)
        saved_path = saver.save(
            sess, directory + '/models/DL-PA3', global_step=epoch)
