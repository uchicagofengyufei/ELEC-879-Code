import os
import math
import json

import argparse
import numpy as np
import tensorflow as tf

from time import time
from functools import partial


from load_sst import DataLoader
from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from utils import find_trainable_variables, convert_gradient_to_tensor, shape_list, assign_to_gpu, average_grads

def decay_learning_rate(lr):

    global_step = tf.Variable(0, trainable=False)

    lr = tf.train.exponential_decay(
        learning_rate=lr,
        global_step=global_step,
        decay_steps=720,
        decay_rate=0.96)
    lr = tf.maximum(lr, 1e-6)
    return lr, global_step

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

#def dropout(x, pdrop, train):
#    return tf.layers.dropout(x, pdrop, training=train)


def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, 0.1, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, 0.1, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns['gelu']
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, 0.1, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, 12, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b


def model(X, M, Y, Is_train, data_params):


    n_vocab = data_params['n_vocab']
    n_special = data_params['n_special']
    max_word = data_params['max_word']
    clf_token = data_params['clf_token']
    clf_pdrop = 0.1
    with tf.variable_scope('transformer', reuse=False):
        we = tf.get_variable("we", [n_vocab+n_special+max_word, 768], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, 0.1, True)

        X = tf.reshape(X, [-1, max_word, 2])  # (batch * 1sent, 161, 2) == (8,161,2)
        M = tf.reshape(M, [-1, max_word])

        h = embed(X, we)  # (8, 161, 768)

        for layer in range(12):
            h = block(h, 'h%d'%layer, train=True, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, 768])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, 768])         # h:(8,161,768)  clf_h:(1288, 768)
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)    # last: clf_token
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*max_word+pool_idx)                 # (8,768)
        clf_h = tf.reshape(clf_h, [-1, 1, 768])  # (4, 2, 768)
        if True and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, 768])  # 8*1*768
        clf_logits = clf(clf_h, 2, train=True)  # 1 sent-->3 classes
        clf_logits = tf.reshape(clf_logits, [-1, 2])
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, tf.reduce_mean(clf_losses), tf.reduce_mean(lm_losses)

def convert_to_embedding(sess, data_iterator):
    train_set = []
    train_label = []
    val_set = []
    val_label = []
    test_set = []
    test_label = []


def run_epoch(sess, logits, clf_losses, data_iterator, phase, batch_size=16, train_op=tf.constant(0)):


    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()

    for x_batch, y_batch, m_batch in data_iterator.sampled_batch(batch_size=batch_size, phase=phase):

        batch_pred, batch_loss, _ = sess.run([logits, clf_losses, train_op],
                                              feed_dict={X: x_batch,
                                              M: m_batch,
                                              Y: y_batch, train_flag: phase == 'train'})

        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss * n_sample

        t_correct += np.sum(np.argmax(batch_pred, axis=1) == y_batch)



    print("{} Loss: {:.4f},  Accuarcy: {:.2f}%, {:.2f} Seconds Used:".
          format(phase, t_loss / n_all, 100 * t_correct / n_all, time()-t0))

    return t_loss/n_all




if __name__ == '__main__':

    data_iterator = DataLoader()
    max_word = data_iterator.max_word

    X = tf.placeholder(tf.int32, [None, max_word, 2])
    M = tf.placeholder(tf.float32, [None, max_word])

    Y = tf.placeholder(tf.int32, [None])
    train_flag = tf.placeholder_with_default(True, shape=())
    print(data_iterator.max_word)
    dp = {'max_word': data_iterator.max_word, 'n_vocab': data_iterator.n_vocab,
          'n_special': 3, 'clf_token': data_iterator.encoder['_classify_']}
    logits, clf_losses, lm_losses = model(X, M, Y, train_flag, data_params=dp)

    #lr, global_step = decay_learning_rate(6.25e-5)
    optimizer = tf.train.AdamOptimizer(6.25e-5)
    train_op = optimizer.minimize(clf_losses + 0.5*lm_losses) #, global_step=global_step)

    params = find_trainable_variables('transformer')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    shapes = json.load(open('./pretrain/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('./pretrain/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:max_word]
    init_params[0] = np.concatenate([init_params[1], (np.random.randn(3, 768)*0.02).astype(np.float32), init_params[0]], 0)
    del init_params[1]

    n_transfer = 1 + 12 * 12
    sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])


    for i in range(0, 200):
        print('Epoch {}...'.format(i))
        run_epoch(sess, logits, clf_losses, data_iterator, 'train', batch_size=16, train_op=train_op)
        run_epoch(sess, logits, clf_losses, data_iterator, 'validation', batch_size=16)
        run_epoch(sess, logits, clf_losses, data_iterator, 'test', batch_size=16)
