import tensorflow as tf
from tensorflow import  keras


#https://github.com/philipperemy/deep-speaker/blob/master/triplet_loss.py

#https://zhuanlan.zhihu.com/p/40889858
#https://kexue.fm/archives/5743
def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_true = keras.backend.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)
    y_true = keras.backend.cast(y_true, 'int32') # 保证y_true的dtype=int32
    batch_idxs = keras.backend.arange(0, keras.backend.shape(y_true)[0])
    batch_idxs = keras.backend.expand_dims(batch_idxs, 1)
    idxs = keras.backend.concatenate([batch_idxs, y_true], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来
    y_true_pred = keras.backend.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin # 减去margin
    _Z = keras.backend.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数
    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    #logZ = keras.backend.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + keras.backend.log(1 - keras.backend.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)
    return - y_true_pred_margin * scale + logZ

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    #print(x1.shape,x2.shape)

    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1),axis=-1))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2),axis=-1))
    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(x1, x2),axis=-1)
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm+1e-9)
    #dot = keras.backend.squeeze(keras.backend.batch_dot(x1, x2, axes=1), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return cosin

def triplet_loss_cosine(y_true, y_pred, alpha=0.2):
    # y_true is not used. we respect this convention:
    # y_true.shape = (batch_size, embedding_size) [not used]
    # y_pred.shape = (batch_size, embedding_size)
    # EXAMPLE:
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # _____________________________________________________
    split = keras.backend.shape(y_pred)[-1] // 3
    anchor = y_pred[:,0:split]
    positive_ex = y_pred[:,split:2 * split]
    negative_ex = y_pred[:,2 * split:]
    # If the loss does not decrease below ALPHA then the model does not learn anything.
    # If all anchor = positive = negative (model outputs the same vector always).
    # Then sap = san = 1. and loss = max(alpha,0) = alpha.
    # On the contrary if anchor = positive = [1] and negative = [-1].
    # Then sap = 1 and san = -1. loss = max(-1-1+0.1,0) = max(-1.9, 0) = 0.
    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = keras.backend.maximum(san - sap + alpha, 0.0)
    total_loss = keras.backend.mean(loss)
    return total_loss

def triplet_loss_dist(y_true, y_pred, alpha=0.2):
    total_lenght = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:, :int(1 / 3 * total_lenght)], y_pred[:, int(1 / 3 * total_lenght):int(
        2 / 3 * total_lenght)], y_pred[:, int(2 / 3 * total_lenght):]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss
