import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, time
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)
import pandas as pd
from losses import sparse_amsoftmax_loss
from models import ResNetV1, ResNetV2
from DenseNet import DenseNet
from ecapa_tdnn import ECAPA_TDNN
from speech_featurizers import TFSpeechFeaturizer
import tensorflow_io as tfio
from tqdm import tqdm
from TFFbank import tf_fbank,tf_fbank2
import numpy as np
# https://github.com/abhimanyu1996/Face-Recognition-using-triplet-loss
# https://github.com/zukakosan/tripletNet
TFFeaturizer = TFSpeechFeaturizer(sample_rate=16000)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
                              tf.TensorSpec(shape=(None), dtype=tf.int32)], experimental_relax_shapes=False)
def train_step_amsoftmax(x, y):
    with tf.GradientTape() as tape:
        logits = model_amsoftmax(x, training=True)
        loss_value = sparse_amsoftmax_loss(y, logits)
    grads = tape.gradient(loss_value, model_amsoftmax.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_amsoftmax.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
                              tf.TensorSpec(shape=(None), dtype=tf.int32)], experimental_relax_shapes=False)
def test_step_amsoftmax(x, y):
    val_logits = model_amsoftmax(x, training=False)
    loss_value = sparse_amsoftmax_loss(y, val_logits)
    #print('\n', loss_value)
    #print('\n  [valid] val_loss:%.4f' % float(loss_value ) )
    val_acc_metric.update_state(y, val_logits)
    return loss_value


sparse_softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
                              tf.TensorSpec(shape=(None), dtype=tf.int32)], experimental_relax_shapes=False)
def train_step_softmax(x, y):
    with tf.GradientTape() as tape:
        logits = model_softmax(x, training=True)
        loss_value = sparse_softmax_loss(y, logits)
    grads = tape.gradient(loss_value, model_amsoftmax.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_amsoftmax.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
                              tf.TensorSpec(shape=(None), dtype=tf.int32)], experimental_relax_shapes=False)
def test_step_softmax(x, y):
    val_logits = model_softmax(x, training=False)
    loss_value = sparse_softmax_loss(y, val_logits)
    #print('\n',loss_value)
    #print('\n  [valid] val_loss:%.4f' % float(loss_value) )
    val_acc_metric.update_state(y, val_logits)
    return loss_value


def trainer(train_dataset, valid_dataset, config):
    # Iterate over the batches of the dataset.
    start_time = time.time()
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset), ncols=80, total=config["train_total_steps"],
                                                     desc='[train] ', file=sys.stdout,initial=config["init_step"]):
        step=step+config["init_step"]
        if config['activation'] == 'amsoftmax':
            loss_value = train_step_amsoftmax(x_batch_train, y_batch_train)
        else:
            loss_value = train_step_softmax(x_batch_train, y_batch_train)
        # Log every 200 batches.
        train_acc = train_acc_metric.result()
        samples = (step + 1) * batch_size
        if step % config["train_per_epoch"] == 0 and step > 1:
            print('\n   **** Finished %s epochs. ****' % (int(step / config["train_per_epoch"]) ) )
        if step % config["log_step_interval"] == 0 and step > 1:
            print("\n[train] step:%s,samples:%d,train_loss:%.4f,train_acc:%.4f,lr:%.4f" % (
            step, samples, float(loss_value), float(train_acc),float(lr_fn(step))))
            # tqdm.write("\n  step:%s,samples:%d,train_loss:%.4f,train_acc:%.4f"%(step,samples,float(loss_value),float(train_acc)) )
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
        valid_loss=0.0
        if (step % config["save_step_interval"] == 0 and step > 1) or (step==config["train_total_steps"]-1):
            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in tqdm(valid_dataset, ncols=80, total=config["valid_total_steps"],desc='[valid] ', file=sys.stdout):
                if config['activation'] == 'amsoftmax':
                    loss=test_step_amsoftmax(x_batch_val, y_batch_val)
                    valid_loss = valid_loss + loss
                else:
                    loss=test_step_softmax(x_batch_val, y_batch_val)
                    valid_loss = valid_loss + loss
            val_acc = val_acc_metric.result()
            print("\n[valid] val_loss:%.4f,val_acc: %.4f " % (float(valid_loss/config["valid_total_steps"]),float(val_acc)))
            # tqdm.write("\n  Validation acc: %.4f" % (float(val_acc)))
            if config['activation'] == 'amsoftmax':
                model_amsoftmax.save(filepath='saved/%s-amsoftmax-%s-%d-%.4f.h5' % (net, config["classes"], step, float(val_acc)))
            else:
                model_softmax.save(filepath='saved/%s-softmax-%s-%d-%.4f.h5' % (net, config["classes"], step, float(val_acc)))
            print("\n  Time taken: %.2fs" % (time.time() - start_time))
            # tqdm.write("\n  Time taken: %.2fs" % (time.time() - start_time))
            val_acc_metric.reset_states()

fb = np.load('data/fbank_80X257.npy')
fb = tf.convert_to_tensor(fb, dtype=tf.float32)
# @tf.autograph.do_not_convert
@tf.function
def parse_fun(item, label):
    sample_rate = 16000
    #288000=16000*18
    #240000=16000*15
    ts=240000
    a_raw_audio = tf.io.read_file(item)
    a_wave, a_sr = tf.audio.decode_wav(a_raw_audio, desired_channels=1, desired_samples=-1)
    if a_sr != sample_rate:
    #if tf.cond(tf.not_equal(a_sr, sample_rate), lambda: 0, lambda: 1):
        a_wave = tfio.audio.resample(a_wave, rate_in=tf.cast(a_sr, dtype=tf.int64),
                                     rate_out=tf.cast(sample_rate, dtype=tf.int64))
    shape0=tf.shape(a_wave)[0]
    if shape0>ts:
        idx=tf.random.uniform(shape=[],minval=0,maxval=shape0-ts,dtype=tf.int32)
        #a_mel=TFFeaturizer.tf_extract2(a_wave[idx:(idx+ts), 0])
        a_mel=tf_fbank2(signal=a_wave[idx:(idx+ts), 0],fb=fb)
    else:
        #a_mel = TFFeaturizer.tf_extract2(a_wave[:, 0])
        a_mel = tf_fbank2(signal=a_wave[:, 0],fb=fb)
    return a_mel, label



if __name__ == '__main__':
    os.makedirs('saved', exist_ok=True)
    parser = argparse.ArgumentParser(description=" speaker model train")
    parser.add_argument("--train_data", "-train", default='data/train_100_3180.csv', type=str, help="train data", )
    parser.add_argument("--valid_data", "-valid", default='data/valid_100_3180.csv', type=str, help="valid data", )
    parser.add_argument("--net", "-net", default='TDNN', type=str, help="ResNetV1,ResNetV2", )
    parser.add_argument("--activation", "-act", default='amsoftmax', type=str, help="amsoftmax,softmax", )
    parser.add_argument("--pretrained", "-p", default=None, type=str, help="pretrained embedding", )
    parser.add_argument("--pretrained_amsoftmax", "-pas", default=None, type=str, help="pretrained amsoftmax", )
    parser.add_argument("--pretrained_softmax", "-ps", default=None, type=str, help="pretrained softmax", )
    parser.add_argument("--batch_size", "-bs", default=24, type=int, help="batch_size", )
    parser.add_argument("--epochs", "-epochs", default=100, type=int, help="epochs", )
    parser.add_argument("--lr", "-lr", default=0.01, type=float, help="learning rate", )
    parser.add_argument("--init_step", "-init_step", default=0, type=int, help="init batch step", )
    args = parser.parse_args()
    print(args)
    config=vars(args)
    config["save_step_interval"]=10000 # save model every 5000 steps
    config["log_step_interval"]=1000
    #print(vars(args))
    net = args.net
    batch_size = args.batch_size
    epochs = args.epochs
    sample_rate = 16000
    # net="SpeakerResNet"
    train = pd.read_csv(args.train_data)
    valid = pd.read_csv(args.valid_data)

    train = train[['filepath', 'name', 'label']]
    valid = valid[['filepath', 'name', 'label']]
    config["classes"] = train.label.nunique()
    train_labels = train.label.values.tolist()  # [:100]
    train = train.filepath.values.tolist()  # [:100]
    valid_labels = valid.label.values.tolist()  # [:10]
    valid = valid.filepath.values.tolist()  # [:10]

    # print(train[:3])
    config["train_samples"] = len(train)
    # classes=len(train.label.unique())
    config["valid_samples"] = len(valid)
    print('train data size:%s' % config["train_samples"])
    print('valid data size:%s' % config["valid_samples"] )
    config["train_total_steps"] = int(epochs * config["train_samples"] / batch_size)
    config["train_per_epoch"] = int( config["train_samples"] / batch_size)
    config["valid_total_steps"] = int( config["valid_samples"] / batch_size)
    print('total_steps:', config["train_total_steps"] )
    config["decay_steps"] = int(epochs * 0.75 * config["train_samples"]  / batch_size)
    #config["init_step"] = 0
    print('decay_steps:', config["decay_steps"])
    # print(train.groupby('label')['filepath'].count(),valid.groupby('label')['filepath'].count())
    print(config)

    with tf.device("/cpu:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices((train, train_labels))
        train_dataset = train_dataset.shuffle(config["train_samples"]).repeat(epochs)
        train_dataset = train_dataset.map(parse_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None, None], []))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid, valid_labels))
        valid_dataset = valid_dataset.map(parse_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.padded_batch(batch_size, padded_shapes=([None, None], []))

    if net == "ResNetV2":
        model_softmax, model_amsoftmax, speark_emb_model = ResNetV2(input_shape=(None, 80), classes=config["classes"])
    elif net == "DenseNet121":
        model_softmax, model_amsoftmax, speark_emb_model = DenseNet(blocks=[6, 12, 24, 16], input_shape=(None, 80),
                                                                    classes=config["classes"], dropout=0.5)
    elif net == "TDNN":
        model_softmax, model_amsoftmax, speark_emb_model = ECAPA_TDNN(input_shape=(None, 80), classes=config["classes"],
                                                                      channels=512, embed_dim=256)
    else:
        model_softmax, model_amsoftmax, speark_emb_model = ResNetV1(input_shape=(None, 80), classes=config["classes"],
                                                                    dropout_rate=0.5)
    # speark_emb_model.summary()
    # model.summary()
    if args.pretrained is not None:
        speark_emb_model.load_weights(args.pretrained)

    if args.pretrained_softmax is not None:
        model_softmax.load_weights(args.pretrained_softmax)
    if args.pretrained_amsoftmax is not None:
        model_amsoftmax.load_weights(args.pretrained_amsoftmax)

    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=config["lr"], decay_steps=config["decay_steps"],end_learning_rate=0.0001,cycle=False)
    #lr_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config["lr"], decay_rate=0.9,decay_steps=config["decay_steps"])
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn, clipvalue=10.0,decay=0.0001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn, momentum=0.9, nesterov=True, clipvalue=10.0,decay=0.0001)
    #optimizer.get_config()

    trainer(train_dataset, valid_dataset, config)
