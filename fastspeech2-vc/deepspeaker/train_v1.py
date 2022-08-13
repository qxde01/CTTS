import os,argparse
import sys

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
import numpy as np
#from utils import TFMel
#from ResNet import SpeakerResNetV2
from losses import triplet_loss_cosine,sparse_amsoftmax_loss
from models import ResNetV1,ResNetV2
from DenseNet import DenseNet
#from ResNet import ResNet50V2
from speech_featurizers import TFSpeechFeaturizer
import tensorflow_io as tfio
TFFeaturizer=TFSpeechFeaturizer(sample_rate=16000)

class ModelCheck(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """
    def __init__(self, filepath,steps=1):
        super(ModelCheck, self).__init__()
        self.filepath =str(filepath)
        self.steps=steps
        #self.best = np.Inf
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
    def set_model(self, model):
        self.model = model
    def _save_model(self, epoch, logs):
        logs = logs or {}
        self.model.save(self.filepath, overwrite=True, options=self._options)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get('val_loss')
        mname = '_%.4f-%03d.h5' % (current, epoch)
        if epoch%self.steps==0  :
            print(epoch, current, self.filepath+mname)
            self.model.save(self.filepath+mname,include_optimizer=False)

#@tf.autograph.do_not_convert
def parse_fun(item,label):
    sample_rate=16000
    a_raw_audio = tf.io.read_file(item)
    a_wave, a_sr = tf.audio.decode_wav(a_raw_audio, desired_channels=1, desired_samples=-1)
    if a_sr != sample_rate:
        a_wave = tfio.audio.resample(a_wave, rate_in=tf.cast(a_sr, dtype=tf.int64),
                                   rate_out=tf.cast(sample_rate, dtype=tf.int64))
    a_mel = TFFeaturizer.tf_extract2(a_wave[:, 0])
    return  a_mel,label
#https://github.com/abhimanyu1996/Face-Recognition-using-triplet-loss
#https://github.com/zukakosan/tripletNet
if __name__ == '__main__':
    os.makedirs('saved',exist_ok=True)
    parser = argparse.ArgumentParser(description=" speaker model train")
    parser.add_argument("--train_data", "-train", default='data/train_200_23549.csv', type=str, help="train data", )
    parser.add_argument("--valid_data", "-valid", default='data/valid_200_23549.csv', type=str, help="valid data", )
    parser.add_argument("--net", "-net", default='ResNetV2', type=str, help="ResNetV1,ResNetV2", )
    parser.add_argument("--activation", "-act", default='amsoftmax', type=str, help="amsoftmax,softmax", )
    parser.add_argument("--pretrained", "-p", default=None, type=str, help="pretrained embedding", )
    parser.add_argument("--pretrained_amsoftmax", "-pas", default=None, type=str, help="pretrained amsoftmax", )
    parser.add_argument("--pretrained_softmax", "-ps", default=None, type=str, help="pretrained softmax", )
    parser.add_argument("--batch_size", "-bs", default=32, type=int, help="batch_size", )
    parser.add_argument("--epochs", "-epochs", default=100, type=int, help="epochs", )
    args = parser.parse_args()
    print(args)
    net=args.net
    #net="SpeakerResNet"
    train=pd.read_csv(args.train_data)
    valid = pd.read_csv(args.valid_data)

    train = train[['filepath','name','label']]
    valid = valid[['filepath','name','label']]
    classes = train.label.nunique()
    train_labels=train.label.values.tolist()#[:100]
    train = train.filepath.values.tolist()#[:100]
    valid_labels = valid.label.values.tolist()#[:10]
    valid = valid.filepath.values.tolist()#[:10]

    #print(train[:3])
    train_sum=len(train)

    #classes=len(train.label.unique())
    print('train data size:%s'%train_sum )
    print('valid data size:%s' % len(valid))
    #print(train.groupby('label')['filepath'].count(),valid.groupby('label')['filepath'].count())
    batch_size=args.batch_size
    epochs=args.epochs
    sample_rate=16000

    with tf.device("/cpu:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices( (train,train_labels) )
        train_dataset = train_dataset.shuffle(train_sum)
        train_dataset = train_dataset.map(parse_fun, num_parallel_calls=24)
        train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=( [None, None],[]))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid,valid_labels) )
        #train_dataset = train_dataset.shuffle(len(filenames))
        valid_dataset = valid_dataset.map(parse_fun, num_parallel_calls=16)
        valid_dataset = valid_dataset.padded_batch(batch_size, padded_shapes=([None, None] ,[]))
    if net=="ResNetV2":
        #model,speark_emb_model=SpeakerResNetV2(input_shape=(None, 80), classes=classes)
        model_softmax,model_amsoftmax,speark_emb_model= ResNetV2( input_shape=(None, 80), classes=classes)
    elif net=="DenseNet121":
        model_softmax, model_amsoftmax, speark_emb_model=DenseNet(blocks=[6,12,24,16],input_shape=(None,80),classes=classes,dropout=0.5)
    else:
        model_softmax,model_amsoftmax,speark_emb_model = ResNetV1(input_shape=(None,80),classes=classes,dropout_rate=0.5)
    #speark_emb_model.summary()
    #model.summary()
    if args.pretrained is not None:
        speark_emb_model.load_weights(args.pretrained)

    if args.pretrained_softmax is not None:
        model_softmax.load_weights(args.pretrained_softmax)
        #model_amsoftmax.layers[-1].set_weights( model_softmax.layers[-1].get_weights())
    if args.pretrained_amsoftmax is not None:
        model_amsoftmax.load_weights(args.pretrained_amsoftmax)
        #model_softmax.layers[-1].set_weights(model_amsoftmax.layers[-1].get_weights())

    decay_steps=int(0.4*train_sum/batch_size)
    print('decay_steps:',decay_steps)
    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=decay_steps,
                                                          end_learning_rate=0.0001,cycle=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=10.0)
    #optimizer=tf.keras.optimizers.SGD(learning_rate=lr_fn,momentum=0.95,nesterov=True,clipvalue=10.0)
    #loss={"triplet":triplet_loss_cosine,"softmax":tf.keras.losses.sparse_categorical_crossentropy}
    #loss=[triplet_loss_cosine,tf.keras.losses.sparse_categorical_crossentropy]
    accuracy=tf.keras.metrics.sparse_categorical_accuracy
    checkpoint = ModelCheck('saved/%s_%s_%s' % (net,args.activation, classes))
    if args.activation=='amsoftmax':
        model_amsoftmax.compile(loss=sparse_amsoftmax_loss, optimizer=optimizer,metrics=[accuracy])
        model_amsoftmax.fit(train_dataset, batch_size=batch_size, validation_data=valid_dataset, epochs=epochs,
              callbacks=[checkpoint])
    elif args.activation=='softmax':
        model_softmax.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=optimizer,
                                metrics=[accuracy])
        model_softmax.fit(train_dataset, batch_size=batch_size, validation_data=valid_dataset, epochs=epochs,
                            callbacks=[checkpoint])
    else:
        print(' only support amsoftmax or softmax mode for training .')
        sys.exit(0)



    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='saved/%s_triple_{val_loss:.4f}-{epoch:04d}.h5'% net  , verbose=1)

