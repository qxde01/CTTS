import os,argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
import numpy as np
#from utils import TFMel
from ResNet import SpeakerResNetV2
from losses import triplet_loss_cosine,sparse_amsoftmax_loss
from models import SpeakerResNetV1
from DenseNet import SpeakerDenseNet
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
    a_raw_audio = tf.io.read_file(item[0])
    a_wave, a_sr = tf.audio.decode_wav(a_raw_audio, desired_channels=1, desired_samples=-1)
    if a_sr != sample_rate:
        a_wave = tfio.audio.resample(a_wave, rate_in=tf.cast(a_sr, dtype=tf.int64),
                                   rate_out=tf.cast(sample_rate, dtype=tf.int64))
    a_mel = TFFeaturizer.tf_extract2(a_wave[:, 0])

    p_raw_audio = tf.io.read_file(item[1])
    p_wave, p_sr = tf.audio.decode_wav(p_raw_audio, desired_channels=1, desired_samples=-1)
    if p_sr != sample_rate:
        p_wave = tfio.audio.resample(p_wave, rate_in=tf.cast(p_sr, dtype=tf.int64),
                                     rate_out=tf.cast(sample_rate, dtype=tf.int64))
    p_mel = TFFeaturizer.tf_extract2(p_wave[:, 0])

    n_raw_audio = tf.io.read_file(item[2])
    n_wave, n_sr = tf.audio.decode_wav(n_raw_audio, desired_channels=1, desired_samples=-1)
    if n_sr != sample_rate:
        n_wave = tfio.audio.resample(n_wave, rate_in=tf.cast(n_sr, dtype=tf.int64),
                                     rate_out=tf.cast(sample_rate, dtype=tf.int64))
    n_mel = TFFeaturizer.tf_extract2(n_wave[:, 0])
    a_r = tf.shape(a_mel)[0]
    p_r = tf.shape(p_mel)[0]
    n_r = tf.shape(n_mel)[0]
    max_r=tf.maximum(tf.maximum(a_r,n_r),p_r)
    a_mel=tf.pad(a_mel,paddings=tf.convert_to_tensor([[0,max_r-a_r],[0,0]] ,dtype=tf.int32))
    p_mel = tf.pad(p_mel, paddings=tf.convert_to_tensor([[0,max_r - p_r], [0, 0]] ,dtype=tf.int32))
    n_mel = tf.pad(n_mel, paddings=tf.convert_to_tensor([[0,max_r - n_r], [0, 0]] ,dtype=tf.int32))
    return  (a_mel, p_mel, n_mel),label
#https://github.com/abhimanyu1996/Face-Recognition-using-triplet-loss
#https://github.com/zukakosan/tripletNet
if __name__ == '__main__':
    os.makedirs('saved',exist_ok=True)
    parser = argparse.ArgumentParser(description=" speaker model train")
    parser.add_argument("--train_data", "-train", default='data/triple_train_3890_2647600.csv', type=str, help="train data", )
    parser.add_argument("--valid_data", "-valid", default='data/triple_valid_3890_139210.csv', type=str, help="valid data", )
    parser.add_argument("--net", "-net", default='SpeakerResNetV1', type=str, help="SpeakerResNetV1,SpeakerResNetV2", )
    parser.add_argument("--pretrained", "-p", default=None, type=str, help="text", )
    parser.add_argument("--batch_size", "-bs", default=16, type=int, help="batch_size", )
    parser.add_argument("--epochs", "-epochs", default=100, type=int, help="epochs", )
    args = parser.parse_args()
    print(args)
    net=args.net
    #net="SpeakerResNet"
    train=pd.read_csv(args.train_data)
    valid = pd.read_csv(args.valid_data)
    classes = train.anchor_label.nunique()
    train = train[['anchor_filepath', 'positive_filepath', 'negative_filepath','anchor_label']]
    valid = valid[['anchor_filepath', 'positive_filepath', 'negative_filepath','anchor_label']]
    train['anchor_label']=train['anchor_label'].astype(np.float32)
    valid['anchor_label'] = valid['anchor_label'].astype(np.float32)
    #train_labels=train.anchor_label.values.tolist()valid
    #valid_labels = valid.anchor_label.values.tolist()
    train_labels=train['anchor_label'].values.tolist()#[:100]
    train = train[['anchor_filepath', 'positive_filepath', 'negative_filepath']].values.tolist()#[:100]
    valid_labels = valid['anchor_label'].values.tolist()#[:10]
    valid = valid[['anchor_filepath', 'positive_filepath', 'negative_filepath']].values.tolist()#[:10]

    #print(train[:3])
    train_sum=len(train)

    #classes=len(train.label.unique())
    print('train data size:%s'%train_sum )
    print('valid data size:%s s' % len(valid))
    #print(train.groupby('label')['filepath'].count(),valid.groupby('label')['filepath'].count())
    batch_size=args.batch_size
    epochs=args.epochs
    sample_rate=16000

    with tf.device("/cpu:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices( (train,train_labels) )
        train_dataset = train_dataset.shuffle(train_sum)
        train_dataset = train_dataset.map(parse_fun, num_parallel_calls=24)
        train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=( ([None, None], [None, None],[None, None]),[]))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid,valid_labels) )
        #train_dataset = train_dataset.shuffle(len(filenames))
        valid_dataset = valid_dataset.map(parse_fun, num_parallel_calls=16)
        valid_dataset = valid_dataset.padded_batch(batch_size, padded_shapes=(([None, None], [None, None],[None, None]),[]))
    if net=="SpeakerResNetV2":
        model,speark_emb_model=SpeakerResNetV2(input_shape=(None, 80), classes=classes)
    elif net=='SpeakerDenseNet':
        model, speark_emb_model =SpeakerDenseNet(input_shape=(None, 80), classes=classes)
    else:
        model, speark_emb_model = SpeakerResNetV1(input_shape=(None, 80), classes=classes,dropout_rate=0.5)
    #speark_emb_model.summary()
    model.summary()
    if args.pretrained is not None:
        speark_emb_model.load_weights(args.pretrained)
    decay_steps=int(0.6*epochs*train_sum/batch_size)
    print('decay_steps:',decay_steps)
    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=decay_steps,
                                                       end_learning_rate=0.0001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn, clipvalue=10.0,nesterov=True,momentum=0.95)
    #loss={"triplet":triplet_loss_cosine,"softmax":tf.keras.losses.sparse_categorical_crossentropy}
    loss=[triplet_loss_cosine,tf.keras.losses.sparse_categorical_crossentropy]
    #loss = [triplet_loss_cosine, sparse_amsoftmax_loss]
    model.compile(loss=loss, optimizer=optimizer,metrics={"softmax":'acc'})
    checkpoint = ModelCheck('saved/%s_%s' % (net,classes ))
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='saved/%s_triple_{val_loss:.4f}-{epoch:04d}.h5'% net  , verbose=1)
    model.fit(train_dataset, batch_size=batch_size, validation_data=valid_dataset, epochs=epochs, callbacks=[checkpoint])
