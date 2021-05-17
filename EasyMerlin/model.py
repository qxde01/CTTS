import sys,joblib
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler



def build_model(net,n_in,n_out,n_layers=6,hidden_layer_size=512,drop_rate=0.5):
    if net =='DNN':
        model=DNN(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='GRU' :
        model=GRU2(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='LSTM' :
        model=LSTM(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='BLSTM' :
        model=BLSTM(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='BGRU' :
        model=BGRU(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='CNN' :
        model=CNN(n_in, n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    elif net=='ResCNN' :
        model=ResCNN(n_in, n_out,  hidden_layer_size=hidden_layer_size, drop_rate=drop_rate)
    else:
        print('your net not support.')
        sys.exit(0)
    return model


def DNN(n_in,n_out,n_layers=6,hidden_layer_size=1024,drop_rate=0.5):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(None,n_in), name='input'))
    for i in range(0,n_layers):
        model.add(keras.layers.Dense(units=hidden_layer_size,activation=None,kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l2(0.00001)))
        #model.add(keras.layers.BatchNormalization(momentum=0.95))
        model.add(keras.layers.Activation('tanh'))
    if drop_rate>0:
        model.add(keras.layers.Dropout(drop_rate))
    model.add(keras.layers.Dense(units=n_out,activation='linear',kernel_initializer="normal",input_dim=hidden_layer_size,name='prediction'))
    #model.compile(loss=keras.losses.mse, optimizer='adam', metrics=[keras.losses.mae])
    model.summary()
    return model

def CNN(n_in,n_out,n_layers=3,hidden_layer_size=512,drop_rate=0.5):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(None,n_in),name='input'))
    #model.add(keras.layers.Dense(units=hidden_layer_size, activation='tanh', kernel_initializer="glorot_normal", input_shape=(None, n_in)))

    for i in range(0, n_layers-1):
        model.add(keras.layers.Conv1D(filters=hidden_layer_size,kernel_size=5,activation=None,kernel_initializer='glorot_normal',padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.95))
        #model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.GRU(units=hidden_layer_size, return_sequences=True))
    if drop_rate > 0:
        model.add(keras.layers.Dropout(drop_rate))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size,name='prediction'))
    model.summary()
    return model

def ResCNN(n_in,n_out,hidden_layer_size=512,drop_rate=0.5):
    inputs=keras.layers.Input(shape=(None,n_in))
    #x=keras.layers.Dense(units=hidden_layer_size, activation=None, kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l2(0.00001))(inputs)
    #x0 = keras.layers.BatchNormalization(momentum=0.95)(x)
    #x1=keras.layers.Activation('tanh')(x0)
    x = keras.layers.GRU(units=hidden_layer_size, return_sequences=True)(inputs)
    x1=keras.layers.Conv1D(filters=hidden_layer_size, kernel_size=3, activation=None, kernel_initializer='glorot_normal',padding='same')(x)
    x1=keras.layers.BatchNormalization(momentum=0.95)(x1)
    x1=keras.layers.LeakyReLU()(x1)
    x2 = keras.layers.Conv1D(filters=hidden_layer_size, kernel_size=3, activation=None,kernel_initializer='glorot_normal',padding='same')(x1)
    #x4=keras.layers.Add()([x2,x])
    x4 = keras.layers.Multiply()([x2,x])
    x4 = keras.layers.BatchNormalization(momentum=0.95)(x4)
    x4 = keras.layers.LeakyReLU()(x4)
    x4 = keras.layers.Conv1D(filters=hidden_layer_size, kernel_size=5, activation=None,kernel_initializer='glorot_normal',padding='same')(x4)
    x4 = keras.layers.BatchNormalization(momentum=0.95)(x4)
    x4 = keras.layers.LeakyReLU()(x4)
    #x4 = keras.layers.GRU(units=hidden_layer_size, return_sequences=True)(x4)
    if drop_rate > 0:
        x4=keras.layers.Dropout(drop_rate)(x4)
    x4=keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size,name='prediction')(x4)
    model=keras.models.Model(inputs,x4)
    model.summary()
    return model





def GRU(n_in,n_out,n_layers=3,hidden_layer_size=512,drop_rate=0.5):
    model = keras.models.Sequential()
    for i in range(0,n_layers):
        model.add(keras.layers.GRU(units=hidden_layer_size,input_shape=(None, n_in),return_sequences=True,recurrent_dropout=drop_rate))
    #model.add(keras.layers.GRU(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=False))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size))
    model.summary()
    return model

def GRU2(n_in,n_out,n_layers=3,hidden_layer_size=512,drop_rate=0.5):
    n_layers =n_layers
    model = keras.models.Sequential()
    #for i in range(0,n_layers):
    model.add(keras.layers.GRU(units=hidden_layer_size,input_shape=(None, n_in),return_sequences=True,recurrent_dropout=0.0))
    model.add(keras.layers.Conv1D(filters=hidden_layer_size, kernel_size=5, activation=None,kernel_initializer='glorot_normal', padding='same'))
    model.add(keras.layers.BatchNormalization(momentum=0.95))
    # model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.GRU(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=True,recurrent_dropout=0.0))
    #model.add(keras.layers.GRU(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=False))
    if drop_rate > 0:
        model.add(keras.layers.Dropout(drop_rate))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size,name='prediction'))
    model.summary()
    return model


def BGRU(n_in,n_out,n_layers=3,hidden_layer_size=384,drop_rate=0.5):
    model = keras.models.Sequential()
    for i in range(0,n_layers):
        model.add(keras.layers.GRU(units=hidden_layer_size,input_shape=(None, n_in),return_sequences=True,recurrent_dropout=drop_rate,go_backwards=True))
    #model.add(keras.layers.GRU(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=False))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size))
    model.summary()
    return model


def LSTM(n_in,n_out,n_layers=3,hidden_layer_size=512,drop_rate=0.5):
    model = keras.models.Sequential()
    for i in range(0,n_layers):
        model.add(keras.layers.LSTM(units=hidden_layer_size,input_shape=(None, n_in),return_sequences=True,
                                    recurrent_dropout=drop_rate))
    #model.add(keras.layers.LSTM(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=False,recurrent_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size))
    model.summary()
    return model

def BLSTM(n_in,n_out,n_layers=3,hidden_layer_size=384,drop_rate=0.5):
    model = keras.models.Sequential()
    for i in range(0,n_layers):
        model.add(keras.layers.LSTM(units=hidden_layer_size,input_shape=(None, n_in),return_sequences=True,
                                    recurrent_dropout=drop_rate,go_backwards=True))
    #model.add(keras.layers.LSTM(units=hidden_layer_size, input_shape=(None, n_in), return_sequences=False,recurrent_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size))
    model.summary()
    return model

def BNConv(x,hidden_layer_size=256,kernel_size=9):
    x=keras.layers.Conv1D(filters=hidden_layer_size, kernel_size=kernel_size, activation=None, kernel_initializer='glorot_normal', padding='same')(x)
    x=keras.layers.BatchNormalization(momentum=0.95)(x)
    x=keras.layers.LeakyReLU()(x)
    return x

def BNSepConv(x,hidden_layer_size=256,kernel_size=9):
    x=keras.layers.SeparableConv1D(filters=hidden_layer_size, kernel_size=kernel_size, activation=None, kernel_initializer='glorot_normal', padding='same')(x)
    x=keras.layers.BatchNormalization(momentum=0.95)(x)
    x=keras.layers.LeakyReLU()(x)
    return x

def build_Mel_model(n_in,n_out,n_layers=3,hidden_layer_size=384,drop_rate=0.5):
    inputs=keras.layers.Input(shape=(None,n_in),name='input')
    mel=keras.layers.GRU(units=hidden_layer_size, return_sequences=True)(inputs)

    for i in range(0, n_layers-1):
        mel=BNConv(mel,hidden_layer_size=hidden_layer_size,kernel_size=9)

    mel=BNSepConv(mel,hidden_layer_size=hidden_layer_size,kernel_size=9)
    if drop_rate > 0:
        mel_drop=keras.layers.Dropout(drop_rate)(mel)
    else:
        mel_drop=mel
    mel_before=keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal", input_dim=hidden_layer_size,name='mel_before')(mel_drop)
    f0_embed = BNSepConv(mel_drop, hidden_layer_size=hidden_layer_size, kernel_size=9)
    f0=keras.layers.Conv1D(filters=1, kernel_size=9, activation=None, kernel_initializer='glorot_normal', padding='same',name='f0')(f0_embed)
    engergy_embed = BNSepConv(mel_drop, hidden_layer_size=hidden_layer_size, kernel_size=9)
    engergy = keras.layers.Conv1D(filters=1, kernel_size=9, activation=None,kernel_initializer='glorot_normal', padding='same', name='engergy')(engergy_embed)
    mel_after=keras.layers.add([mel,f0_embed,engergy_embed],name='encode_merge')
    if drop_rate > 0:
        mel_after_drop=keras.layers.Dropout(drop_rate)(mel_after)
    else:
        mel_after_drop=mel_after
    mel_after = keras.layers.Dense(units=n_out, activation='linear', kernel_initializer="normal",input_dim=hidden_layer_size, name='mel_after_dense')(mel_after_drop)
    mel_after=keras.layers.add([mel_after,mel_before],name='mel_after')
    model=keras.models.Model(inputs=inputs,outputs=[mel_before,mel_after,f0,engergy])
    model.summary()
    infer = keras.models.Model(inputs=inputs, outputs=mel_after)
    return model,infer


class DataGenerator2(keras.utils.Sequence):
    def __init__(self, filelist,DATA_ROOT , batch_size=1,ty='acoustic',net='DNN', shuffle=True):
        self.batch_size = batch_size
        self.filelist=filelist
        self.DATA_ROOT  = DATA_ROOT
        self.ty=ty
        self.net = net
        self.shuffle = shuffle
        if self.ty=='acoustic':
            self.X_mms=joblib.load(self.DATA_ROOT + 'X_acoustic_mms.pkl')
            self.Y_std=joblib.load(self.DATA_ROOT + 'Y_acoustic_std.pkl')
        else:
            self.X_mms = joblib.load(self.DATA_ROOT + 'X_duration_mms.pkl')
            self.Y_std = joblib.load(self.DATA_ROOT + 'Y_duration_std.pkl')

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return int(np.ceil(len(self.filelist) / float(self.batch_size)))

    def __getitem__(self, index):
        fpx = self.DATA_ROOT + 'X_' +self.ty + '/' + self.filelist[index] + '.npz'
        X = np.load(fpx)['x']
        fpy = self.DATA_ROOT + 'Y_' + self.ty + '/' + self.filelist[index] + '.npz'
        Y = np.load(fpy)['y']
        X=self.X_mms.transform(X)
        Y=self.Y_std.transform(Y)
        if self.net=='DNN':
            return X, Y
        else:
            X=X.reshape(1,X.shape[0],X.shape[1])
            Y = Y.reshape(1, Y.shape[0], Y.shape[1])
        return X,Y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.filelist)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, filelist,DATA_ROOT ,n_in=471,n_out=199, batch_size=16,ty='acoustic',seq_length=1000, shuffle=True):
        self.batch_size = batch_size
        self.filelist=filelist
        self.DATA_ROOT  = DATA_ROOT
        self.ty=ty
        #self.net = net
        self.shuffle = shuffle
        self.seq_length=seq_length
        self.samples_num=len(self.filelist)
        self.n_in=n_in
        self.n_out=n_out
        if self.ty=='acoustic':
            self.X_mms=joblib.load(self.DATA_ROOT + 'X_acoustic_mms.pkl')
            self.Y_std=joblib.load(self.DATA_ROOT + 'Y_acoustic_std.pkl')
        else:
            self.X_mms = joblib.load(self.DATA_ROOT + 'X_duration_mms.pkl')
            self.Y_std = joblib.load(self.DATA_ROOT + 'Y_duration_std.pkl')

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return int(np.ceil(len(self.filelist) / float(self.batch_size)))

    def __getitem__(self, index):
        i = index * self.batch_size
        length = min(self.batch_size, (self.samples_num - i))

        batch_inputs = np.zeros((length, self.seq_length, self.n_in), dtype=np.float32)
        batch_outputs = np.zeros((length, self.seq_length, self.n_out), dtype=np.float32)
        for i_batch in range(0, length):
            fpx = self.DATA_ROOT + 'X_' + self.ty + '/' + self.filelist[i + i_batch] + '.npz'
            fpy = self.DATA_ROOT + 'Y_' + self.ty + '/' + self.filelist[i + i_batch] + '.npz'
            X = np.load(fpx)['x']
            Y = np.load(fpy)['y']
            X = self.X_mms.transform(X)
            Y = self.Y_std.transform(Y)
            mm=min(X.shape[0],self.seq_length)
            batch_inputs[i_batch,:mm,:]=X
            batch_outputs[i_batch, :mm, :] = Y
        return batch_inputs,batch_outputs

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.filelist)


class DataGenerator_mel(keras.utils.Sequence):
    def __init__(self, filelist,DATA_ROOT ,n_in=471,n_out=199, batch_size=16,seq_length=1000, shuffle=True):
        self.batch_size = batch_size
        self.filelist=filelist
        self.DATA_ROOT  = DATA_ROOT
        #self.ty=ty
        #self.net = net
        self.shuffle = shuffle
        self.seq_length=seq_length
        self.samples_num=len(self.filelist)
        self.n_in=n_in
        self.n_out=n_out
        self.X_mms = joblib.load(self.DATA_ROOT + 'X_linguistic_mms_std.pkl')
        self.Y_mel_std = joblib.load(self.DATA_ROOT + 'Y_mel_std.pkl')
        self.Y_f0_std = joblib.load(self.DATA_ROOT + 'Y_f0_std.pkl')
        self.Y_energy_std = joblib.load(self.DATA_ROOT + 'Y_energy_std.pkl')

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return int(np.ceil(len(self.filelist) / float(self.batch_size)))

    def __getitem__(self, index):
        i = index * self.batch_size
        length = min(self.batch_size, (self.samples_num - i))

        batch_inputs = np.zeros((length, self.seq_length, self.n_in), dtype=np.float32)
        batch_mel = np.zeros((length, self.seq_length, self.n_out), dtype=np.float32)
        batch_f0 = np.zeros((length, self.seq_length, 1), dtype=np.float32)
        batch_energy = np.zeros((length, self.seq_length, 1), dtype=np.float32)
        for i_batch in range(0, length):
            fpx = self.DATA_ROOT + 'X_linguistic/' + self.filelist[i + i_batch] + '.npy'
            fpy1 = self.DATA_ROOT + 'Y_mel/' + self.filelist[i + i_batch] + '.npy'
            fpy2 = self.DATA_ROOT + 'Y_f0/' + self.filelist[i + i_batch] + '.npy'
            fpy3 = self.DATA_ROOT + 'Y_energy/' + self.filelist[i + i_batch] + '.npy'

            X = np.load(fpx)
            Y_mel = np.load(fpy1)
            Y_f0=np.load(fpy2)
            Y_energy = np.load(fpy3)

            X = self.X_mms.transform(X)
            Y_mel = self.Y_mel_std.transform(Y_mel)
            Y_f0 = self.Y_f0_std.transform(Y_f0.reshape(-1,1))
            Y_energy = self.Y_energy_std.transform(Y_energy.reshape(-1,1))
            mm=min(X.shape[0],self.seq_length)
            batch_inputs[i_batch,:mm,:]=X
            batch_mel[i_batch, :mm, :] = Y_mel
            batch_f0[i_batch, :mm, :] = Y_f0.reshape( Y_f0.shape[0],1 )
            batch_energy[i_batch, :mm, :] = Y_energy.reshape(Y_energy.shape[0],1)
        return batch_inputs,(batch_mel,batch_mel,batch_f0,batch_energy)

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.filelist)
