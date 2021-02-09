import numpy as np
import joblib,os,argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from models import DataGenerator,build_model
from adamweightdecay import  WarmUp,AdamWeightDecay
import tensorflow_model_optimization as tfmot

#prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

class ModelCheck(keras.callbacks.Callback):

    def __init__(self, filepath,steps=10,pruning=False):
        super(ModelCheck, self).__init__()
        self.filepath =str(filepath)
        self.steps=steps
        self.best = np.Inf
        self.pruning=pruning

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
        mname = '_%.4f-%06d.h5' % (current, epoch)
        if epoch%self.steps==0 or self.best<=current :
            print(epoch, current, self.filepath+mname)
            try:
                if self.pruning==True:
                    model0=tfmot.sparsity.keras.strip_pruning(self.model)
                    model0.save(self.filepath + mname,include_optimizer=False)
                else:
                    self.model.save(self.filepath+mname,include_optimizer=False)
                #keras.models.save_model(self.model, pruned_keras_file, include_optimizer=False)
            except Exception as e:
                print(e)
        if self.best<=current:
            self.best=current


def train_RNN(DATA_ROOT,model=None,opt='RMSprop',net='GRU',wp=10,n_in=467, n_out=1, n_layers=3, hidden_layer_size=512,epochs=100,batch_size=16,drop_rate=0.5,lr=0.001,seq_length=1000,pruning=False):
    if model is None:
        model = build_model(net=net,n_in=n_in, n_out=n_out, n_layers=n_layers, hidden_layer_size=hidden_layer_size,drop_rate=drop_rate)
    train_filelist=open(os.path.join(DATA_ROOT , 'train_filelist.txt') ).read().split('\n')
    train_filelist=[x for x in train_filelist if len(x.strip())>0]
    nn=len(train_filelist)
    val_filelist = open(os.path.join(DATA_ROOT ,  'val_filelist.txt') ).read().split('\n')
    val_filelist=[x for x in val_filelist if len(x.strip())>0]
    train_gen=DataGenerator(train_filelist,DATA_ROOT  = DATA_ROOT,ty='acoustic',n_in=n_in, n_out=n_out,batch_size=batch_size,seq_length=seq_length)
    val_gen = DataGenerator(val_filelist,DATA_ROOT  = DATA_ROOT, ty='acoustic', n_in=n_in, n_out=n_out,batch_size=batch_size,seq_length=seq_length, shuffle=False)
    calllist = [ ModelCheck(os.path.join(DATA_ROOT,'saved/acoustic_%s_%s' % (net, hidden_layer_size)) ,steps=10,pruning=pruning),
                 #keras.callbacks.ModelCheckpoint(os.path.join(DATA_ROOT , 'saved/acoustic_%s_%s_{val_loss:.4f}-{epoch:06d}.h5' %(net,hidden_layer_size)),save_best_only=True,save_weights_only=False),
                #ModelCheck(os.path.join(DATA_ROOT,'saved/acoustic_%s_%s_{val_loss:.4f}-{epoch:06d}.h5' % (net, hidden_layer_size)) ),
                #keras.callbacks.TensorBoard(log_dir='./logs',update_freq='epoch'),
                ]
    if pruning==True:
        print('Fine-tune pre-trained model with pruning...')
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                 final_sparsity=0.70,
                                                                 begin_step=15*nn//batch_size,
                                                                 end_step=epochs*nn//batch_size)}
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        pr_callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='logs'),
        ]
        calllist=calllist+pr_callbacks

    learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr, decay_steps=epochs*nn//batch_size//2, end_learning_rate=0.0000001 )

    learning_rate_fn = WarmUp(initial_learning_rate=lr,decay_schedule_fn=learning_rate_fn,warmup_steps=wp*nn//batch_size)

    if opt=='SGD':
        opt=keras.optimizers.SGD(learning_rate=learning_rate_fn,momentum=0.95,nesterov=True)
    elif opt=='Adam':
        opt=keras.optimizers.Adam(learning_rate=learning_rate_fn,amsgrad=True)
    elif opt=='AdamW':
        opt = AdamWeightDecay(learning_rate=learning_rate_fn,
            weight_decay_rate=0.001,beta_1=0.9,beta_2=0.98,epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias", 'batch_normalization', 'BatchNormalization',
                                       'dense', 'Dense'] )
    else:
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate_fn, momentum=0.95)

    model.compile(loss=keras.losses.mae, optimizer=opt)

    model_json = model.to_json()
    with open(os.path.join(DATA_ROOT , 'saved/acoustic_%s_%s.json' %(net,hidden_layer_size)), 'w') as file:
        file.write(model_json)

    model.fit(train_gen, batch_size=batch_size, epochs=epochs,validation_data=val_gen, callbacks=calllist,workers=5)
    if pruning == True:
        model = tfmot.sparsity.keras.strip_pruning(model)
    model.save(os.path.join(DATA_ROOT ,  'saved/acoustic_%s_%s_%06d.h5' % (net,hidden_layer_size,epochs)) ,include_optimizer=False)

def get_args():
    parser = argparse.ArgumentParser(description="train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='DNN')
    parser.add_argument("--batch_size","-batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs","-epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument('--pretrained',"-p",type=str,default=None,help='  model  h5 file')
    parser.add_argument("--learning_rate",'-lr', type=float,default=0.001, help="learning_rate")
    parser.add_argument("--pruning", '-prun', type=int, default=0, help="model pruning ")
    parser.add_argument("--warmup", '-wp', type=int, default=50, help="warmup ")
    parser.add_argument("--seq_length", '-seq', type=int, default=1984, help="seq length ")
    parser.add_argument('--DATA_ROOT','-data_root', type=str,default='./thchs30_250_demo/',help="data root")
    parser.add_argument("--optimizers", '-opt', type=str, default='AdamW', help="optimizers:SGD,Adam,RMSprop")
    parser.add_argument("--fs", '-fs', type=int, default=22050, help="fs")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    DATA_ROOT = args.DATA_ROOT

    batch_size = args.batch_size
    epochs = args.epochs
    net = args.net
    lr=args.learning_rate
    pretrained=args.pretrained
    opt=args.optimizers
    wp=args.warmup
    #decay=args.decay
    seq_length=args.seq_length
    pruning=False
    if args.pruning>0:
        pruning=True
    if os.path.exists(os.path.join(DATA_ROOT , 'saved') )==False:
        os.mkdir(os.path.join(DATA_ROOT , 'saved'))

    n_in=471
    fs=args.fs
    if fs==48000:
        n_out=199
    elif fs==16000:
        n_out=187
    elif fs ==22050:
        n_out=190
    else:
        pass

    if pretrained is not None:
        model=keras.models.load_model(os.path.join(DATA_ROOT ,'saved',pretrained),custom_objects={'AdamWeightDecay':AdamWeightDecay})
    else:
        model=None

    if net=='DNN':
        train_RNN(DATA_ROOT,model=model,opt=opt,wp=wp, net=net, n_in=n_in, n_out=n_out,n_layers=6, hidden_layer_size=1024, drop_rate=0.5,batch_size=batch_size,epochs=epochs,lr=lr,seq_length=seq_length,pruning=pruning)
    elif net in ['GRU','LSTM','CNN','ResCNN']:
        train_RNN(DATA_ROOT,model=model,opt=opt,wp=wp, net=net, n_in=n_in, n_out=n_out, n_layers=3, hidden_layer_size=512,batch_size=batch_size,epochs=epochs, drop_rate=0.35,lr=lr,seq_length=seq_length,pruning=pruning)
    elif net in ['BLSTM','BGRU'] :
        train_RNN(DATA_ROOT,model=model,opt=opt,wp=wp, net=net, n_in=n_in, n_out=n_out, n_layers=3, hidden_layer_size=384,batch_size=batch_size,epochs=epochs, drop_rate=0.35,lr=lr,seq_length=seq_length,pruning=pruning)
    else:
        pass