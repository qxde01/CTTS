"""Prepare acoustic/linguistic/duration features.
usage:
    prepare_features.py [options] <DATA_ROOT>

"""
from __future__ import division, print_function, absolute_import
#from docopt import docopt
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts
from os.path import join
from glob import glob
import pysptk
import pyworld
from scipy.io import wavfile
import wavio
from tqdm import tqdm
from os.path import basename, splitext, exists
import os,joblib
import sys,argparse

global DATA_ROOT

max_num_files = None
order = 59
frame_period = 5
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

class LinguisticSource(FileDataSource):
    def __init__(self, add_frame_features=False, subphone_features=None, use_phone_alignment=False, question_path=None):
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.test_paths = None
        self.use_phone_alignment = use_phone_alignment
        if question_path is None:
            self.binary_dict, self.continuous_dict = hts.load_question_set(join('misc', "questions-mandarin.hed"))
        else:
            self.binary_dict, self.continuous_dict = hts.load_question_set(question_path)

    def collect_files(self):
        if self.use_phone_alignment:
            files = sorted(glob(join(DATA_ROOT, "label_phone_align", "*.lab")))
        else:
            files = sorted(glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return files[:max_num_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(labels, self.binary_dict, self.continuous_dict,add_frame_features=self.add_frame_features,subphone_features=self.subphone_features)
        if self.add_frame_features:
            indices = labels.silence_frame_indices().astype(np.int)
        else:
            indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)
        #print('Linguistic:',features.shape)
        return features.astype(np.float32)


class DurationFeatureSource(FileDataSource):
    def __init__(self, use_phone_alignment=False):
        self.use_phone_alignment = use_phone_alignment

    def collect_files(self):
        if self.use_phone_alignment:
            files = sorted(glob(join(DATA_ROOT, "label_phone_align", "*.lab")))
        else:
            files = sorted(glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return files[:max_num_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)
        #print('DurationFeature:',features.shape)
        return features.astype(np.float32)


class AcousticSource(FileDataSource):
    def __init__(self, use_phone_alignment=False):
        self.use_phone_alignment = use_phone_alignment

    def collect_files(self):
        wav_paths = sorted(glob(join(DATA_ROOT, "wav", "*.wav")))
        if self.use_phone_alignment:
            label_paths = sorted( glob(join(DATA_ROOT, "label_phone_align", "*.lab")))
        else:
            label_paths = sorted(glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return wav_paths[:max_num_files], label_paths[:max_num_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        #print(wav_path)
        #fs, x = wavfile.read(wav_path)
        d=wavio.read(wav_path)
        fs,x=d.rate,d.data
        print(fs,wav_path)
        if len(x.shape)>1:
            x=x[:,0]
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spectrogram, order=order, alpha=pysptk.util.mcepalpha(fs))
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        vuv = (lf0 != 0).astype(np.float32) #1
        lf0 = interp1d(lf0, kind="slinear")

        mgc = apply_delta_windows(mgc, windows) #180
        lf0 = apply_delta_windows(lf0, windows) #3
        bap = apply_delta_windows(bap, windows) #3 biaobei 15

        features = np.hstack((mgc, lf0, vuv, bap)) # 187 biaobei 199
        #print('mgc:',mgc.shape)
        #print('lf0:', lf0.shape)
        #print('vuv:', vuv.shape)
        #print('bap:', bap.shape)

        # Cut silence frames by HTS alignment
        labels = hts.load(label_path)
        features = features[:labels.num_frames()]
        indices = labels.silence_frame_indices()
        if len(indices)>0:
            features = np.delete(features, indices, axis=0)
        #print(features.shape) #
        return features.astype(np.float32)

def get_args():
        parser = argparse.ArgumentParser(description="train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--DATA_ROOT', '-data_root', type=str, default='thchs30_250_demo/', help="data root")
        args = parser.parse_args()
        return args
if __name__ == "__main__":
    args = get_args()
    print(args)
    DATA_ROOT = args.DATA_ROOT
    DST_ROOT = DATA_ROOT
    max_num_files =1150000 # int(args["--max_num_files"])
    #overwrite = True #args["--overwrite"]
    use_phone_alignment =True # args["--use_phone_alignment"]
    question_path =None # args["--question_path"]
    X_acoustic_mms = MinMaxScaler(feature_range=(0.01, 0.99),copy=False)
    Y_acoustic_std = StandardScaler(copy=False)
    X_duration_mms = MinMaxScaler(feature_range=(0.01, 0.99),copy=False)
    Y_duration_std = StandardScaler(copy=False)

    X_duration_source = LinguisticSource(add_frame_features=False, subphone_features=None, use_phone_alignment=use_phone_alignment, question_path=question_path)
    Y_duration_source = DurationFeatureSource( use_phone_alignment=use_phone_alignment)

    X_duration = FileSourceDataset(X_duration_source)
    Y_duration = FileSourceDataset(Y_duration_source)

    # Features required to train acoustic model
    # X -> Y
    # X: linguistic
    # Y: acoustic
    subphone_features = "full" if not use_phone_alignment else "coarse_coding"
    X_acoustic_source = LinguisticSource(add_frame_features=True, subphone_features=subphone_features,use_phone_alignment=use_phone_alignment, question_path=question_path)
    Y_acoustic_source = AcousticSource(use_phone_alignment=use_phone_alignment)
    X_acoustic = FileSourceDataset(X_acoustic_source)
    Y_acoustic = FileSourceDataset(Y_acoustic_source)

    # Save as files
    X_duration_root = join(DST_ROOT, "X_duration")
    Y_duration_root = join(DST_ROOT, "Y_duration")
    X_acoustic_root = join(DST_ROOT, "X_acoustic")
    Y_acoustic_root = join(DST_ROOT, "Y_acoustic")

    skip_duration_feature_extraction = exists(X_duration_root) and exists(Y_duration_root)
    skip_acoustic_feature_extraction = exists(X_acoustic_root) and exists(Y_acoustic_root)

    for d in [X_duration_root, Y_duration_root, X_acoustic_root, Y_acoustic_root]:
        if not os.path.exists(d):
            print("mkdirs: {}".format(d))
            os.makedirs(d)

    # Save features for duration model
    x_d_n=y_d_n=0
    if not skip_duration_feature_extraction:
        print("Duration linguistic feature dim", X_duration[0].shape) #(?, 467)
        print("Duration feature dim", Y_duration[0].shape) #dim(?, 1)
        for idx, (x, y) in tqdm(enumerate(zip(X_duration, Y_duration))):
            name = splitext(basename(X_duration.collected_files[idx][0]))[0]
            xpath = join(X_duration_root, name + ".bin")
            ypath = join(Y_duration_root, name + ".bin")
            x_d_n=max(x_d_n,x.shape[0])
            y_d_n = max(y_d_n, y.shape[0])
            #x.tofile(xpath)
            #y.tofile(ypath)
            X_duration_mms.partial_fit(x)
            Y_duration_std.partial_fit(y)
            np.savez(xpath.replace('.bin', ''), x=x)
            np.savez(ypath.replace('.bin', ''), y=y)
    else:
        print("Features for duration model training found, skipping feature extraction.")

    # Save features for acoustic model
    x_a_n = y_a_n = 0
    if not skip_acoustic_feature_extraction:
        print("Acoustic linguistic feature dim", X_acoustic[0].shape) #(?,471)
        print("Acoustic feature dim", Y_acoustic[0].shape) #(?,187)
        for idx, (x, y) in tqdm(enumerate(zip(X_acoustic, Y_acoustic))):
            name = splitext(basename(X_acoustic.collected_files[idx][0]))[0]
            xpath = join(X_acoustic_root, name + ".bin")
            ypath = join(Y_acoustic_root, name + ".bin")
            #x.tofile(xpath)
            #y.tofile(ypath)
            x_a_n = max(x_a_n, x.shape[0])
            y_a_n = max(y_a_n, y.shape[0])
            X_acoustic_mms.partial_fit(x)
            Y_acoustic_std.partial_fit(y)
            np.savez(xpath.replace('.bin', ''), x=x)
            np.savez(ypath.replace('.bin', ''), y=y)
    else:
        print("Features for acousic model training found, skipping feature extraction.")
    joblib.dump(X_acoustic_mms, join(DATA_ROOT ,'X_acoustic_mms.pkl') )
    joblib.dump(Y_acoustic_std, join(DATA_ROOT + 'Y_acoustic_std.pkl'))
    joblib.dump(X_duration_mms, join(DATA_ROOT + 'X_duration_mms.pkl'))
    joblib.dump(Y_duration_std, join(DATA_ROOT + 'Y_duration_std.pkl'))
    print('X acoustic max seq length:',x_a_n)
    print('Y acoustic max seq length:', y_a_n)
    print('X duration max seq length:', x_d_n)
    print('Y duration max seq length:', y_d_n)

    filelist=sorted(os.listdir(os.path.join( DATA_ROOT,'label_phone_align')) )
    filelist =[x.replace('.lab','') for x in filelist]
    np.random.shuffle(filelist)
    n = len(filelist)
    ns = int(0.95 * n)
    train_filelist=filelist[:ns]
    val_filelist = filelist[ns: ]
    #test_filelist = filelist[(n-val_n):]
    with open( os.path.join(DATA_ROOT,'train_filelist.txt'),'w' ) as f:
        f.write( '\n'.join(train_filelist) )

    with open(os.path.join(DATA_ROOT,'val_filelist.txt'),'w') as f:
        f.write( '\n'.join(sorted(val_filelist) ))

    sys.exit(0)