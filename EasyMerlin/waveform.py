import numpy as np
from scipy.io import wavfile
import os,joblib,time
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
#import tensorflow.keras.backend as K
#K.set_floatx('float16')
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from nnmnkwii.io import hts
from nnmnkwii import paramgen
from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.frontend import merlin as fe
import pyworld
import pysptk
from mtts.mandarin_frontend import txt2label
from config import *

binary_dict, continuous_dict = hts.load_question_set(hed_path)
X_acoustic_mms = joblib.load(acoustic_mms_path)
Y_acoustic_std = joblib.load(acoustic_std_path)
X_duration_mms = joblib.load(duration_mms_path)
Y_duration_std = joblib.load(duration_std_path)
duration_model = load_model(duration_model_path)
acoustic_model = load_model(acoustic_model_path)


#duration_model.summary()
#acoustic_model.summary()

def gen_parameters(y_predicted):
    # Number of time frames
    T = y_predicted.shape[0]
    # Split acoustic features
    mgc = y_predicted[:, :lf0_start_idx]
    lf0 = y_predicted[:, lf0_start_idx:vuv_start_idx]
    vuv = y_predicted[:, vuv_start_idx]
    bap = y_predicted[:, bap_start_idx:]
    # Perform MLPG
    #Y_acoustic_std.var_
    mgc_variances=np.tile(Y_acoustic_std.var_[:lf0_start_idx], (T, 1))
    #mgc_variances = np.tile(Y_var[ty][:lf0_start_idx], (T, 1))
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(Y_acoustic_std.var_[lf0_start_idx:vuv_start_idx], (T, 1))
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(Y_acoustic_std.var_[bap_start_idx:], (T, 1))
    bap = paramgen.mlpg(bap, bap_variances, windows)
    return mgc, lf0, vuv, bap

def gen_waveform(y_predicted, do_postfilter=False):
    y_predicted = trim_zeros_frames(y_predicted)
    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted)
    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    #print(bap.shape)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    return generated_waveform

def gen_duration(hts_labels, duration_model):
    # Linguistic features for duration
    #hts_labels = hts.load(label_path)
    duration_linguistic_features = fe.linguistic_features(hts_labels,binary_dict, continuous_dict,add_frame_features=False,subphone_features=None).astype(np.float32)
    # Apply normalization
    duration_linguistic_features =X_duration_mms.transform(duration_linguistic_features)
    if len(duration_model.inputs[0].shape)==3:
        # seq2seq
        n1,n2=duration_linguistic_features.shape
        duration_linguistic_features=duration_linguistic_features.reshape(1,n1,n2)
    duration_predicted=duration_model.predict(duration_linguistic_features)
    if len(duration_predicted.shape)==3:
        duration_predicted=duration_predicted.reshape(duration_predicted.shape[1],duration_predicted.shape[2])
    duration_predicted=Y_duration_std.inverse_transform(duration_predicted)
    duration_predicted = np.round(duration_predicted)
    # Set minimum state duration to 1
    duration_predicted[duration_predicted <= 0] = 1
    hts_labels.set_durations(duration_predicted)
    return hts_labels


def test_one_utt(txt, duration_model, acoustic_model, post_filter=True):
    # Predict durations
    #txt = '中华人民共和国中央人民政府今天成立了'
    label=txt2label(txt)
    #hts_labels = hts.load(path=label_path)
    hts_labels = hts.load(lines=label)
    duration_modified_hts_labels = gen_duration(hts_labels, duration_model)
    # Linguistic features
    linguistic_features = fe.linguistic_features(duration_modified_hts_labels, binary_dict, continuous_dict,add_frame_features=True,subphone_features="coarse_coding")
    # Trim silences
    indices = duration_modified_hts_labels.silence_frame_indices()
    linguistic_features = np.delete(linguistic_features, indices, axis=0)
    linguistic_features=X_acoustic_mms.transform(linguistic_features)
    if len(acoustic_model.inputs[0].shape) == 3:
        # RNN
        n1, n2 = linguistic_features.shape
        linguistic_features = linguistic_features.reshape(1, n1, n2)
        acoustic_predicted = acoustic_model.predict(linguistic_features)
        acoustic_predicted = acoustic_predicted.reshape(acoustic_predicted.shape[1], acoustic_predicted.shape[2])
    else:
        acoustic_predicted = acoustic_model.predict(linguistic_features)

    acoustic_predicted = Y_acoustic_std.inverse_transform(acoustic_predicted)
    out=gen_waveform(acoustic_predicted, post_filter)
    out=out.astype(np.int16)

    return out
if __name__ == '__main__':
    test=pd.read_csv('misc/test01.csv')
    n=test.shape[0]
    save_dir = os.path.join("resu" )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cost_times=[]
    for i in range(0,n):
        txt=test.text[i]
        name = str(test.name[i])
        t0=time.time()
        waveform = test_one_utt(txt, duration_model, acoustic_model,post_filter=True)
        wavfile.write(os.path.join(save_dir, name+'-'+txt[:5] + "-%s.wav" %len(txt)), rate=fs, data=waveform )


