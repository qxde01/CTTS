from nnmnkwii.io import hts
import os,argparse,joblib
from glob import glob
import numpy as np
import librosa,pyworld
from nnmnkwii.datasets import FileDataSource,FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from tqdm import tqdm
global DATA_ROOT
from  sklearn.preprocessing import StandardScaler,MinMaxScaler

sample_rate=22050
hop_size=256
frame_period =1000*hop_size/sample_rate   #5
#hop_size=int(frame_period*sample_rate/1000)
frame_shift_in_micro_sec=int(frame_period*10000)

fft_len=1024
mel_dim=80
window='hann'
fmin=50
fmax=7600
_hts=hts.HTSLabelFile(frame_shift_in_micro_sec=frame_shift_in_micro_sec)
def is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return x <= lower or x >= upper


def remove_outlier(x, p_bottom: int = 25, p_top: int = 75):
    """Remove outlier from x."""
    p_bottom = np.percentile(x, p_bottom)
    p_top = np.percentile(x, p_top)

    indices_of_outliers = []
    for ind, value in enumerate(x):
        if is_outlier(value, p_bottom, p_top):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0

    # replace by mean f0.
    x[indices_of_outliers] = np.max(x)
    return x



class MelSource(FileDataSource):
    def __init__(self, use_phone_alignment=False):
        self.use_phone_alignment = use_phone_alignment
        self.binary_dict, self.continuous_dict = hts.load_question_set(os.path.join('misc', "questions-mandarin.hed"))

    def collect_files(self):
        wav_paths = sorted(glob(os.path.join(DATA_ROOT, "wav", "*.wav")))
        if self.use_phone_alignment:
            label_paths = sorted( glob(os.path.join(DATA_ROOT, "label_phone_align", "*.lab")))
        else:
            label_paths = sorted(glob(os.path.join(DATA_ROOT, "label_state_align", "*.lab")))

        if max_num_files is not None and max_num_files > 0:
            return wav_paths[:max_num_files], label_paths[:max_num_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        d,fs=librosa.load(wav_path,sr=sample_rate)
        #audio, _ = librosa.effects.trim(
        #    audio,top_db=config["trim_threshold_in_db"],frame_length=config["trim_frame_size"],hop_length=config["trim_hop_size"] )
        D = librosa.stft( d, n_fft=fft_len, hop_length=hop_size, win_length=None,window=window,   pad_mode="reflect"  )
        S, _ = librosa.magphase(D)
        mel_basis = librosa.filters.mel(sr=fs,n_fft=fft_len,n_mels=mel_dim, fmin=fmin, fmax=fmax )
        #mel_basis=librosa.effects.feature.melspectrogram(d,sr=fs,n_fft=fft_len,hop_length=hop_size,n_mels=mel_dim,fmin=0,htk=True)
        mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T
        #features=features[None,:,:]

        _f0, t = pyworld.dio(d.astype(np.double), fs=sample_rate, f0_ceil=fmax, frame_period=frame_period )
        f0 = pyworld.stonemask(d.astype(np.double), _f0, t, sample_rate)

        # extract energy
        labels = _hts.load(label_path)
        features = fe.linguistic_features(labels, self.binary_dict, self.continuous_dict,add_frame_features=True,subphone_features='coarse_coding',frame_shift_in_micro_sec=frame_shift_in_micro_sec)
        num_frames=labels.num_frames(frame_shift_in_micro_sec=frame_shift_in_micro_sec)
        indices = labels.silence_frame_indices(frame_shift_in_micro_sec=frame_shift_in_micro_sec)
        #print(fs, wav_path, mel.shape[0],labels.num_frames())
        mel = mel[:num_frames]
        if len(f0) >= len(mel):
            f0 = f0[: len(mel)]
        else:
            f0 = np.pad(f0, (0, len(mel) - len(f0)))
        energy = np.sqrt(np.sum(S ** 2, axis=0))
        energy=energy[: len(mel)]

        assert (len(mel) == len(f0) == len(energy)),"error:%s,%s,%s,%s" %(wav_path,len(mel), len(f0), len(energy))

        f0 = remove_outlier(f0)
        energy = remove_outlier(energy)

        if len(indices)>0:
            features = np.delete(features, indices, axis=0)
            mel = np.delete(mel, indices, axis=0)
            f0=np.delete(f0,indices,axis=0)
            energy = np.delete(energy, indices, axis=0)
        #print(features.shape) #
        print(wav_path, mel.shape[0],f0.shape[0], energy.shape[0],features.shape[0],num_frames,len(indices))
        return mel,f0,energy,features

def get_args():
        parser = argparse.ArgumentParser(description="train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--DATA_ROOT', '-data_root', type=str, default='E:/tts/thchs30_250_demo/', help="data root")
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    mel_std = StandardScaler(copy=False)
    f0_std = StandardScaler(copy=False)
    energy_std = StandardScaler(copy=False)
    X_linguistic_mms = MinMaxScaler(feature_range=(0.01, 0.99), copy=False)
    args = get_args()
    max_num_files=100000
    DATA_ROOT = args.DATA_ROOT
    use_phone_alignment = True
    d_mel = os.path.join(DATA_ROOT, "Y_mel")
    d_f0 = os.path.join(DATA_ROOT, "Y_f0")
    d_energy = os.path.join(DATA_ROOT, "Y_energy")
    d_linguistic = os.path.join(DATA_ROOT, "X_linguistic")
    for d in [d_mel,d_f0,d_energy,d_linguistic]:
        if not os.path.exists(d):
            print("mkdirs: {}".format(d))
            os.makedirs(d)
    Y_acoustic_source = MelSource(use_phone_alignment=use_phone_alignment)
    Y_acoustic = FileSourceDataset(Y_acoustic_source)
    _max_len=0
    for idx, (mel,f0,energy,features) in tqdm(enumerate(Y_acoustic)):
        name = os.path.splitext(os.path.basename(Y_acoustic.collected_files[idx][0]))[0]
        mel_std.partial_fit(mel)
        if len(f0)>_max_len:
            _max_len=len(f0)
        f0_std.partial_fit(f0.reshape(-1,1))
        energy_std.partial_fit(f0.reshape(-1,1))
        X_linguistic_mms.partial_fit(features)
        ypath1 = os.path.join(d_mel, name)
        ypath2 = os.path.join(d_f0, name)
        ypath3 = os.path.join(d_energy, name)
        ypath4 = os.path.join(d_linguistic, name)
        # x.tofile(xpath)
        # y.tofile(ypath)
        #np.savez(xpath.replace('.bin', ''), x=x)
        np.save(ypath1, mel)
        np.save(ypath2, f0)
        np.save(ypath3, energy)
        np.save(ypath4, features)
    print('mex seq length:',_max_len)
    joblib.dump(mel_std, DATA_ROOT + 'Y_mel_std.pkl')
    joblib.dump(f0_std, DATA_ROOT + 'Y_f0_std.pkl')
    joblib.dump(energy_std, DATA_ROOT + 'Y_energy_std.pkl')
    joblib.dump(X_linguistic_mms, DATA_ROOT + 'X_linguistic_mms_std.pkl')
