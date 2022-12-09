import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)
# from typing import Tuple
# from python_speech_features import fbank
MILLISECONDS_TO_SECONDS = 0.001
epsilon = 1.1920928955078125e-07


def tf_magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    #if tf.shape(frames)[1] > NFFT:
        #tf.print('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.'%( int(tf.shape(frames)[1]), NFFT ))
    complex_spec = tf.signal.rfft(frames, [NFFT])
    return tf.abs(complex_spec)


def tf_preemphasis(
        signal: tf.Tensor,
        coeff=0.97,
):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


def tf_powspec(frames: tf.Tensor, NFFT: int) -> tf.Tensor:
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * tf.square(tf_magspec(frames, NFFT))


def tf_slice_signal(signal: tf.Tensor, frame_length: int, frame_step: int, nfft: int = 512) -> tf.Tensor:
    # if self.center:
    #    signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
    window = tf.signal.hann_window(frame_length, periodic=True, dtype=tf.float32)
    signal=tf.concat([signal, tf.zeros(frame_length-tf.size(signal) % frame_length)],axis=0)
    #print("window:",window.shape)
    #left_pad = int((nfft - frame_length) // 2)
    #right_pad = int(nfft - frame_length - left_pad)
    #window = tf.pad(window, [[left_pad, right_pad]])
    framed_signals = tf.signal.frame(signal, frame_length=frame_length, frame_step=frame_step)
    #print("frame:",framed_signals.shape)
    row_means = tf.reduce_mean(framed_signals, axis=1, keepdims=True)  # size (m, 1)
    framed_signals = framed_signals - row_means
    framed_signals *= window
    return framed_signals


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * tf_log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def tf_filterbanks(nfilt: int = 80, nfft: int = 512, samplerate: int = 16000, lowfreq: int = 0, highfreq: int = None) -> tf.Tensor:
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = tf_hz2mel(lowfreq)
    highmel = tf_hz2mel(highfreq)
    # melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    melpoints = tf.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = tf.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    fbank = tf.Variable(tf.zeros([nfilt, nfft // 2 + 1]))
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            tmp1 = (float(i) - bin[j]) / (bin[j + 1] - bin[j])
            fbank[j, i].assign(tmp1)
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            tmp2 = (bin[j + 2] - float(i) ) / (bin[j + 2] - bin[j + 1])
            fbank[j, i].assign(tmp2)
    return tf.convert_to_tensor(fbank,dtype=tf.float32)


#@tf.function
def tf_fbank(signal: tf.Tensor, samplerate: int = 16000, winlen: float = 0.025, winstep: float = 0.01,
             nfilt: int = 80, nfft: int = 512, lowfreq: int = 20, highfreq: int = None, preemph: float = 0.97) -> tf.Tensor:
    # winfunc=lambda x:numpy.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = tf_preemphasis(signal, preemph)
    # frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc) #(1130, 400)
    frames = tf_slice_signal(signal, frame_length=int(winlen * samplerate), frame_step=int(winstep * samplerate), nfft=nfft)
    pspec = tf_powspec(frames, nfft)

    # this stores the total energy in each frame
    #energy = tf.reduce_sum(pspec, axis=1)
    # if energy is zero, we get problems with log
    # energy = tf.where(energy == 0, epsilon, energy)

    fb = tf_filterbanks(nfilt=nfilt, nfft=nfft, samplerate=samplerate, lowfreq=lowfreq, highfreq=highfreq)
    feat = tf.matmul(pspec, tf.transpose(fb))
    #feat = tf.matmul(pspec, fb)
    #print("-->banks:",fb.shape,fb.numpy().sum())

    #np.save(file='data/fbank_80X257.npy',arr=fb.numpy().T)
    # compute the filterbank energies

    # feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    feat = tf.where(feat <= epsilon, epsilon, feat)
    feat = tf.math.log(feat)
    return feat

#@tf.function
def tf_fbank2(signal: tf.Tensor, samplerate: int = 16000, winlen: float = 0.025, winstep: float = 0.01,
              nfft: int = 512, fb: tf.Tensor = None, preemph: float = 0.97) -> tf.Tensor:


    #highfreq = highfreq or samplerate / 2
    signal = tf_preemphasis(signal, preemph)
    # frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc) #(1130, 400)
    frames = tf_slice_signal(signal, frame_length=int(winlen * samplerate), frame_step=int(winstep * samplerate), nfft=nfft)
    pspec = tf_powspec(frames, nfft)

    # this stores the total energy in each frame
    #energy = tf.reduce_sum(pspec, axis=1)
    # if energy is zero, we get problems with log
    # energy = tf.where(energy == 0, epsilon, energy)

    #fb = tf_filterbanks(nfilt=nfilt, nfft=nfft, samplerate=samplerate, lowfreq=lowfreq, highfreq=highfreq)
    #feat = tf.matmul(pspec, tf.transpose(fb))
    feat = tf.matmul(pspec, fb)
    #print("-->banks:",fb.shape,fb.numpy().sum())

    #np.save(file='data/fbank_80X257.npy',arr=fb.numpy().T)
    # compute the filterbank energies

    # feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    #print(feat.numpy().max(),feat.numpy().min())
    feat = tf.where(feat == 0.0, epsilon, feat)
    feat = tf.math.log(feat)
    #feat= tf_log10(feat)
    #row_means = tf.reduce_mean(feat, axis=1, keepdims=True)  # size (m, 1)
    #feat = feat - row_means
    return feat

def visualize(wave, x1, x2,name='mel'):
    import matplotlib.pyplot as plt
    print(x1.shape, x2.shape)
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    # ax4 = fig.add_subplot(414)
    im = ax1.imshow(np.rot90(x1), aspect="auto", interpolation="none")
    ax1.set_title("TF-logfbank")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
    im = ax2.imshow(np.rot90(x2), aspect="auto", interpolation="none")
    ax2.set_title("fbank")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)

    ax3.set_title(f"Audio")
    # im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
    plt.plot(wave)
    # plt.show()
    plt.savefig('%s.png' %name )
    # fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
    return None

def test(signal):
    fb = np.load('data/fbank_80X257.npy')
    fb = tf.convert_to_tensor(fb, dtype=tf.float32)
    for i in range(0,5):
        s=signal[:(i*512*4+8*16000)]
        #feat = tf_fbank(s, samplerate=16000, winlen=0.025, winstep=0.01,nfilt=80, nfft=512, lowfreq=20, highfreq=None, preemph=0.97)
        feat = tf_fbank2(s, samplerate=16000, winlen=0.025, winstep=0.01, nfft=512, fb=fb, preemph=0.97)
        feat1, energy1 = fbank(s, nfilt=80, winfunc=np.hamming)
        print("  ==>:",i,len(s),len(s)/512,feat.shape)
        feat1 = np.log(feat1)
        visualize(s, feat, feat1, name='mel6_%s' %i)

if __name__ == '__main__':
    import soundfile as sf
    import numpy as np
    from python_speech_features import fbank

    signal, rate = sf.read("E:/story-girl-wav/StoryGirl/storygirl001_11-10-8-02.wav")
    #pad=np.zeros(512-len(signal) % 512)
    #signal=np.hstack([signal,pad])
    signal = signal.astype(np.float32)
    test(signal)
    #feat  = tf_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
    #                        nfilt=80, nfft=512, lowfreq=20, highfreq=None, preemph=0.97)
    #feat1, energy1 = fbank(signal, nfilt=80, winfunc=np.hamming)
    #feat1=np.log(feat1)
    #print(feat.shape, feat1.shape)
    #print(feat)
    #print(feat1)
    #visualize(signal, feat, feat1)
