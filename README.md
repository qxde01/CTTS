# CTTS
Chinese text to speech ,中文语音合成
## EasyMerlin
  综合github上的几个项目对[merlin](https://github.com/CSTR-Edinburgh/merlin)的简化，模型基于`tf2.keras`
### 使用mel特征替换mgc特征，可以使用melgan等神经网络声码器,可以提升音质
对于`sample_rate=22050,hop_size=256`的melgan声码器，merlin前端特征对齐参数是：
```
frame_period =1000*hop_size/sample_rate   
frame_shift_in_micro_sec=int(frame_period*10000)
```
## fastspeech2
 基于fastspech2的音色克隆。包括tacotron2、fastspeech2、hifi-gan训练，整个过程基于[TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)。

### 参考
* [merlin](https://github.com/CSTR-Edinburgh/merlin)
* [merlin-baseline](https://github.com/r9y9/icassp2020-espnet-tts-merlin-baseline)
* [MTTS](https://github.com/Jackiexiao/MTTS)
* [中文文本正则化](https://github.com/speechio/chinese_text_normalization)
