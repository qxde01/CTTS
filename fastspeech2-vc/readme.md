# 基于fastspeech2的音色迁移
根据[TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)修改

### data 训练数据结构 
```
dump/
├── baker_mapper.json
├── stats_energy.npy
├── stats_f0.npy
├── stats.npy
├── train_utt_ids.npy
└── valid_utt_ids.npy
├── train
│   ├── durations
│   │   ├── XXXX_0001-durations.npy
│   ├── ids
│   │   ├── XXXX_0001-ids.npy
│   ├── norm-feats
│   │   ├── XXXX_0001-norm-feats.npy
│   ├── raw-energies
│   │   ├── XXXX_0001-raw-energy.npy
│   ├── raw-f0
│   │   ├── XXXX_0001-raw-f0.npy
│   ├── raw-feats
│   │   ├── XXXX_0001-raw-feats.npy
│   ├── spk-emb
│   │   ├── XXXX_0001-spk-emb.npy
│   │   ├── XXXX_0002-spk-emb.npy
│   └── wavs
│       ├── XXXX_0001-wave.npy
├── valid
│   ├── durations
│   │   ├── XXXX_0001-durations.npy
│   ├── ids
│   │   ├── XXXX_0001-ids.npy
│   ├── norm-feats
│   │   ├── XXXX_0001-norm-feats.npy
│   ├── raw-energies
│   │   ├── XXXX_0001-raw-energy.npy
│   ├── raw-f0
│   │   ├── XXXX_0001-raw-f0.npy
│   ├── raw-feats
│   │   ├── XXXX_0001-raw-feats.npy
│   ├── spk-emb
│   │   ├── XXXX_0001-spk-emb.npy
│   │   ├── XXXX_0002-spk-emb.npy
│   └── wavs
│       ├── XXXX_0001-wave.npy
```
### tacotron2 多speakers 时长对齐
### fastspeech2 多speakers TTS-Mel合成器训练
### fastspeech2-vc 多speakers VC合成器训练
### hifigan 声码器训练
### deepspeaker speakerEmbedding 提取器训练

### 参考
* [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)
* [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
* [LightSpeech](https://github.com/nmfisher/TensorFlowTTS)
* [RepVGG](https://github.com/hoangthang1607/RepVGG-Tensorflow-2/blob/main/repvgg.py)
* [ECAPA_TDNN](https://github.com/wenet-e2e/wespeaker)

