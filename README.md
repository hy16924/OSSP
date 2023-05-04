## Perturbation AUTOVC: Voice Conversion from Perturbation and Autoencoder Loss

This repository provides a PyTorch implementation of Perturbation AUTOVC.

### Audio Demo

The audio demo for Perturbation AUTOVC can be found...

### Dependencies
- Python 3
- Numpy
- PyTorch...
- librosa
- tqdm
- parselmouth
- torchaudio
- omegaconf
- HiFi-GAN vocoder
  for more information, please refer to **HiFi-GAN github 주소 입력**
  
 ### Pre-traied HiFi-GAN
 
 Download pretrained hifi-gan config and checkpoint
 from [hifi-gan](http://github.com/jik876/hifi-gan)
 to `./configs/hifi-gan/UNIVERSAL_V1` **수정~**
 
 ### Speaker Encoder
 
 We use the ECAPA-TDNN as a speaker encoder.
 Download pretrained ecapa-tdnn checkpoint from []
  
 ### Pre-trained models
 | Perturbation AUTOVC | Speaker Encoder | HiFi-GAN Vocoder |
 |---------------------|-----------------|------------------|
 | [link](???????????????????????) |
 
 ### Datasets
 
 Datasets used when training are:
 - VCTK:
    - CSTR VCTK Corpus: English Multi speaker Corpus for CSTR Voice Coloning Toolkit
    - https://datashare.ed.ac.uk/handle/10283/2651

 Place datasets at `datasets/wavs/`
 
 ### 1. Perturbation audio and Mel-spectrogram to waveform
 
 If you prefer `praat-parselmouth` **check**, `python make_metadata.py'
 
 `python make_metadata.py`
 
 ### 2. Train model
 
 `python main.py`
 
 ### Conversion
 
 `python conversion.py`
 
 ### Generator

`python conversion.py`

