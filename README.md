# Audio-Classification-Net
Code designed to classify the audio recordings from the audio-MINST database on kaggle: https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist.
This code includes 6 files: 
1) audio_utilities.py, for loading audio, rechanneling audio, resampling audio, truncating and padding audio, time shifting audio, generating MEL spectograms from audio files, and generated augmented MEL spectograms with time and frequency masks.
   This file essentially holds the functions for creating useful data from simple .wav (or other file type) audio files. 
2) GPU_CHOOSE.py, for setting the "device" variable to the proper GPU/CPU for training. I trained this model on an apple silicon GPU.
3) metadata.py, which uses the provided metadata from kaggle to create a dataframe containing the given metadata, the speaker ID, and the path for each audio file. I added the speaker ID as a column because that is the classification type that I want my network to classify on.
4) data_loader.py, which defines a PyTorch dataset class that allows use to use the functions defined in audio_utilities on audio datasets.
5) CNN1.py, which contains the architecture for a CNN that will use convolution to extract feature maps from mel spectograms, and then use a classifier architecture to make predictions for what speaker the audio is from
6) Lucas_Arnaiz_W4-5_InitialCNN.ipynb, which is my jupyter notebook that uses the prior python files to train a model, and then generate helpful graphs to analyze the model's performance and outputs. 
