import math, random
import torch
import librosa
import torchaudio
from torchaudio import transforms
import numpy as np
from torch.utils.data import Dataset
from IPython.display import Audio

class AudioUtil():
    # load an audio file with librosa
    @staticmethod
    def open_audio_librosa(audio_file):
        sig, sr = librosa.load(audio_file, sr=None)
        sig = torch.from_numpy(sig).unsqueeze(0)
        return (sig, sr)
    
    # load an audio file with torchaudio
    @staticmethod
    def open_audio_torchaudio(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    #convert audio to desired number of channels
    @staticmethod
    def rechannel(aud, num_channels):
        sig, sr = aud
        if (sig.shape[0] == num_channels):
            return aud
        if (num_channels == 1):
            #convert from stero to mono by choosing only the first audio channel
            resig = sig[:1, :]
        else:
            #convert from mono to stereo by duplicating the audio channel
            resig = torch.cat([sig, sig])
        return ((resig, sr))
    
    #resample the audio to a desired sampling rate, one channel at a time
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr): # if the same sampling rate, return the original audio
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
       
    #pad/truncate the signal to a fixed length 'max_ms' in milliseconds
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
        # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig, sr)
    
    #shift the signal in time by a random amount, within the limits of "shift_limit", which is a percentage of the total signal length
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud 
        _, sig_len = sig.shape # get the length of the signal
        shift_amt = int(random.random() * shift_limit * sig_len) # calculate the amount to shift
        return (sig.roll(shift_amt), sr) # roll the signal by the shift amount
    
    #generate a mel spectrogram from a raw audio signal
    #n_mels: number of mel frequencies, or the height resolution of the spectrogram
    #n_fft: number of data points used in each block for the fast fourier transform. The higher this is, the higher the frequency resolution but the lower the time resolution
    #hop_len: number of audio frames between STFT columns. The higher this is, the lower the time resolution
    @staticmethod
    def mel_spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    #generate masks for data augmentation, by masking out random sections of the mel spectrogram
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape #get the shape of the spectrogram
        mask_value = spec.mean() #value to replace the masked sections with
        aug_spec = spec #initialize the augmented spectrogram

        freq_mask_param = max_mask_pct * n_mels #calculate the maximum width of the frequency mask
        for _ in range(n_freq_masks): #apply the frequency masks
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value) 

        time_mask_param = max_mask_pct * n_steps #calculate the maximum width of the time mask
        for _ in range(n_time_masks): #apply the time masks
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec 
