import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from audio_utilities import AudioUtil

class SoundDS(Dataset):
    # Initialize the dataset 
    def __init__(self, df, data_path): 
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self): # return the size of the dataset
        return len(self.df)
    
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + "/" + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open_audio_torchaudio(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.mel_spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return (aug_sgram, class_id)
        
# myds = SoundDS(df, data_path)

# # Random split of 80:20 between training and validation
# num_items = len(myds)
# num_train = round(num_items * 0.8)
# num_val = num_items - num_train
# train_ds, val_ds = random_split(myds, [num_train, num_val])

# # Create training and validation data loaders
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
# val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
