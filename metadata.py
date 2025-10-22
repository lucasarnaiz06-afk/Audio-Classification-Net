import json
from pathlib import Path
import pandas as pd

download_path = Path('/Users/lucasarnaiz/Desktop/p-ai/audioMNIST')  # find path for data
metadata_file = download_path / 'audioMNIST_meta.txt' #find metadata file

with open(metadata_file, 'r') as file: 
    data = json.load(file) #load metadata file

df_meta = pd.DataFrame.from_dict(data, orient='index') #convert to dataframe
df_meta.index.name = 'speaker_id' #name index column
df_meta.reset_index(inplace=True) #reset index to make speaker_id a column

df_meta.head() #show first 5 rows of dataframe

audio_paths = []
for folder in sorted(download_path.iterdir()): #iterate through folders in download path
    if folder.is_dir() and folder.name.isdigit(): #check if folder is a directory and its name is a digit
        speaker_id = folder.name #get speaker_id from folder name
        for wav_file in folder.glob('*.wav'): #iterate through .wav files in folder
            audio_paths.append({ 
                'speaker_id' : speaker_id,
                'relative_path' : str(wav_file.relative_to(download_path))
                }) #append a dictionary containing the speaker_id and relative path to our list

df_audio = pd.DataFrame(audio_paths) #convert list of dictionaries to dataframe
df_audio.head() #show first 5 rows of dataframe

df = df_audio.merge(df_meta, on='speaker_id', how='left') #merge the two dataframes, so that we have metadata for each audio file
df['classID'] = df['speaker_id'].astype(int) - 1  # Convert to 0-59 range
df.head() #show first 5 rows of dataframe



X = df['relative_path'] #features are the relative paths to the audio files
y = df['speaker_id'] #labels are the speaker ids

# print(X.head())
# print(y.head())
