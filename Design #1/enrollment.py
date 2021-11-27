import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
import joblib


# ## Loading Audio Files
base = "./Audio/"
file_name = "enrollment sp-Ryan"
enrollment = str(base+file_name+".wav")


# load audio file and save audio file and sample rate
enrollLib, sr1 = librosa.load(enrollment)

audio_dur1 = len(enrollLib) / sr1

# frame audio file with 20ms frame and 10ms hop length
frame_length = 500 # frame_size will be 948 frames (20 ms)
hop_length = 256
n_mfcc = 13 #num of mfcc features extracted for each frame


#Normalize both audio files so that amplitudes are within -1 to 1 scale
enrollLib = librosa.util.normalize(enrollLib)

# extract mfcc features in an array
mfccsEnroll = librosa.feature.mfcc(y=enrollLib, sr=sr1, n_mfcc = n_mfcc, hop_length=hop_length)


gmmEnroll = GaussianMixture(n_components=n_mfcc, random_state = 42, covariance_type='full').fit(mfccsEnroll)

pkl_file = file_name+".pkl"
joblib.dump(gmmEnroll, base+pkl_file) 
print(base+pkl_file +' file created')


