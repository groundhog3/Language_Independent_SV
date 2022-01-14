import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
import joblib #used to save gmm model in a file to save on procesing time in pickle format
import time #used to track processing time
from data_class import Speaker_Verification as sv #parent class which contains shared fields

# tracks time
start_time = time.time()

# ## Loading Audio Files
base = sv("enrollment").base
enrollment_base = sv("enrollment").enrollment_base
enrollment = sv("enrollment").enrollment

# load audio file and save audio file and sample rate
enrollLib, sr1 = librosa.load(enrollment)

audio_dur1 = len(enrollLib) / sr1

# frame audio file
hop_length1 = sv("enrollment").hop_length 
n_mfcc = sv("enrollment").n_mfcc #num of mfcc features extracted for each frame

#Normalize both audio files so that amplitudes are within -1 to 1 scale
enrollLib = librosa.util.normalize(enrollLib)

# extract mfcc features in an array
mfccsEnroll = librosa.feature.mfcc(y=enrollLib, sr=sr1, n_mfcc = n_mfcc, hop_length=hop_length1)

#create Gaussian Mixture Model
gmmEnroll = GaussianMixture(n_components=n_mfcc, random_state = 42, covariance_type='full').fit(mfccsEnroll)

#save model in pickle format for future use - saves on processing time
pkl_file = enrollment_base+".pkl"
joblib.dump(gmmEnroll, base+pkl_file) 
print(base+pkl_file +' file created')
print(gmmEnroll.means_.shape)

print("Execution Time:--- %s seconds ---" % (time.time() - start_time))