import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
import joblib #used to save gmm model in a file to save on procesing time in pickle format
import time #used to track processing time
from data_class import Speaker_Verification as sv #parent class which contains shared fields
import sys

# tracks time
start_time = time.time()

# ## Loading Audio Files
base = sv("enrollment").base
enrollment_base = sv("enrollment").enrollment_base
enrollment = sv("enrollment").enrollment
validation_length = sv("enrollment").validation_length

enrollmentFiles = sv("enrollment").enrollmentFiles

print('')
print(enrollmentFiles)

# load audio file and save audio file and sample rate
enrollLib, sr1 = librosa.load(enrollment)
enrollLib = librosa.resample(enrollLib, sr1, 22050)
sr1 = 22050

audio_dur1 = len(enrollLib) / sr1
ratio = audio_dur1/validation_length

# frame audio file
hop_length1 = sv("enrollment").hop_length 
n_mfcc = sv("enrollment").n_mfcc #num of mfcc features extracted for each frame

#Normalize both audio files so that amplitudes are within -1 to 1 scale
enrollLib = librosa.util.normalize(enrollLib)

# extract mfcc features in an array
mfccsEnroll = librosa.feature.mfcc(y=enrollLib, sr=sr1, n_mfcc = n_mfcc, hop_length=hop_length1, htk=True)
#mfccsEnroll = zscore(mfccsEnroll, axis=1, ddof=100)

num_of_cols = int((mfccsEnroll.shape[1])/(ratio))+1

i=0
count = 1
while i < int(mfccsEnroll.shape[1]-num_of_cols):
    #print(i, i+num_of_cols-1)
    mfccsSub = mfccsEnroll[:, i:i+num_of_cols]
    gmmPart = GaussianMixture(n_components=1, covariance_type='diag', n_init = 100)
    gmmPart.fit(mfccsSub)

    #save model in pickle format for future use - saves on processing time
    pkl_file = enrollment_base+f" #{count}.pkl"
    joblib.dump(gmmPart, base+pkl_file) 
    print(base+pkl_file +' file created', gmmPart.means_.shape)
    i+=num_of_cols
    count +=1

mfccsSub = mfccsEnroll[:, -num_of_cols:]
gmmPart = GaussianMixture(n_components=1, covariance_type='diag', n_init = 100)
gmmPart.fit(mfccsSub)

#save model in pickle format for future use - saves on processing time
pkl_file = enrollment_base+f" #{count}.pkl"
joblib.dump(gmmPart, base+pkl_file) 
print(base+pkl_file +' file created')
print(gmmPart.means_.shape)

print("Execution Time:--- %s seconds ---" % (time.time() - start_time)) 