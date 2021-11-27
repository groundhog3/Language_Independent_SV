import librosa # used for audio analysis
# import librosa.display # used to displaying audio analysis
# import IPython.display as ipd #toolkit used to use Audio class
# import matplotlib.pyplot as plt #provides a matlab style for plotting
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
from math import log, ceil
import joblib
import time

# tracks time
start_time = time.time()


# ## Loading Audio Files
base = "./Audio/"
enrollment_base = "enrollment sp-Ryan"
enrollment_wav = enrollment_base+".wav"
enrollment_pickle = base+enrollment_base+".pkl"
enrollment = str(base+enrollment_wav)
validation = str(base+"validation sp-Louis #2.wav")


# load audio file and save audio file and sample rate
test1, sr1 = librosa.load(enrollment)
test2, sr2 = librosa.load(validation)

#audio_dur1 = len(test1) / sr1
audio_dur2 = len(test2) / sr2

# frame audio file with 20ms frame and 10ms hop length
frame_length = 500 # frame_size will be 948 frames (20 ms)
hop_length = 256
n_mfcc = 13 #num of mfcc features extracted for each frame


#Normalize both audio files so that amplitudes are within -1 to 1 scale
test1 = librosa.util.normalize(test1)
test2 = librosa.util.normalize(test2)


# extract mfcc features in an array
mfccs1 = librosa.feature.mfcc(y=test1, sr=sr1, n_mfcc = n_mfcc, hop_length=hop_length)
mfccs2 = librosa.feature.mfcc(y=test2, sr=sr2, n_mfcc = n_mfcc,hop_length=hop_length)

audio_dur1 = librosa.get_duration(y=test1, sr=sr1)
audio_dur2 = librosa.get_duration(y=test2, sr=sr2)
ratio = ceil(audio_dur1/audio_dur2)

mfccs2_scaled = np.kron(mfccs2, np.ones((1,ceil(ratio))))
mfccs2_scaled = mfccs2_scaled[:, 0:len(mfccs1[0])]

# ## Speaker Modeling
# I am using Gaussian Mixture Modelling to model extracted MFCC features. 

#GMM of first MFCC feature vector (array)
#gm1 = GaussianMixture(n_components=n_mfcc, random_state = 42, covariance_type='full').fit(mfccs1)
gm1 = joblib.load(enrollment_pickle)

#GMM of second MFCC feature vector (array)
gm2 = GaussianMixture(n_components=n_mfcc, random_state = 42).fit(mfccs2_scaled)

log_like1 = abs(gm1.score(mfccs2_scaled))
log_like2 = abs(gm2.score(mfccs1))


likeh_ratio = abs(log(log_like2)-log(log_like1))
print('Log_LH#1: '+str(log_like1)+'\t'+'Log_LH#2: '+str(log_like2)+'\tLikelihood Ratio: '+str(likeh_ratio))

threshold = 0.025
if(likeh_ratio <= threshold):
    print("{:s} and {:s} are the same speaker. ACCEPT".format(str(enrollment)[33:-1],str(validation)[33:-1]))
    match = True
else:
    print("{:s} and {:s} aren't the same speaker. REJECT".format(str(enrollment)[33:-1],str(validation)[33:-1]))
    match = False
print("Execution Time:--- %s seconds ---" % (time.time() - start_time))

