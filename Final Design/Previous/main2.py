import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
from math import log, ceil
import joblib #used to save gmm model in a file to save on procesing time in pickle format
import time #used to track processing time
from data_class import Speaker_Verification as sv #parent class which contains shared fields
from tabulate import tabulate as tb

# ## Loading Audio Files
base = sv("main").base 
speaker = sv("main").speaker 
enrollment_base = sv("main").enrollment_base 
enrollment_wav = sv("main").enrollment_wav 
enrollment_pickle = sv("main").enrollment_pickle
enrollment = sv("main").enrollment 
validationFiles = sv("main").validationFiles

 # frame audio file with 512ms hop length
hop_length = sv("main").hop_length
n_mfcc = sv("main").n_mfcc #num of mfcc features extracted for each frame


threshold_values = [0.010,0.015,0.020,0.025,0.030]


#this list will contain IDs of speakers. Ex. ["Al", "Bob", ...]
validationSpeakers = [] 
for i in range(len(validationFiles)):
    validationSpeakers.append(sv.getSpeaker(validationFiles[i]))


print('')
for t in range(len(threshold_values)):
    testResults = []
    result_array = []
    result_array2 = [ [ None for i in range(2) ] for j in range(len(validationFiles)+1) ]
    threshold = threshold_values[t]
    for i in range(len(validationFiles)):
        start_time = time.time() # tracks time

        validation = str(base+validationFiles[i])

        # load audio file and save audio file and sample rate
        test1, sr1 = librosa.load(enrollment)
        test2, sr2 = librosa.load(validation)

        #audio_dur1 = len(test1) / sr1
        audio_dur2 = len(test2) / sr2

        #Normalize both audio files so that amplitudes are within -1 to 1 scale
        test1 = librosa.util.normalize(test1)
        test2 = librosa.util.normalize(test2)

        # extract mfcc features in an array
        mfccs1 = librosa.feature.mfcc(y=test1, sr=sr1, n_mfcc = n_mfcc, hop_length=hop_length)
        mfccs2 = librosa.feature.mfcc(y=test2, sr=sr2, n_mfcc = n_mfcc,hop_length=hop_length)


        # get ratio of enrollment audio sample dur to validation audio sample dur
        audio_dur1 = librosa.get_duration(y=test1, sr=sr1)
        audio_dur2 = librosa.get_duration(y=test2, sr=sr2)
        ratio = ceil(audio_dur1/audio_dur2)

        #Since the validation is smaller, it is scaled up in size using the Kronecker product.
        mfccs2_scaled = np.kron(mfccs2, np.ones((1,ceil(ratio))))
        mfccs2_scaled = mfccs2_scaled[:, 0:len(mfccs1[0])]

        # ## Speaker Modeling
        # I am using Gaussian Mixture Modelling to model extracted MFCC features. 

        #GMM of both MFCC feature vectors (array)
        gm1 = joblib.load(enrollment_pickle)
        gm2 = GaussianMixture(n_components=n_mfcc).fit(mfccs2_scaled)
        
        #Calculating Log Likelihood of Both Models
        log_like1 = abs(gm1.score(mfccs2_scaled))
        log_like2 = abs(gm2.score(mfccs1))

        likeh_ratio = abs(log(log_like2)-log(log_like1))
        #print('Log_LH#1: '+str(log_like1)+'\t'+'Log_LH#2: '+str(log_like2)+'\tLikelihood Ratio: '+str(likeh_ratio))

        if(likeh_ratio <= threshold):
            #print("{:s} and {:s} are the same speaker. \nACCEPT".format(str(enrollment),str(validation)))
            match = True
            result_array2[i][0] = 'Accept'
        else:
            #print("{:s} and {:s} aren't the same speaker. \nREJECT".format(str(enrollment),str(validation)))
            match = False
            result_array2[i][0] = 'Reject'
        testResults.append(match)
        #print("Execution Time:--- %s seconds ---\n" % (time.time() - start_time))

    TAR, TRR, FAR, FRR, users, imposters, accuracy, total= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(len(validationFiles)):
        total += 1
        isUser = (speaker==validationSpeakers[i])
        if (isUser):
            if(isUser == testResults[i]):
                TAR += 1
                result_array2[i][1] = 'This is true accept'
            else:
                FRR += 1
                result_array2[i][1] = 'This is false reject'
            users += 1
        else:
            if(isUser == testResults[i]):
                TRR += 1
                result_array2[i][1] = 'This is true reject'
            else:
                FAR += 1
                result_array2[i][1] = 'This is false accept'
            imposters +=1
        if(isUser == testResults[i]):
            accuracy += 1

    print(TAR, TRR, FAR, FRR)
    TAR = TAR/total 
    TRR = TRR/total
    FAR = FAR/total
    FRR = FRR/total
    EER = (FAR+FRR)*0.5
    accuracy = accuracy/len(validationFiles)

    print('Speaker:', speaker, 'Threshold', threshold, 'Hop Length', hop_length)
    print(tb([['General Accuracy', str(round(accuracy*100, 3))+'%'],
            ['Equal Error Rate (EER)', str(round(EER*100, 3))+'%'],
            ['True Accept Rate (TAR)', str(round(TAR*100, 3))+'%'], 
            ['True Reject Rate (TRR)', str(round(TRR*100, 3))+'%'],
            ['False Accept Rate (FAR)', str(round(FAR*100, 3))+'%'],
            ['False Reject Rate (FRR)', str(round(FRR*100, 3))+'%']], 
            headers=['Authentication Measure', 'Percentage']))




