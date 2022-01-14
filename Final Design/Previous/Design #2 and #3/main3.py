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

threshold_values = [0.05]

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
        mfccs2 = librosa.feature.mfcc(y=test2, sr=sr2, n_mfcc = n_mfcc, hop_length=hop_length)

        #mfccs1 = zscore(mfccs1, axis=1, ddof=1)
        #mfccs2 = zscore(mfccs2, axis=1, ddof=1)

        # get ratio of enrollment audio sample dur to validation audio sample dur
        audio_dur1 = librosa.get_duration(y=test1, sr=sr1)
        audio_dur2 = librosa.get_duration(y=test2, sr=sr2)
        ratio = ceil(audio_dur1/audio_dur2)


        #GMM of both MFCC feature vectors (array)
        gm1 = joblib.load(enrollment_pickle)
        gm2 = GaussianMixture(n_components=n_mfcc).fit(mfccs2)
        
        #print(str(validation))
        num_of_cols = int((mfccs2.shape[1]))

        #Calculating Log Likelihood of Both Models
        log_like1 = abs(gm1.score(mfccs2))
        log_like2 = 0
        #log_like2 = abs(gm2.score(mfccs1[i:i+num_of_cols]))

        l, counter= 0, 0
        while l < int(mfccs1.shape[1]-num_of_cols-1):
            mfccsSub = mfccs1[:, l:l+num_of_cols]
            if(mfccsSub.shape[1] != num_of_cols):
                print('true')
                np.resize(mfccsSub,(n_mfcc,num_of_cols))
            log_like2 += abs(gm2.score(mfccsSub))
            counter += 1
            l+=num_of_cols
        
        log_like2 /= counter

        likeh_ratio = abs(log(log_like2)-log(log_like1))
        print('Log_LH#1: '+str(log_like1)+'\t'+'Log_LH#2: '+str(log_like2)+'\tLikelihood Ratio: '+str(likeh_ratio))

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

    TA, TR, FA, FR, accuracy, total= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(len(validationFiles)):
        isUser = (speaker==validationSpeakers[i])
        if (isUser):
            if(isUser == testResults[i]):
                TA += 1
                result_array2[i][1] = 'This is true accept'
            else:
                FR += 1
                result_array2[i][1] = 'This is false reject'
        else:
            if(isUser == testResults[i]):
                TR += 1
                result_array2[i][1] = 'This is true reject'
            else:
                FA += 1
                result_array2[i][1] = 'This is false accept'
           
    print(TA, TR, FA, FR)
    TAR = TA/(TA+FR) #TP/TP+FN
    TRR = TR/(TR+FA) #TN/TN+FP
    FAR = FA/(FA+TR) #FP/FP+TN
    FRR = FR/(FR+TA) #FN/FN+TP
    EER = (FAR+FRR)*0.5
    accuracy = (TA+TR)/(TA+TR+FA+FR)
    #accuracy = accuracy/len(validationFiles)

    print('Speaker:', speaker, 'Threshold', threshold, 'Hop Length', hop_length)
    print(tb([['General Accuracy', str(round(accuracy*100, 3))+'%'],
            ['Equal Error Rate (EER)', str(round(EER*100, 3))+'%'],
            ['True Accept Rate (TAR)', str(round(TAR*100, 3))+'%'], 
            ['True Reject Rate (TRR)', str(round(TRR*100, 3))+'%'],
            ['False Accept Rate (FAR)', str(round(FAR*100, 3))+'%'],
            ['False Reject Rate (FRR)', str(round(FRR*100, 3))+'%']], 
            headers=['Authentication Measure', 'Percentage']))




