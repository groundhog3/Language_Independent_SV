import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
from math import log, ceil
import joblib #used to save gmm model in a file to save on procesing time in pickle format
import time #used to track processing time
from data_class import Speaker_Verification as sv #parent class which contains shared fields
from tabulate import tabulate as tb
from scipy.stats.distributions import chi2
import sys

exec(open("enrollment3.py").read()) #call enrollment script
print('\n'+ sys.argv[0] + ' started.')
# ## Loading Audio Files
base = sv("main").base 
speaker = sv("main").speaker 
enrollment_base = sv("main").enrollment_base 
enrollment_wav = sv("main").enrollment_wav 
enrollment_pickle = sv("main").enrollment_pickle
enrollment = sv("main").enrollment 
validationFiles = sv("main").validationFiles
language = str(sv("main").language)[:-1]
ratio = sv("main").ratio

 # frame audio file with a specified hop length
hop_length = sv("main").hop_length
n_mfcc = sv("main").n_mfcc #num of mfcc features extracted for each frame

threshold_values = []
for i in range(100, 1, -1):
    threshold_values.append(float(10**(-i)))

threshold_values.extend(list(np.around(np.arange(0.02, 0.5, 0.01), decimals=3)))
threshold_values.extend([0.9,1])
threshold_values.insert(0,0)

#threshold_values = [0, 0.01, 0.02, 0.04, 0.05, 0.07, 0.01, 0.2, 0.3,  1]

likeh_ratio_array = []

#this list will contain IDs of speakers. Ex. ["Al", "Bob", ...]
validationSpeakers = [] 
for i in range(len(validationFiles)):
    validationSpeakers.append(sv.getSpeaker(validationFiles[i]))

print('')

testResults = [ [ None for i in range(len(validationFiles)) ] for j in range(len(threshold_values)) ]
result_array = []
result_array2 = [ [ None for i in range(2) ] for j in range(len(validationFiles)) ]

for i in range(len(validationFiles)):
    start_time = time.time() # tracks time

    validation = str(base+validationFiles[i])

    # load audio file and save audio file and sample rate
    test2, sr2 = librosa.load(validation)
    test2 = librosa.resample(test2, sr2, 22050)
    sr2 = 22050

    test2 = librosa.util.normalize(test2)

    # extract mfcc features in an array
    mfccs2 = librosa.feature.mfcc(y=test2, sr=sr2, n_mfcc = n_mfcc,hop_length=hop_length)

  
    #GMM of both MFCC feature vectors (array)
    gm1 = joblib.load(enrollment_pickle)
    gm2 = GaussianMixture(n_components=1, covariance_type='diag').fit(mfccs2)
    
    #print(str(validation))
    num_of_cols = int((mfccs2.shape[1]))

    #Calculating Log Likelihood of Both Models
    log_like1 = ((gm1.score(mfccs2))) 
    log_like2 = ((gm2.score(mfccs2))) 
    
    #chi-square test of independence used to compare two models
    #Output: P-value
    Λ = 2 * (log_like2-log_like1)
    df = int(2*(gm2.means_.shape[1])/ratio+1)  # df = degrees of freedom model 1 has over model 2
    df = 2 #
    likeh_ratio = chi2.sf(Λ, df) # this is a p value

    likeh_ratio_array.append(likeh_ratio)
    
    print('Log_LH#1: '+str(log_like1)+'\t'+'Log_LH#2: '+str(log_like2)+'\tLikelihood Ratio: '+str(likeh_ratio))
    
for t in range(len(threshold_values)):
    threshold = threshold_values[t]
    for e in range(len(validationFiles)):
        if(likeh_ratio_array[e] >= threshold):
            #print("{:s} and {:s} are the same speaker. ACCEPT".format(str(speaker),str(validationSpeakers[e])))
            match = True
            result_array2[e][0] = 'Accept'
        else:
            #print("{:s} and {:s} aren't the same speaker. REJECT".format(str(speaker),str(validationSpeakers[e])))
            match = False
            result_array2[e][0] = 'Reject'
        testResults[t][e] = match

print('\nSpeaker:', speaker, '\tHop Length:', hop_length, '\tLanguage:', language)
for t in range(len(threshold_values)):
    TA, TR, FA, FR, accuracy, total= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(validationFiles)):
        isUser = (speaker==validationSpeakers[i])
        if (isUser): #this is when the validation speaker is a REGISTERED user
            if(isUser == testResults[t][i]):
                TA += 1
                result_array2[i][1] = 'This is a true accept.'
            else:
                FR += 1
                result_array2[i][1] = 'This is a false reject.'
        else: #this is when the validation speaker is an IMPOSTER
            if(isUser == testResults[t][i]): 
                TR += 1
                result_array2[i][1] = 'This is a true reject.'  
            else:
                FA += 1
                result_array2[i][1] = 'This is a false accept.'      
    
    TAR = TA/(TA+FR) #True Accept Rate = TP/TP+FN
    TRR = TR/(TR+FA) #True Reject Rate = TN/TN+FP
    FAR = FA/(FA+TR) #False Accept Rate = FP/FP+TN
    FRR = FR/(FR+TA) #False Reject Rate = FN/FN+TP
    EER = (FAR+FRR)*0.5
    accuracy = (TA+TR)/(TA+TR+FA+FR)
    
    # Print threshold, TA, TR, FA, FR, ACC%, EER%, FAR, TAR, FRR
    print(threshold_values[t], TA, TR, FA, FR, str(round(accuracy*100, 4)), str(round(EER*100, 4)), str(round(FAR, 5)), str(round(TAR, 5)), str(round(FRR, 5)))
   
    '''print('Speaker:', speaker, 'Threshold', threshold_values[t], 'Hop Length', hop_length)
    print(tb([['General Accuracy', str(round(accuracy*100, 3))+'%'],
            ['Equal Error Rate (EER)', str(round(EER*100, 3))+'%'],
            ['True Accept Rate (TAR)', str(round(TAR*100, 3))+'%'], 
            ['True Reject Rate (TRR)', str(round(TRR*100, 3))+'%'],
            ['False Accept Rate (FAR)', str(round(FAR*100, 3))+'%'],
            ['False Reject Rate (FRR)', str(round(FRR*100, 3))+'%']], 
            headers=['Authentication Measure', 'Percentage']))'''



