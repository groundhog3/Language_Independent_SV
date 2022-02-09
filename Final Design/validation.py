import librosa # used for audio analysis
import numpy as np #used to create large, multi-dimensional arrays 
from sklearn.mixture import GaussianMixture #used to create GuassianMixture models
from math import log, ceil
import joblib #used to save gmm model in a file to save on procesing time in pickle format
import time #used to track processing time
from data_class import Speaker_Verification as sv #parent class which contains shared fields
from scipy.stats.distributions import chi2
from scipy.stats import f as fValue
import noisereduce as nr
import sys

def f_test(x, y, log1, log2):
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1- fValue.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

exec(open("enrollment4.py").read()) #call enrollment script

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
enrollmentFiles = sv("main").enrollmentFiles


 # frame audio file with a specified hop length
hop_length = sv("main").hop_length
n_mfcc = sv("main").n_mfcc #num of mfcc features extracted for each frame

threshold_values = []
threshold_values.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
threshold_values.extend(list(np.around(np.arange(0.1, 1, 0.02), decimals=4)))
threshold_values.extend([1])

likeh_ratio_array = []

#this list will contain IDs of speakers. Ex. ["Al", "Bob", ...]
validationSpeakers = [] 
for i in range(len(validationFiles)):
    validationSpeakers.append(sv.getSpeaker(validationFiles[i]))

print('')
print(enrollmentFiles)

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
    mfccs2 = librosa.feature.mfcc(y=test2, sr=sr2, n_mfcc = n_mfcc,hop_length=hop_length, htk=True)
    #mfccs2 = zscore(mfccs2, axis=1, ddof=100)

    #GMM of both MFCC feature vectors (array)
    gm2 = GaussianMixture(n_components=1, covariance_type='diag').fit(mfccs2)
    
    #print(str(validation))
    num_of_cols = int((mfccs2.shape[1]))

    #Calculating Log Likelihood/F Value of Both Models
    log_like1 = 0
    f_test_value = 0

    log_like2 = ((gm2.score(mfccs2))) 
    for i in range(0, ratio):
        gmPart = joblib.load(base+enrollmentFiles[i])
        log_like1 += abs(gmPart.score(mfccs2))
        f_test_value += f_test(gmPart.means_, gm2.means_, log_like1/(i+1), log_like2)[1]
        
    log_like1 = -1*log_like1/ratio

    #Chi-square test of independence used to compare two models
    #Output: P-value
    Λ = 2 * abs(log_like1-log_like2)
    df = 1  # df = degrees of freedom model 1 has over model 2
    likeh_ratio = chi2.sf(Λ, df) # this is a p value
    
    likeh_ratio = f_test_value/ratio

    likeh_ratio_array.append(likeh_ratio)
    
    print('Log_LH#1: '+str(log_like1)+'\t'+'Log_LH#2: '+str(log_like2)+'\tLikelihood Ratio: '+str(likeh_ratio))

for t in range(len(threshold_values)):
    threshold = threshold_values[t]
    for e in range(len(validationFiles)):
        if(likeh_ratio_array[e] <= threshold):
            match = True
            result_array2[e][0] = 'Accept'
        else:
            match = False
            result_array2[e][0] = 'Reject'
        testResults[t][e] = match

print(f'\nSpeaker: {speaker}\tHop Length: {hop_length}\tLanguage: {language}\tMFCC: {n_mfcc}\t')

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
    print(f'{threshold_values[t]:.2e}', TA, TR, FA, FR, str(round(accuracy*100, 4)), str(round(EER*100, 4)), str(round(FAR, 5)), str(round(TAR, 5)), str(round(FRR, 5)))
   
   


