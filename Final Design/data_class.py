from dataclasses import dataclass #used to create a dataclass to set parent class
import re #used for regex matching
from numpy import string_ 
from os import listdir, walk
from os.path import isfile, join

@dataclass
class Speaker_Verification:
    def __init__(self, type):
        language = "Vietnamese/"
        base = "./Audio/Common Voice Dataset/"+language
        speaker = "3_vi"

        #language = "Malayalam/"
        #base = "./Audio/December Fair Test #1/"+language
        #speaker = "geo"
        enrollment_base = "enrollment sp-"+speaker
        enrollment_wav = enrollment_base+".wav"
        enrollment_pickle = base+enrollment_base+".pkl"
        enrollment = str(base+enrollment_wav)
        validationFiles = []

        hop_length = 128 #hop length in ms
        n_mfcc = 21 #number of MFCCs extracted per frame
        validation_length = 5 #seconds
        enrollment_length = 40 #seconds
        self.ratio = enrollment_length/validation_length

        for root, dirs, files in walk(base):
            # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.wav'):
                    # print whole path of files
                    wav = file
                    if "validation" in str(wav):
                        validationFiles.append(wav)
        
        if (type == "enrollment"):
            self.base = base
            self.enrollment_base = enrollment_base
            self.enrollment = enrollment
            self.hop_length = hop_length
            self.n_mfcc = n_mfcc
            self.validation_length = validation_length 
            self.hop_length = hop_length
            
        elif (type == "main"):
            self.base = base
            self.speaker = speaker
            self.enrollment_base = enrollment_base
            self.enrollment_wav = enrollment_wav
            self.enrollment_pickle = enrollment_pickle
            self.enrollment = enrollment
            self.validationFiles = validationFiles
            self.hop_length = hop_length
            self.n_mfcc = n_mfcc 
            self.validation_length = validation_length 

        self.language = language

    def getSpeaker(file) -> string_:
        sp = re.findall("sp-(.+)#", file)
        if(len(sp) <= 0 or len(sp) > 1):
            return "ERROR"
        return sp[0].replace(" ", "")
    
   
    