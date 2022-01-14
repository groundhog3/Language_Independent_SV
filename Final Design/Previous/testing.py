from os import listdir, walk
from os.path import isfile, join

mypath = "Audio/English"

validationFiles =[]

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for root, dirs, files in walk(mypath):
    # select file name
    for file in files:
        # check the extension of files
        if file.endswith('.wav'):
            # print whole path of files
            wav = file
            if "validation" in str(wav):
                validationFiles.append(wav)

print(validationFiles)