from pydub import AudioSegment

enroll = 40 * 1000 #Enrollment Length (milliseconds)
val = 5 * 1000 #Validation Length (milliseconds)

directory = "/Users/ryanmechery/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Mass Academy/STEM/Design #1/Audio/Common Voice Dataset/English/"
file = "sp-8_en.wav"


clip = AudioSegment.from_wav(directory + file)
enroll_audio = clip[0:enroll]
enroll_audio.export(directory+"enrollment " + file, format="wav") #Exports enrollment file in base dir

count = 1
for i in range(enroll, len(clip)-val, val):
    #print("Begin:", i/1000, "\tEnd:", (i+5000)/1000)
    validation_name = f"validation {file.split('.')[0]} #{count}.wav"
    print(validation_name)
    validation =  clip[i:i+val]
    validation.export(directory+validation_name, format="wav") #Exports enrollment file in base dir
    count += 1
