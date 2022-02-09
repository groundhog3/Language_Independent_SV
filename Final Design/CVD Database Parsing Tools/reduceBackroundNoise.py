import noisereduce as nr 
from scipy.io import wavfile

#load data 
directory = "/Users/ryanmechery/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Mass Academy/STEM/Design #1/Audio/Common Voice Dataset/English/"
file = "sp-2_en.wav"
rate, data = wavfile.read("mywav.wav") 
# select section of data that is noise 
noisy_part = data[10000:15000] 
# perform noise reduction 
reduced_noise = nr.reduce_noise(audio_clip=data,noise_clip=noisy_part, verbose=True)