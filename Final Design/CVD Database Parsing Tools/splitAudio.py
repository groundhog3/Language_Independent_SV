from pydub import AudioSegment
import os

directory = "/Users/ryanmechery/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Mass Academy/STEM/Design #1/Audio/Common Voice Dataset/English/sp-8_en/"

audio_files = []

name = "sp-8_en"

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.mp3'):
            print(file)
            audio_files.append(AudioSegment.from_file(directory+file, format="mp3"))
            

combined = AudioSegment.empty()

for mp3 in audio_files:
    combined += mp3

file_handle = combined.export(directory+name+'.mp3', format="mp3")
