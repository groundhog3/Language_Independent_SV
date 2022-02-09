import shutil, os

base_path = '/Users/ryanmechery/Downloads/cv-corpus-7.0-2021-07-21/en/clips/'
target_path = "/Users/ryanmechery/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Mass Academy/STEM/Design #1/Audio/Common Voice Dataset/English/sp-8_en"

age="thirties"
gender="female"
client_id="06e304a62b3ecddd2442064494751a93afa3a6b04f7579941a128a08324e3e728c295e872903723d33a7484e858ed1db940cc9c57503af8da0e764a9007d70c4"

file_list = []

fileSearchPath = 'File Search Tool/fileSearch.txt'


with open(fileSearchPath, "r") as tf:
    file_list = tf.read().split('\n')

print(file_list)

for f in file_list:
    file_name = str(base_path + f+'.mp3')
    shutil.copy(file_name, target_path)

with open(target_path+'/client_id.txt', 'w') as f:
    f.write(f"{client_id}\n{gender}\t{age}")
