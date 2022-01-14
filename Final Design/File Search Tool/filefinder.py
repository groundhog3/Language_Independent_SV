import shutil, os

base_path = '/Users/ryanmechery/Downloads/cv-corpus-7.0-2021-07-21/vi/clips/'
target_path = "/Users/ryanmechery/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Mass Academy/STEM/Design #1/Audio/Common Voice Dataset/Vietnamese/sp-3_vi"

age="sixties"
gender="male"
client_id="c14ad590de005be3a512e187900c5ad5c76921cf1883b3349cda8e57e390901f453aae312b3c027b31acc18b7e2f3e857b8f8618ab1eb3a0be7298f8d1c001ce"

file_list = []

fileSearchPath = 'File Search Tool/fileSearch.txt'


with open(fileSearchPath, "r") as tf:
    file_list = tf.read().split('\n')

print(file_list)

for f in file_list:
    file_name = str(base_path + f)
    shutil.copy(file_name, target_path)

with open(target_path+'/client_id.txt', 'w') as f:
    f.write(f"{client_id}\n{gender}\t{age}")
