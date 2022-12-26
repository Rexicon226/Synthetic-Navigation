
# import required module
import os
# assign directory
directory = '/home/dr/Synthetic-Navigation/ML/train_images/clean'
noise_directory = '/home/dr/Synthetic-Navigation/ML/train_images/noise'
 
# iterate over files in
# that 
iter = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        iter += 1
print("Files in Clean Directory: {}".format(iter))
total = iter
iter = 0
for filename in os.listdir(noise_directory):
    f = os.path.join(noise_directory, filename)
    if os.path.isfile(f):
        iter += 1
print("Files in Noise Directory: {}".format(iter))
total = total + iter
print("Total: {}".format(total))
