import os

# folder path
dir_path = "/srv/disk00/kaiwenyu_data_process/result/EPIC_08_10"
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)