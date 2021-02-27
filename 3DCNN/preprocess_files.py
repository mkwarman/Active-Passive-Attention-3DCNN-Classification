from os import listdir

DATA_LOCATION = '_data'  # Folder containing data files

# Remove header lines leaving only data
filenames = listdir(DATA_LOCATION)
for filename in filenames:
    with open(DATA_LOCATION + '/' + filename, "r") as f:
        lines = f.readlines()
    with open(DATA_LOCATION + '/' + filename, "w") as f:
        for line in lines:
            if not line.startswith('#'):
                f.write(line)
