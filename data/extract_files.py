import os
import tarfile

current_directory = os.getcwd()
files = os.listdir(current_directory)
out_directory = os.path.join(current_directory, 'dataset')

for file_name in files:
    if file_name.endswith("tar.gz"):
        tar = tarfile.open(file_name, "r:gz")

        for member in tar:
            if member.name.endswith('png'):
                member.name = os.path.basename(member.name)  # remove the path
                tar.extract(member, out_directory)

        tar.close()

