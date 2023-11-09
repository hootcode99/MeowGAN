import os
import tarfile

directory = os.path.dirname(__file__)
data_directory = os.listdir(os.path.dirname(__file__))
out_directory = directory + "\\dataset"

for file_name in data_directory:
    if file_name.endswith("tar.gz"):
        tar = tarfile.open(file_name, "r:gz")

        for member in tar:
            if member.name.endswith('png'):
                member.name = os.path.basename(member.name)  # remove the path
                tar.extract(member, out_directory)

        tar.close()

