import os
import glob
from zipfile import ZipFile


INPUT_DIR = os.path.join("data")
DATA_DIR = os.path.join("staging","data09-21")

def unzip_unpack(INPUT_DIR):
    for zfile in glob.glob("*.zip"):
        with ZipFile(zfile) as zitem:
            for file in zitem:
                file.extract('', path=DATA_DIR)

if __name__ == "__main__":

    os.makedirs(DATA_DIR, exist_ok=True)