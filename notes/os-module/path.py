import os
from  pathlib import Path


# current_dir= os.getcwd()
# print("Get Current Working Directory", os.getcwd())
# print("Absolute Path:",os.path.abspath(__file__))
# print("Absolute Path Directory:",os.path.dirname(os.path.abspath(__file__)))
# print(os.path.join(os.getcwd(),"path"))

print("Get Current Working Directory", Path.cwd())
print("Absolute Path:",Path(__file__))
print("Absolute Path:",Path(__file__).resolve().parent)
print(Path(__file__).resolve().parent/"path")