# Histopathology-Cancer-Detection
https://github.com/HagenMarin/Histopathology-Cancer-Detection.git \
used libraries:

Pytorch\
Torchvision\
pathlib\
glob\
cv2\
pandas\
pickle\
scikit learn\
numpy\
matplotlib

Setup:
1. install all of the libraries listed above in your environment
2. clone the repository
3. download the data from https://www.kaggle.com/competitions/histopathologic-cancer-detection/data and extract everything into the folder of the repository
4. create 3 folders called "test_batches128", "train_batches128" and "valid_batches128"
5. run run_to_create_batches.py (those Batches will also be needed for the Collab)
6. if you get an error make sure all of the used libraries are installed in your env
7. create an empty folder called model

Training:\
run train.py (you can use "python train.py -h" or "python train.py -help" to see all the commandline options)

Collab:\
if you want to run the collab yourself you need to upload the folder "test_batches128" and the file "train_labels.csv" into your google drive (after running run_to_create_batches.py !!!)
