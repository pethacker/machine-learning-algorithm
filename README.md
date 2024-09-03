# machine-learning-algorithm
## master
The master branch is about processing the CIC-IDS2017 dataset, which contains merging the dataset, expanding the dataset, extracting features, training and testing, the dataset can be downloaded via https://www.unb.ca/cic/datasets/ids-2017.html.

The code of merging dataset is used to deal with csv file, while the code of extracting features is used to deal with json file, you can refer to the use of joy tool to get the json file, the relevant address: https://github.com/cisco/joy (Note that the bottom of the code of joy uses the logic of python2, if you are using a python3 environment, please modify the underlying code to facilitate the use). The confusion matrix drawing code is used to facilitate the observation of accuracy in training and testing, if you only want to focus on the final accuracy, you can not use the code. The code is run in the order merge_dataset, balanced_dataset, transfer, data_processing.
## main
The vpn_dataset and vpn_dataset_process in main are for processing and training the cnn model, where the relevant dataset can be downloaded by going to https://www.stratosphereips.org/datasets-malware, selecting the appropriate dataset and then modifying the code in the relevant paths in the code.

The packets.pickle file was created in my own computer environment for direct use in training, but given the low configuration of my computer, I suggest you reconfigure a pickle file for training.