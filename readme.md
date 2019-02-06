In order to reproduce our results, one can find two kinds of executable scripts visualize_*.py and train_*.py

For both kinds of scripts the data is needed to get it download the from here:
https://motchallenge.net/data/MOT17.zip
extract it and copy all folders with the from MOT17-**-SDP into the './dataset_utils/datasets/MOT17' Folder

For running the train_yolo.py, visualize_yolo.py, visualize_yolo_lstm.py pretrained models are needed which can be found
https://drive.google.com/drive/folders/1fv6HbPDP1nclEOLRm6mDg_siSypyJNJ8?usp=sharing
and should be copied into the './models' folder in order to run the visualize scripts out of the box

train_*.py scripts are used to train the networks. If the pretrained models are in the model folder it will work out of
the box and log the training results in the log folder. If you want to continue from your own snapshots you need to
change the path for the pretrained model in the script

visualize_*.py scripts will produce a output video based on the pretrained models and one sequence part which wasn't
used for training. If you want to change the sequence or use your own snapshot just change the paths in the scripts

Additionally we provided for completeness the files we used to train our Networks on Google Colab. These are experimental
and used for rapid prototyping. The final trainings methods are included in the train_*.py scripts