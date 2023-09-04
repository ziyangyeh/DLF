#! /bin/bash

mkdir -p data/raw
mkdir data/processed

pip install -U kaggle

kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/raw

unzip data/raw/pima-indians-diabetes-database.zip -d data/raw/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes -P data/processed
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale -P data/processed
