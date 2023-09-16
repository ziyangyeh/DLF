#! /bin/bash

MY_PATH="$(dirname -- "${BASH_SOURCE[0]}")"

mkdir -p $MY_PATH/data/raw
mkdir $MY_PATH/data/processed

pip install -U kaggle

kaggle datasets download -d uciml/pima-indians-diabetes-database -p $MY_PATH/data/raw

unzip $MY_PATH/data/raw/pima-indians-diabetes-database.zip -d $MY_PATH/data/raw/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes -P $MY_PATH/data/processed
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale -P $MY_PATH/data/processed
