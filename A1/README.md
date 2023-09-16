# Deep Learning Fundamental - Assignment 1

| Author | ID |
|--------|-----------|
| Ziyang Ye| a1707805  |

## 1. Install Requirements
(a) Run `pip install -r requirements.txt`.

## 2. Download Dataset

(a) Run `bash download_dataset.sh`.
(b) Make sure `data` directory looks like:
```
data
├── processed
│   ├── diabetes
│   └── diabetes_scale
└── raw
    ├── diabetes.csv
    └── pima-indians-diabetes-database.zip
```

## 3. Process Dataset
(a) Run `data_vis_and_scale.ipynb`.
(b) Make sure `3` new csv files are generated in `data` directory: `raw.csv; scaled.csv; z_score.csv`.

