## This Repo is for [Kaggle - Linking Writing Processes to Writing Quality Competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality)



### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```



### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_api_key
```

#### 2. Download Dataset

```bash
kaggle competitions download -c linking-writing-processes-to-writing-quality
unzip linking-writing-processes-to-writing-quality.zip
```

```bash
kaggle datasets download -d hiarsl/writing-quality-challenge-constructed-essays
unzip writing-quality-challenge-constructed-essays.zip
```



### Run Code

#### 1.  Run Simple Models

```bash
cd models
python {file_name}.py
```

- Models will be saved in the main directory.

#### 2. Save Features

```bas
python feature_generator.py
```

- Final train csv will be saved in ``features.csv `` under the main directory.
- Feature names will be saved as a list in ``columns.txt`` under the main directory.
- See exploratory data analysis in ``features_eda.ipynb``
