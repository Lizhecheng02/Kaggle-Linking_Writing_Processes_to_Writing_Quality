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

<img src="C:\Users\86183\Desktop\Weixin Image_20231217143813.png" alt="Final Look" style="zoom: 150%;" />



### Run Codes

#### 1.  Run Simple Models

```bas
cd models
python run {file_name}.py
```

Models will be saved in the main directory.
