## This Repo is for [Kaggle - Linking Writing Processes to Writing Quality Competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality)



### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
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

```bash
python feature_generator.py
```

- Final train csv will be saved in ``features.csv `` under the main directory.
- Feature names will be saved as a list in ``columns.txt`` under the main directory.
- See exploratory data analysis in ``features_eda.ipynb``

#### 3. Run Classification Models (Poor Results)

```bash
cd classifiers
python {file_name}.py
```




# [617th Solution Write-Up] Summary and Reflection


### 1. Conclusion
I am very pleased to have participated in this meaningful competition. Although I did not win a medal after the shakeup, I learned a lot in the feature type table competition. Hope to apply what I have learned next time and achieve better results. 
My thoughts aren't of much reference value, just simply serve to put a definitive end to this competition and share some findings.

### 2. Feature Selection

Thanks to these excellent public notebooks: [Feature Engineering: Sentence & paragraph features](https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features), [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features), [LGBM (X2) + NN](https://www.kaggle.com/code/cody11null/lgbm-x2-nn).

- Important features:
	1. ``sentence_features``, ``word_features``, ``paragraph_features``
	```
	**use word count with different lengths**
	for word_l in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = df[df['word_len'] == word_l].groupby(['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
	```
	2. ``pause_time_features``
	```
	**only add one new feature**
	pauses_zero_sec=pl.col('time_diff').filter(  
        pl.col('time_diff') < 0.5
	).count(),
	```
	3. ``time-related features``, ``word count features``, ``cursor position features``
	```
	temp = df.group_by("id").agg(
        pl.sum('action_time').suffix('_sum'),
        pl.mean(num_cols).suffix('_mean'),
        pl.std(num_cols).suffix('_std'),
        pl.median(num_cols).suffix('_median'),
        pl.min(num_cols).suffix('_min'),
        pl.max(num_cols).suffix('_max'),
        pl.quantile(num_cols, 0.25).suffix('_quantile25'),
        pl.quantile(num_cols, 0.75).suffix('_quantile75')
	)
	```
	
	4. ``gaps = [1]``
	5. ``punctuations count``
	```
	def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret
	```

- Features with not very obvious effects
  1. ``activity count``, ``event_count`` (Tried using Tf-idf and regular proportion calculations, there was basically no difference.)
  1. ``gaps = [10, 20, ..., 50, 100]``

In the end, used about 210 features. I tried more features (for example, constructing 300+, 600+, and 700+ features), but the scores on the leaderboard were poor, only around 0.595+. Therefore, I did not adopt them in the final model. In fact, when there are only about 2,500 training data entries, there shouldn't be too many features.

### 3. Models
- Used LGBM, XGB, and CB, three traditional tree models, with equal allocation in the model proportions.
- Unable to obtain desired results with NN and TabNet.
### 4. Ideas that Could Not be Realized

After reading this creative discussion [here](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/451852). I tried using a classification model to assist in making certain adjustments to the regression model. 

- Binary Classification

  ```
  count_label0 = 0
  count_label1 = 0
  def create_binary_score(score):
      global count_label0, count_label1
      if score <= 1.5 or score >= 4.5:
          count_label0 += 1
          return 0
      else:
          count_label1 += 1
          return 1
  ```

  I hoped to use a binary classification method to differentiate between marginal scores and middle scores, but the final accuracy was only around 82%. After combining it with the regression model, the results were not satisfactory, so I ultimately abandoned this approach.

- Five-category classification

  ```
  def convert_score_to_category(score):
      if score <= 2.5:
          return 0
      elif score == 3.0:
          return 1
      elif score == 3.5:
          return 2
      elif score == 4.0:
          return 3
      elif score == 4.5:
          return 4
      elif score >= 5.0:
          return 5
  ```

  Here, in order to balance the data volume of each label, I set the division method as mentioned above. Previously, I tried treating each score as a separate category and added weights to minimize the impact of sample imbalance. However, due to the large discrepancy, the model was ultimately unable to train properly. Even when I divided it into the five categories mentioned above, the final classification accuracy was only just over 50%.

After reviewing others' solutions, it seems that no one used this idea, indicating that this method indeed does not work well.

### 5. Summarization

- In this competition, it seems that features are not particularly important. Many high-scoring solutions are also based on making minor modifications to the baseline.
- How to extract more information from text and even use language models to construct features is a very effective approach.
- Building a trustworthy CV is crucial. In the competition, my CV has consistently lacked a strong correlation with the LB, which directly led to shakeup.

### 6. End

Thank you to ``Kaggle`` and ``THE LEARNING AGENCY LAB`` for hosting a very meaningful competition. The tabular competition has been a process of accumulating experience, and I have learned a lot during this process. Wish everyone good luck.