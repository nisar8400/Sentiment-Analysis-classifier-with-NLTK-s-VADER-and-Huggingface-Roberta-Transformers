
#/////////////////////////////////////  Naive Byese Classifier ////////////////////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

df = pd.read_csv(r'C:\Users\Amjad Maaz\PycharmProjects\month_1\Reviews.csv')
# print(df.shape)
df = df.head(10)
# print(df.head())

ax = df['Score'].value_counts().sort_index( ).plot(kind = 'bar', title = 'Count of review Stars', figsize = (10,5))
ax.set_xlabel('Reviews Stars')
# plt.show()


# //////////////////////////////// basics of NLTK //////////////////////////////////

example = df['Text'][5]
# print(example)
tokens = nltk.wordpunct_tokenize(example)

# print(tokens[:10])

tagged = nltk.pos_tag(tokens)
# print(tagged)



entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()

# /////////////////////////////////////////////////////////////// Vader Sentiment Scoring /////////////////

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from tqdm.auto import tqdm

sia = SentimentIntensityAnalyzer()

# print(sia.polarity_scores('I an not soo much happy'))


res = { }

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# vaders =pd.DataFrame(res).T
# vaders = vaders.reset_index().rename(columns={'index' : 'Id'})
#
# vaders = vaders.merge(df, how = 'left')
# vaders = vaders.iloc[:, :5]
# print(vaders['Score'])
#
# fig, axs = plt.subplots(1,3, figsize = (10,3))
#
#
# ax = sns.barplot(data= vaders, x='Score', y='pos', ax=axs[0])
# ax = sns.barplot(data= vaders, x='Score', y='neu', ax=axs[1])
# ax = sns.barplot(data= vaders, x='Score', y='neg', ax=axs[2])
# ax.set_title('Positive')
# ax.set_title('Neutral')
# ax.set_title('Negtive')
# plt.tight_layout()
# plt.show()
# //////////////////////////////// ROBERTA PRETRAINED MODEL ////////////////////////////////////

# from transformers import AutoTokenizer
# from transformers import AutoModelForTokenClassification
# from scipy.special import softmax
#
#
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForTokenClassification.from_pretrained(MODEL)


# print(example)
# sia.polarity_scores(example)


# //////////////////////////////////////////////////////////////

# ////////////////////////// Run for Roberta Model ////////////////////////

# def polarity_scores_roberta(example):
#     encoded_text = tokenizer(example, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dic = {'roberta_neg' : scores[0],
#                   'roberta_neu' : scores[1],
#                   'roberta_pos' : scores[2]
#                   }
#     return scores_dic
#
#
# res = {}

# for i, row in tqdm(df.iterrows(), total=len(df)):
#     try:
#         text = row['Text']
#         myid = row['Id']
#         vader_result= sia.polarity_scores(text)
#         roberta_result = polarity_scores_roberta(text)
#         vader_result_name = {}
#         for key, value in vader_result.items():
#             vader_result_name[f'vader_{key}'] = value
#             roberta_result = polarity_scores_roberta(text)
#             both = {**vader_result_name, **roberta_result}
#             res[myid] = both
#     except RuntimeError:
#         print(f'Broke for id {myid}')
#
# print(res)
#
# results_df =pd.DataFrame(res).T
# results_df = results_df.reset_index().rename(columns={'index': 'Id'})
#
# results_df =results_df.merge(df, how='left')

# results_df = results_df.iloc[:, :5]
# print(results_df)


# //////////////////////////// CompareScores Between Models /////////////////////////

#
# print(results_df.columns)
# # sns.pairplot(results_df, vars={"vader_neg", "vader_neu", "vader_pos", "roberta_neg", "roberta_neu", "roberta_pos"},
# #              hue='Score', palette='tab10')
#
# plt.show()
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ From kagle \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

print(results_df.columns)

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()