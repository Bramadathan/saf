import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
# Read in data
df = pd.read_csv('https://raw.githubusercontent.com/bananapeely3123/Dataa/main/Review.csv')
print(df.shape)
# df = df.head(500)
print(df.shape)
df.head()
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()
example = df['Text'][50]
print(example)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
tokens = nltk.word_tokenize(example)
tokens[:10]
tagged = nltk.pos_tag(tokens)
tagged[:10]
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)
# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']

        # Skip rows where the 'Text' column is null or not a string
        if pd.isnull(text) or not isinstance(text, str):
            continue

        # Apply sentiment analysis only if 'Text' is not null and is a string
        res[myid] = sia.polarity_scores(text)
    except RuntimeError:
        print(f'Broke for id {myid}')
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
# Now we have sentiment score and metadata
vaders.head()
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# VADER results on example
print(example)
sia.polarity_scores(example)
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

        # Skip rows where the 'Text' column is null or not a string
        if pd.isnull(text) or not isinstance(text, str):
            continue

        # Perform VADER sentiment analysis
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value

        # Perform RoBERTa sentiment analysis
        roberta_result = polarity_scores_roberta(text)

        # Combine VADER and RoBERTa results
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except Exception as e:
        print(f'Error occurred for id {myid}: {e}')
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
results_df.columns
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")
sent_pipeline('Make sure to like and subscribe!')
sent_pipeline('booo')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
def detect_fake_comment(text):
    # Check if the text is null
    if pd.isnull(text):
        return 'null'

    # Analyze sentiment of the comment
    sentiment_score = sia.polarity_scores(str(text))['compound']

    # If sentiment score is very positive or very negative, consider it as fake comment
    if sentiment_score >= 0.5 or sentiment_score <= -0.5:
        return 'fake'
    else:
        return 'genuine'
df['predicted_class'] = df['Text'].apply(detect_fake_comment)
df['true_class'] = 'genuine'
accuracy = (df['predicted_class'] == df['true_class']).mean()
print("Accuracy of fake comment detection:", accuracy)
df['comment_class'] = df['Text'].apply(detect_fake_comment)
ax = df['comment_class'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Comments by Classification',
          figsize=(10, 5))
ax.set_xlabel('Comment Classification')
plt.show()
accuracy = (df['predicted_class'] == df['true_class']).mean()
print("Accuracy of fake comment detection:", accuracy)
num_fake_comments = df['predicted_class'].value_counts()['fake']
print("Number of fake comments:", num_fake_comments)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
pip install tqdm
from tqdm import tqdm



res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']

        # Skip rows where the 'Text' column is null or not a string
        if pd.isnull(text) or not isinstance(text, str):
            continue

        # Tokenize the text only if it's not null and is a string
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Store sentiment scores
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        res[myid] = scores_dict
    except Exception as e:
        print(f'Error occurred for id {myid}: {e}')
results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
sns.pairplot(data=results_df,
             vars=['roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='comment_class',
             palette='tab10')
plt.show()
comment=" Apple Vision Pro sets a new standard for professional creativity. Its unparalleled display clarity and advanced processing capabilities empower creators to bring their visions to life with stunning precision. With Apple Vision Pro, innovation meets inspiration, elevating the creative process to new heights."
comment2= " Just got my hands on the new Nothing Phone and I'm seriously impressed! Sleek, minimalist design with a powerful performance to match. It's like holding the future in your palm. Bye bye, cluttered distractions – hello, simplicity!"
detect_fake_comment(comment2)
comment3=" this is a very good product for my daily use"
detect_fake_comment(comment3)
detect_fake_comment(comment)
