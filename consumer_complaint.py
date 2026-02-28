#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_palette("husl")
from wordcloud import WordCloud


# In[2]:


df = pd.read_csv("input_data/complaints.csv.zip", low_memory=False)


# In[3]:


print(df.shape)
df.head(3)


# In[4]:


# For this I only care about consumer complaint narrative and product
cols = ['Consumer complaint narrative', 'Product']
df = df[cols].dropna()
print(df.shape)
(df.value_counts())


# In[5]:


plt.figure(figsize=(10,5))
df['Product'].value_counts().plot(kind='bar', color='steelblue')
plt.title("Complaints by Product (Department)")
plt.xticks(rotation=45, ha ='right')
plt.tight_layout()
plt.show()


# In[6]:


# I will just keep top 6 
top_products = df['Product'].value_counts().nlargest(6).index

df = df[df['Product'].isin(top_products)]

print(df['Product'].value_counts())


# In[7]:


merge_map = {
    'Credit reporting, credit repair services, or other personal consumer reports': 'Credit reporting',
    'Credit reporting or other personal consumer reports': 'Credit reporting',
    'Debt collection': 'Debt collection',
    'Mortgage': 'Mortgage',
    'Checking or savings account': 'Checking or savings account',
    'Money transfer, virtual currency, or money service': 'Money transfer'
}

df['Product'] = df['Product'].map(merge_map)

df = df.dropna(subset=['Product'])
print(df['Product'].value_counts())


# In[8]:


# I will downsample each class to 10,000 samples. 
df_dsp = (
    df.groupby('Product', group_keys=False)
    .apply(lambda x: x.sample(10000, random_state=56))
    .reset_index(drop=True)
)

print(df_dsp['Product'].value_counts())
print("Total:", len(df_dsp))


# In[9]:


df_dsp.rename(columns={'Consumer complaint narrative':'complaint'}, inplace=True)


# In[10]:


# Clean the text
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'x{2,}', ' ', text) #remove XXXX redacted info
    text = re.sub(r'[^a-z\s]', ' ', text) #remove special chars/numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_dsp['complaint_clean'] = df_dsp['complaint'].apply(clean_text)

df_dsp[['complaint', 'complaint_clean']].head(3)


# In[11]:


# Check the average complaint length

df_dsp['word_count'] = df_dsp['complaint_clean'].apply(lambda x: len(x.split()))

print(df_dsp['word_count'].describe())


# ## Classification
# I am using TF-IDF and Logistic Regression as my baseline. It's fast, interpretable, and works well on text.

# In[12]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_dsp['label'] = le.fit_transform(df_dsp['Product'])

print(dict(zip(le.classes_, le.transform(le.classes_))))


# In[13]:


df_dsp.sample(5)


# In[14]:


print(len(df_dsp['complaint_clean']), len(df_dsp['label']))


# In[15]:


# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_dsp['complaint_clean'], df_dsp['label'],
    test_size = 0.2, random_state = 56, stratify = df_dsp['label']
)

print(f"Train:{len(X_train)} | Test:{len(X_test)}")


# In[16]:


# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), max_df=0.95, min_df=5)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.fit_transform(X_test)

print(X_train_tfidf.shape)


# In[17]:


# Train Logistic regression
from sklearn.linear_model import LogisticRegression

mod = LogisticRegression(max_iter=1000, random_state=42)
mod.fit(X_train_tfidf, y_train)


# In[18]:


# Evaluate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = mod.predict(X_test_tfidf)

print(classification_report(y_pred, y_test, target_names=le.classes_))


# In[19]:


disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, 
                                                xticks_rotation=45, colorbar=False)
disp.ax_.grid(False)
disp.ax_.set_xticklabels(le.classes_, rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.show()


# Despite balanced training data, logistic regression with TF-IDF collapsed toward a semantically central class due to vocabulary overlap. Check column 3, the model is heavily predicting debt collection, which means when uncertain, it predicts class3. Also notice that how macro avg F1 < than accuracy? This suggest that model performs worse on at least one class and that class is likely a minority class. Despite balancing the class, I encountered class imbalance issue. This demonstrates limitations of bag-of-words models for nuanced complaint categorization. I need to switch to a transformer-based model.

# In[20]:


from transformers import pipeline
embedder = pipeline('feature-extraction', model='distilbert-base-uncased', 
                    truncation=True, max_length=128)

def get_embedding(text):
    output = embedder(text[:512])  # truncate input
    return np.mean(output[0], axis=0)  # mean pool

print("Test embedding shape:", get_embedding("test complaint").shape)


# In[21]:


# Generate embedding
from tqdm import tqdm
tqdm.pandas()

print("Embedding train set...")
X_train_emb = np.array([get_embedding(t) for t in tqdm(X_train)])

print("Embedding test set...")
X_test_emb = np.array([get_embedding(t) for t in tqdm(X_test)])

print("Done! Shape:", X_train_emb.shape)


# In[22]:


lr_model = LogisticRegression(max_iter=1000, random_state=56) # samples are manually balanced to 10K, 'class_weight' removed.
lr_model.fit(X_train_emb, y_train)
print("Done!")


# In[23]:


y_pred = lr_model.predict(X_test_emb)
print(classification_report(y_test, y_pred, target_names=le.classes_))


# In[24]:


# Download nltk data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import sumy


# In[25]:


# Summarizer function
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize (text, num_sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary)


# In[26]:


# use original complaint, not complaint_clean. Because cleaning affected the sentence organization. Can go back and modify cleaning function.
sample_idx = X_test.index[2]
original = df_dsp.loc[sample_idx, 'complaint']
print("ORIGINAL:\n", original)
print("\nSUMMARY:\n", summarize(original))


# In[27]:


def process_complaint(complaint_text):
    # 1. Summarize
    summary = summarize(complaint_text)
    
    # 2. Clean for embedding
    cleaned = clean_text(complaint_text)
    
    # 3. Embed & predict
    embedding = get_embedding(cleaned).reshape(1, -1)
    label_idx = lr_model.predict(embedding)[0]
    department = le.inverse_transform([label_idx])[0]
    confidence = lr_model.predict_proba(embedding).max() * 100
    
    return {
        "department": department,
        "confidence": f"{confidence:.1f}%",
        "summary": summary
    }


# In[28]:


test_complaint = df_dsp['complaint'].iloc[5]
result = process_complaint(test_complaint)

print(f"📋 COMPLAINT:\n{test_complaint}\n")
print(f"🏢 ROUTE TO: {result['department']}")
print(f"🎯 CONFIDENCE: {result['confidence']}")
print(f"📝 SUMMARY: {result['summary']}")


# The pipeline works. But the routing seems slightly off — "Withdrawal from account" should go to Checking or savings account, not Debt collection. This may be expected since it's a very short complaint with little context for the model to work with.

# In[29]:


# Let's test with a more detailed complaint

complaints = [
    "I have been trying to get a fraudulent account removed from my credit report for months. The credit bureau keeps verifying the account despite me sending them proof that I never opened it.",
    "I took out a 30 year fixed mortgage in 2018 and the bank has been charging me incorrect escrow amounts every month. They refuse to provide a proper escrow analysis.",
    "A debt collector keeps calling me 5 times a day even after I sent them a cease and desist letter. They are violating the FDCPA by continuing to harass me.",
]

for c in complaints:
    result = process_complaint(c)
    print(f"📋 COMPLAINT: {c[:80]}...")
    print(f"🏢 ROUTE TO: {result['department']}")
    print(f"🎯 CONFIDENCE: {result['confidence']}")
    print(f"📝 SUMMARY: {result['summary']}")
    print("-" * 60)


# In[30]:


# Save model aftifacts
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Saved!")


# In[31]:


# Lets try SBERT
get_ipython().system('{sys.executable} -m pip install sentence-transformers')


# In[32]:


# Lets try SBERT
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('all-mpnet-base-v2')

print("Embedding train set...")
X_train_sbert = sbert.encode(X_train.tolist(), batch_size=64, show_progress_bar=True)

print("Embedding test set...")
X_test_sbert = sbert.encode(X_test.tolist(), batch_size=64, show_progress_bar=True)

print("Shape:", X_train_sbert.shape)


# In[33]:


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = LinearSVC(max_iter=2000, class_weight='balanced')
svm_calibrated = CalibratedClassifierCV(svm)  # needed for predict_proba
svm_calibrated.fit(X_train_sbert, y_train)
print("Done!")


# In[34]:


y_pred_sbert = svm_calibrated.predict(X_test_sbert)
print(classification_report(y_test, y_pred_sbert, target_names=le.classes_))


# In[35]:


import pickle

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_calibrated, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Saved!")


# In[36]:


# Confusion matrix for final model. 
plt.figure(figsize=(8,6))
cm_final = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_sbert, display_labels=le.classes_, 
                                                xticks_rotation=45, colorbar=False)
cm_final.ax_.grid(False)
cm_final.ax_.set_xticklabels(le.classes_, rotation=30, ha='right', fontsize=8)
plt.tight_layout()

plt.savefig("figures/sbert_cm.png", dpi=300, bbox_inches="tight") 

plt.show()


# In[37]:


# Create a combined bar chart
models = ['TF-IDF + LR', 'DistilBERT + LR', 'SBERT + SVM']
accuracy = [28, 84, 89]
macro_f1 = [19, 84, 89]

x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(8, 6))

b1 = ax.bar(x - 0.2, accuracy, 0.4, label='Accuracy', color='#3498db')
b2 = ax.bar(x + 0.2, macro_f1, 0.4, label='Macro F1', color='#2ecc71')

for bar in b1 + b2:
    ax.text(bar.get_x() + 0.2, bar.get_height() + 1,
            f'{bar.get_height()}%', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 105)
ax.set_ylabel('Score (%)')
ax.set_title('Model Progression', fontweight='bold')
ax.legend(loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

plt.savefig("figures/model_comparison.png", dpi=300, bbox_inches="tight")

plt.show()

