import pandas as pd
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "News _dataset"
fake_df=pd.read_csv(DATA_DIR/"Fake.csv")
true_df=pd.read_csv(DATA_DIR/"True.csv")
fake_df["label"]=0
true_df["label"]=1
data=pd.concat([fake_df,true_df],axis=0)
data=data.sample(frac=1,random_state=42).reset_index(drop=True)

import re
def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+","",text)
    text=re.sub(r"[^a-z\s]","",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

data["Clean_Text"]=data["text"].apply(clean_text)

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords",quiet=True)
stop_words=set(stopwords.words("english"))

def remove_stopwords(text):
    words=text.split()
    final_text=[word for word in words if word not in stop_words]
    return " ".join(final_text)
data["Final_Text"]=data["Clean_Text"].apply(remove_stopwords)

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

x_train, x_test, y_train, y_test = train_test_split(
    data["Final_Text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", MultinomialNB())
])

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

cm=confusion_matrix(y_test,y_pred)
print("Confusuion matrix:",cm)

joblib.dump(model, BASE_DIR/"fake_news_pipeline.pkl")

user_input=input("Enter a News Article:")
clean_input=clean_text(user_input)
final_input=remove_stopwords(clean_input)
prediction=model.predict([final_input])

if prediction[0]==0:
    print("News Article is FAKE")
else:
    print("News Article is REAL")
