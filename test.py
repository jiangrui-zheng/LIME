import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


df = pd.read_csv('/data/jzheng36/hatemoderate/hatemoderate/all_example.csv', sep = '\t')

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=42)

# Vectorize the text using Tfidf
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

#  classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_vectorized, y_train)

pipeline = make_pipeline(vectorizer, rf)

# Create Lime Text Explainer
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Choose an instance to explain
idx = 1 # Index of the instance in your test set
text_instance = X_test.iloc[idx]

# Explain the prediction
exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6)
exp.show_in_notebook(text=True)