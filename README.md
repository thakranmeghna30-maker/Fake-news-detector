# Fake News Detector

A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP).

## Project Overview

This project uses text preprocessing and machine learning to analyze news article text and predict whether it belongs to the Fake or Real category.

The application provides a simple web interface built with Streamlit, where users can enter a news article and receive a prediction.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Joblib

## How It Works

1. Load and combine the fake and real news datasets.
2. Assign labels to the articles (0 = Fake, 1 = Real).
3. Clean the text by removing URLs, special characters and extra spaces.
4. Remove English stopwords.
5. Convert text into numerical features using TF-IDF.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate the model using accuracy and a confusion matrix.
8. Save the trained machine learning pipeline using Joblib.
9. Load the saved pipeline in the Streamlit application to make predictions.

## Project Structure

```text
Fake-news-detector/
│
├── Fake news detector main.py
├── fake_news_app.py
├── fake_news_pipeline.pkl
├── requirements.txt
├── README.md
│
└── News _dataset/
    ├── Fake.csv
    └── True.csv
```

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/thakranmeghna30-maker/Fake-news-detector.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Fake-news-detector
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run fake_news_app.py
   ```

## Model Performance

The model achieved approximately **94.2% accuracy** on the test dataset in the current evaluation.

## Important Note

This project is an educational machine learning application. Its predictions are based on patterns learned from the training dataset. It does not independently verify the factual accuracy of news articles.

## Future Improvements

- Experiment with additional machine learning algorithms.
- Improve text preprocessing.
- Explore larger and more diverse datasets.
- Improve the user interface.
