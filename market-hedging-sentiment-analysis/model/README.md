# Market Hedging Based On Sentiment Analysis Project

This project aims to perform sentiment analysis on various datasets collected from different sources such as Twitter, Reddit, and financial news headlines. The goal is to analyze the sentiment expressed in text data and gain insights into public opinions and emotions on different topics.

## Datasets

### Dataset 1: Apple Twitter Sentiment

- **File:** apple_twitter_sentiment_texts.csv
- **Columns:** text, sentiment
- **Description:** Cleaned tweets related to Apple products with sentiment labels.

### Dataset 2: Covid-19 Indian Sentiments

- **File:** Covid_19_Indian_Sentiments.csv
- **Columns:** text, sentiment
- **Description:** Tweets from India discussing COVID-19 topics labeled with sentiments: fear, sadness, anger, and joy.

### Dataset 3: Sentiment Analysis for Financial News

- **File:** FinancialPhraseBank.csv
- **Columns:** sentiment, news_headline
- **Description:** Sentiments for financial news headlines from a retail investor's perspective.

### Dataset 4: Twitter and Reddit Sentimental Analysis

- **Files:** Twitter.csv, Reddit.csv
- **Columns:** clean_comment, category
- **Description:** Tweets from Twitter and comments from Reddit with sentiments on political leaders and the next Prime Minister of India.

### Dataset 5: Twitter US Airline Sentiment

- **File:** Twitter_US_Airline_Sentiment.csv
- **Columns:** tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, user_timezone
- **Description:** Sentiment analysis of tweets about major U.S. airlines, categorizing tweets as positive, negative, or neutral, and providing additional metadata.

## Analysis

The project involves preprocessing the text data, applying sentiment analysis techniques such as natural language processing (NLP) and machine learning algorithms, and visualizing the results. In particular, a bidirectional LSTM (Long Short-Term Memory) model has been used for sentiment analysis with an achieved accuracy of 94.31%.

## Usage

To use the datasets for sentiment analysis, load the desired dataset into your preferred data analysis tool or programming environment (e.g., Python with pandas). Preprocess the text data as needed, apply sentiment analysis techniques, and analyze the results.

Feel free to contribute to this project by adding more datasets, improving analysis techniques, or providing feedback on existing code and documentation.


