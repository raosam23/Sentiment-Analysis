from flask import Flask, Response, request, render_template
import snscrape.modules.twitter as sntwitter
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traceback

app = Flask(__name__)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# def compute_graph():
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     fig = plt.figure(figsize=(7, 7))
#     colors = ("red", "green", "white")
#     wp = {'linewidth': 2, 'edgecolor': "black"}
#     tags = test_data['sentiment'].value_counts()
#     explode = (0.1, 0.1, 0.1)
#     tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
#               startangle=90, wedgeprops=wp, explode=None, label='')
#     plt.title('Distribution of sentiments')

# def preprocess_text(text):
#     # Tokenize text
#     tokens = word_tokenize(text.lower())

#     # Remove stopwords and stem remaining words
#     stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

#     # Join stemmed tokens back into a single string
#     preprocessed_text = ' '.join(stemmed_tokens)

#     return preprocessed_text 

def preprocess_text(Tweet):
    # Remove mentions and links
    Tweet = re.sub(r'@\S+|https?://\S+', '', Tweet)
    # Remove non-alphabetic characters and convert to lowercase
    Tweet = re.sub('[^a-zA-Z]', ' ', Tweet.lower())
    # Tokenize Tweet
    tokens = nltk.word_tokenize(Tweet)
    # Remove stopwords and stem remaining words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    # Join stemmed tokens back into a single string
    preprocessed_Tweet = ' '.join(stemmed_tokens)
    return preprocessed_Tweet 

# def preprocess_Text(Tweet):
#     # Remove mentions and links
#     Tweet = re.sub(r'@\S+|https?://\S+', '', Tweet)
#     # Remove non-alphabetic characters and convert to lowercase
#     Tweet = re.sub('[^a-zA-Z]', ' ', Tweet.lower())
#     # Tokenize Tweet
#     tokens = nltk.word_tokenize(Tweet)
#     # Remove stopwords and stem remaining words
#     stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
#     # Join stemmed tokens back into a single string
#     preprocessed_Tweet = ' '.join(stemmed_tokens)
#     return preprocessed_Tweet   


@app.route('/')
def twitter_data_form():
    return render_template('form.html')



@app.route('/scrape_twitter_data', methods=['POST'])
def scrape_twitter_data():
    try:
        words = request.form['words']
        global csvFileName
        csvFileName = words.split()[0]+'.csv'
        hashtags_string = request.form['hashtags']
        lang = 'en'
        since = request.form['since']
        till = request.form['till']
        limits = int(request.form['limits'])
        tweets = []
        if len(hashtags_string)==0:
            query = f'{words} lang:{lang} until:{till} since:{since} -filter:links -filter:replies'
        else:
            hashtag_list = [f'#{word}' for word in hashtags_string.split()]
            # print(hashtag_list)
            hashtags = ' OR '.join(hashtag_list)
            # print(actual_hashtag)
            # query = f'({words} OR {hashtags}) since:{since} until:{till}'
            query = f'{words} (#{hashtags}) lang:{lang} until:{till} since:{since} -filter:links -filter:replies'

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():

            if len(tweets) == limits:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.content])

        df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
        csv = df.to_csv(index=False)

        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename="+csvFileName})
    except:
        tb_str = traceback.format_exc()
        return render_template('failure.html', reason = 'to fetch tweets', traceback = tb_str)


@app.route('/perform_sentiment', methods = ['POST'])
def sentiment():
    try:
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        csvPath = os.path.join(downloads_path,csvFileName)
        df = pd.read_csv(csvPath)
        df['Tweet_punct'] = df['Tweet'].apply(lambda x: remove_punct(x))

        df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))

        s=(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week','treatment', 'associated', 'patients', 'may','day', 'case','old'])

        nltk.download('stopwords')
        global stopword
        stopword = nltk.corpus.stopwords.words('english')
        #stopword.extend(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
        #               'treatment', 'associated', 'patients', 'may','day', 'case','old'])

        df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
        global ps
        ps = nltk.PorterStemmer()
        df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))

        global wn
        wn = nltk.WordNetLemmatizer()
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))

        countVectorizer = CountVectorizer(analyzer=clean_text) 
        countVector = countVectorizer.fit_transform(df['Tweet'])
        # print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
        # print(countVectorizer.get_feature_names())

        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        tweets = countVectorizer.get_feature_names_out()
        accuracy=0.0
        for tweet in tweets:
            scores = sia.polarity_scores(tweet)
            # print(tweet, scores['compound'])
            accuracy+=scores['compound']

        average=accuracy/len(tweets)

        result =''

        if average>0:
            result = 'Positive Sentiments'
        elif average<0:
            result = 'Negetive Sentiments'
        else:
            result = 'Neutral Sentiments'
            
        return render_template('result.html', result=result, polar = round(average,4))
    
    except:
        tb_str = traceback.format_exc()
        return render_template('failure.html', reason = 'to compute sentiments', traceback = tb_str)
    
@app.route('/perform_naive_bayes', methods = ['POST'])
def naive_bayes():
    try:
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        csvPath = os.path.join(downloads_path,csvFileName)
        # return render_template('naive_bayes_result.html', result='Hello World')
        # Load data
        # train_path = ''
        train_data = pd.read_csv('.\\CSV Files\\train.csv')
        test_data = pd.read_csv(csvPath)

        # Drop NaN values
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Preprocess data
        global stop_words
        stop_words = set(stopwords.words('english'))
        global stemmer
        stemmer = PorterStemmer()

        # Apply the preprocess_text function to both test and train data
        train_data['text'] = train_data['text'].apply(preprocess_text)
        test_data['Tweet'] = test_data['Tweet'].apply(preprocess_text)

        # Extract features
        vectorized = CountVectorizer()
        X_train = vectorized.fit_transform(train_data['text'])
        X_test = vectorized.transform(test_data['Tweet'])
        y_train = train_data['sentiment']

        # Train model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluate model on test data
        y_predicted = model.predict(X_test)

        # Save predictions to a file
        test_data['sentiment'] = y_predicted
        nblabelled_csv = 'nblabelled_'+csvFileName
        test_data.to_csv('.\\CSV Files\\'+nblabelled_csv, index=False)
        # print('CSV file was generated')
            
        # compute sentiment percentages    
        positive_percentage = round(((sum(y_predicted == 'positive')/len(y_predicted))*100),2)
        negative_percentage = round(((sum(y_predicted == 'negative')/len(y_predicted))*100),2)
        neutral_percentage = round(((sum(y_predicted == 'neutral')/len(y_predicted))*100),2)

        # create pie chart
        plt.close('all')
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_percentage, negative_percentage, neutral_percentage]
        colors = ['#00ff00', '#ff0000', '#808080']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Analysis Results for '+csvFileName+' data using Naive Bayes')

        # save plot to a file, replacing it if it already exists
        plot_file = csvFileName+'sentiment_analysis_plot.png'
        if os.path.exists('.\\static\\'+plot_file):
            os.remove('.\\static\\'+plot_file)
        plt.savefig('.\\static\\'+plot_file)


        return render_template('naive_bayes_result.html', positive=positive_percentage, neutral = neutral_percentage, negative = negative_percentage, plot_file=plot_file)
    except:
        tb_str = traceback.format_exc()
        return render_template('failure.html',reason = 'to compute Sentiments', traceback = tb_str)

@app.route('/perform_naive_accuracy', methods = ['POST'])
def naive_acc() :
    try:
        def preprocess_text(text):
            # Remove mentions and links
            text = re.sub(r'@\S+|https?://\S+', '', text)
            # Remove non-alphabetic characters and convert to lowercase
            text = re.sub('[^a-zA-Z]', ' ', text.lower())
            # Tokenize text
            tokens = nltk.word_tokenize(text)
            # Remove stopwords and stem remaining words
            stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
            # Join stemmed tokens back into a single string
            preprocessed_text = ' '.join(stemmed_tokens)
            return preprocessed_text
        
        
        # Assign sentiment labels
        def get_sentiment_label(sentiment):
            if sentiment == 'positive':
                return 1
            elif sentiment == 'negative':
                return -1
            else:
                return 0

        # Load data
        nblabelled_csv = 'nblabelled_'+csvFileName
        data = pd.read_csv('.\\CSV Files\\'+nblabelled_csv, encoding='utf-8')
        # Preprocess data
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        data['Tweet'] = data['Tweet'].apply(preprocess_text)
        data['sentiment_label'] = data['sentiment'].apply(get_sentiment_label)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data['sentiment_label'], test_size=0.2, random_state = 42)
        # Extract features
        vectorizer = CountVectorizer()
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)
        # Train model
        model = MultinomialNB()
        model.fit(X_train_features, y_train)
        # Evaluate model
        y_pred = model.predict(X_test_features)
        accuracy = round((accuracy_score(y_test, y_pred) * 100),2)
        return render_template('naive_accuracy.html', accuracy = accuracy, csv=csvFileName)
    except :
        tb_str = traceback.format_exc()
        return render_template('failure.html',reason = 'to compute Accuracy', traceback = tb_str)

if __name__ == '__main__':
    app.run(debug=True)
