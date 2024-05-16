import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64 
from nltk.sentiment import SentimentIntensityAnalyzer
UPLOAD_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template("upload.html")


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        f = request.files.get('file')
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('success.html')


@app.route('/show_data')
def showData():

    data_file_path = session.get('uploaded_data_file_path', None)
    df = pd.read_csv(data_file_path, encoding='unicode_escape')
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for text in df['reviewText']:
        sentiment = sia.polarity_scores(str(text))
        sentiment_scores.append(sentiment['compound'])

    df['sentiment_score'] = sentiment_scores
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral')

    sentiment_counts = df['sentiment_category'].value_counts()
    sentiments = sentiment_counts.index
    counts = sentiment_counts.values
    char_remove = ['_', '-', 'data', 'Data', 'DATA', 'csv', 'CSV', '.','static\\']
    
    my = data_file_path
    for i in char_remove:
        if i in my:
            my = my.replace(i, ' ')

    
    plt.bar(sentiments, counts)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis of ' + my.replace('.csv', ''))
    plt.savefig(data_file_path + '.png')
    img = f"{data_file_path}.png"
    return render_template('data.html', image_path=img)

@app.route('/upload')
def upload():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)