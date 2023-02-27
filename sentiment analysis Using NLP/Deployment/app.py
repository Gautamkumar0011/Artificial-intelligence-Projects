from flask import Flask, render_template, request
import io
import requests     # Importing request to extract content from a URL
from bs4 import BeautifulSoup as bs    # BeautifulSoup is for web scrapping...
import re
import os
from PIL import Image
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/')
def home():
    
    return render_template("index.html")

@app.route('/success', methods = ["post"])
def success():
    tm = request.form["urlc"]
 
    iphone12_reviews = []
    
    for i in range(1,21):
        ip = []
        url = tm + str(i)
        response = requests.get(url)
        soup = bs(response.content, "html.parser")
    # creating soup object to iterate over the extracted contant
        reviews = soup.find_all("span", attrs = {"class", "a-size-base review-text review-text-content"})
    # Extracting the content under specific tags
        for i in range(len(reviews)):
            ip.append(reviews[i].text)
            
        iphone12_reviews = iphone12_reviews + ip
    # write reviews in to a text file
    with open("iphone12.txt", "w", encoding = 'utf8') as output:
        output.write(str(iphone12_reviews))
        
    #join all the reviews into single paragraph
    ip_rev_string = " ".join(iphone12_reviews)
    
    # remove unwanted characters from raw ext if exists
    ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
    
    ip_reviews_words = ip_rev_string.split(" ")
    
    ip_reviews_words = ip_reviews_words[1:]
    # ignoring the first empty space
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer( use_idf = True, ngram_range = (1, 1))
    bag_of_words = vectorizer.fit_transform(ip_reviews_words)
    
    from nltk.corpus import stopwords
    top_w = stopwords.words("English")
    top_w.extend(["Amazon", "iphone12", "time", "ios", "phone", "device", "product", "day"])
    ip_reviews_words = [w for w in ip_reviews_words if not w in top_w]
    ip_rev_string = " ".join(ip_reviews_words)
    
    
    wordcloud = WordCloud(background_color = 'White',
                          width = 1800,
                          height = 1400
                         ).generate(ip_rev_string)
    wordcloud.to_file("static/pics/corpus_wordcloud.png")
    
    
    with open("positive-words.txt", 'r') as pos:
      poswords = pos.read().split("\n")
      
    ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

    wordcloud_pos_in_pos = WordCloud(
                           background_color = 'White',
                           width = 1800,
                           height = 1400
                          ).generate(ip_pos_in_pos)
    wordcloud_pos_in_pos.to_file("static/pics/positive_words.png")
     
     
    with open("negative-words.txt",'r') as neg:
       negwords = neg.read().split("\n")
       
    ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

    wordcloud_neg_in_neg = WordCloud(
                            background_color = 'white',
                            width = 1800,
                            height = 1400
                           ).generate(ip_neg_in_neg)
    wordcloud_neg_in_neg.to_file("static/pics/negative_words.png")
      
      
    imageList = os.listdir('static/pics')
    imagelist = ['pics/' + image for image in imageList]
    return render_template("data.html", imagelist = imagelist)
  
if __name__ == '__main__':
    
    app.run(debug = True)
    