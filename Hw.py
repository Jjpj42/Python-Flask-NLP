import glob
import itertools
import os
from collections import Counter, defaultdict
from fileinput import filename
import shutil
from markupsafe import Markup
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from flask import Flask, flash, redirect, render_template, request
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
import spacy
nlp = spacy.load('en_core_web_sm')
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import pandas as pd
import tokenizer
import spacy
from spacy import displacy
from textblob import TextBlob
doc = nlp(u'This is a sentence.')
model_path = "fake-news-jtrp"

model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer =AutoTokenizer.from_pretrained(model_path)

app= Flask(__name__)
#เปลี่ยน path file ตามที่เราเอาโฟลเดอร์ Doc ไปลง
app.config["UPLOAD_PATH"] = r"C:\Users\MR.TANATORN\Desktop\Python-Flask-NLP-main\Doc"
i=1
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")



@app.route("/upload_file", methods=["GET","POST"])
def upload_file():
    
    if request.method == 'POST' and len(request.files.getlist('Docfile')) != 0:
        articles = []
        i=1
        #การเช็ค direct ว่ามี folder หรือ file มั้ย ถ้ามีอยู่แล้ว มันจะลบตัวเดิมออกละใส่ตัวใหม่เข้าไป
        for filename in os.listdir(r'C:\Users\MR.TANATORN\Desktop\Python-Flask-NLP-main\Doc'):
                file_path = os.path.join(r'C:\Users\MR.TANATORN\Desktop\Python-Flask-NLP-main\Doc', filename)
        
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    gg = 0
        # save ไฟล์ โดยการเปลี่ยนชื่อ 
        for Doc_file in request.files.getlist('Docfile'):
            Doc_file.filename= f"File{i}.txt"
            Doc_file.save(os.path.join(app.config['UPLOAD_PATH'],Doc_file.filename))
            i = i+1 
        
        
        i=0
        
        Path_direc = r'C:\Users\MR.TANATORN\Desktop\Python-Flask-NLP-main\Doc'
        BOW=[]
        Spa=""
        # เรียกอ่านไฟล์และทำ Back of Word 
        for Doc_file in os.listdir(Path_direc):
            f = open(f"./Doc/File{i+1}.txt", "r")
            print(f)
            article = f.read()
            Spa += article + "\n"
            # Tokenize the article: tokens
            tokens = word_tokenize(article)
            # Convert the tokens into lowercase: lower_tokens
            lower_tokens = [t.lower() for t in tokens]
            # Retain alphabetic words: alpha_only
            alpha_only = [t for t in lower_tokens if t.isalpha()]
            # Remove all stop words: no_stops
            no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
            # Instantiate the WordNetLemmatizer
            wordnet_lemmatizer = WordNetLemmatizer()
            # Lemmatize all tokens into a new list: lemmatized
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
            #list_article
            articles.append(lemmatized)
            BOW.append(Counter(no_stops).most_common(1))
            i+=1
            
            
        #ค้นหาคำในไฟล์ 
        dictionary = Dictionary(articles)
        #print(dictionary)
        key_word = request.form['Word']
        #แปลงจากเลข id ของข้อความเป็นข้อความ
        result_word = dictionary.token2id.get(key_word)
        check = True
        if result_word == None :
            check = False
        #print(check)    
        corpus = [dictionary.doc2bow(a) for a in articles]
        #print(corpus)
        top_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            top_word_count[word_id] += word_count
        sorted_word_count = sorted(top_word_count.items(), key=lambda w: w[1],reverse=True)
  
        #tfidf
        tfidf = TfidfModel(corpus)
        tfidf_weights = []
        for doc in corpus : tfidf_weights+=(tfidf[doc])
        Top5 = []
        sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        
        for term_id, weight in sorted_tfidf_weights[:5]:
            Top5.append(f'{dictionary.get(term_id)}, {weight}')


        
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(Spa)
        show_spa = displacy.render(doc,style='ent')
        
        
        return render_template("index.html",msg="Files has been uploaded successfully",Show=BOW ,tf_idf= sorted_tfidf_weights[:5],top_tfidf=Top5,bool_check=check,key_word=key_word,show_spa=Markup(show_spa))  
        
        
    return render_template("index.html",msg="Please choose file") 
    
@app.route('/fakenews_print', methods = ['GET', 'POST'])
def fakenews_print():
    if request.method == 'POST':
        def get_prediction(text, convert_to_label=False):
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512,return_tensors="pt")
            # perform inference to our model
            outputs = model(**inputs)
            # get output probabilities by doing softmax
            probs = outputs[0].softmax(1)
            # executing argmax function to get the candidate label
            d = {0: "reliable",1: "fake"}
            if convert_to_label:
                return d[int(probs.argmax())]
            else:
                return int(probs.argmax())
        word1 = request.form['word1']
        
        msg1 = str(get_prediction(word1, convert_to_label=True ))

        
    return render_template('index.html',msg1 = msg1)

@app.route('/sentiment', methods = ['GET', 'POST'])
def sentiment():
    text = request.form['Sentiment']
    import nltk
    nltk.download('vader_lexicon') 
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text))))['compound']

    if(score > 0):
        label = 'This sentence is positive'
    elif(score == 0):
        label = 'This sentence is neutral'
    else:
        label = 'This sentence is negative'

    return render_template('index.html',varible = label)






if __name__ == '__main__':
    app.run(port=5500, debug = True)