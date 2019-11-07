from flask import Flask,render_template,url_for,request,flash
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from wtforms import validators
import re as regex

app = Flask(__name__,template_folder='template')
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config.from_object(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("spam data merger.csv")
    df_data = df[["CONTENT","CLASS"]]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        comment = request.form['hash1']
        if(comment==""):
            flash('Data is required')
            return home()

        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        jugde=""
        sp=0
        if(my_prediction==0):
            jugde="Spam"
        elif(my_prediction==1):
             jugde="Ham"
        num_sentence = len(regex.split("[!?.]+", comment))
        num_word=len(regex.split("\s",comment))
        s1=len(regex.findall("[aeiouyAEIOUY]+",comment))
        s2=len(regex.findall("[^aeiouyAEIOUY]+[eE]\\b",comment))
        s3=len(regex.findall("\\b[^aeiouyAEIOUY]*[eE]\\b",comment))
        num_syllable=s1-(s2-s3)
        flesch=206.835 - 1.015 *  num_word / num_sentence - 84.6 *  num_syllable / num_word
        if(flesch>=90):
               read="Very Easy"
               grade="5th Level"
        elif(flesch>=80):
               read="Easy"
               grade="6th Level"
        elif(flesch>=70):
               read="Fairly Easy"
               grade="7th-8th Level"
        elif(flesch>=60):
               read="Standard"
               grade="10th Level"
        elif(flesch>=50):
              read="Flairly Difficult"
              grade="12th Level"
        elif(flesch>=30):
              read="Difficult"
              grade="College Level"
        elif(flesch<29):
              read="Very Confusing"
              grade="Graduate Level"
        return render_template('/home.html',prediction = jugde,num_word=len(regex.split("\s",comment)),num_sentence=num_sentence,syllable=num_syllable,grade=grade,read=read,comment=comment)



if __name__ == '__main__':
    app.run(host='127.0.0.3', port=3000,debug=True)