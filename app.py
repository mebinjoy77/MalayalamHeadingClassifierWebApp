import flask
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
with open(f'models/mallu_model.pkl','rb') as m:
    model = pickle.load(m)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('main.html'))
@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv('train.csv')
    data1= data.headings
    cv = CountVectorizer()
    xc = cv.fit_transform(data1)
    if flask.request.method == 'POST':
        head = flask.request.form['heading']
        dat = [head]
        vect = cv.transform(dat).toarray()
        mypred = model.predict(vect)
    return flask.render_template('result.html',prediction = mypred)


    
if __name__ == '__main__':
    app.run(debug=True)
