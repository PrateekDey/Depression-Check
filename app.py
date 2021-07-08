from flask import Flask, render_template, request
import pickle

# change the filename before executing
clf = pickle.load(
    open(r'C:\Users\Win10\Documents\ML\Depression\Model\model.pkl', 'rb'))

# change the filename before executing
tfidf = pickle.load(
    open(r'C:\Users\Win10\Documents\ML\Depression\Model\vect.pkl', 'rb'))


def estimate(x):
    vec = tfidf.transform([x])
    rs = clf.predict(vec)
    return rs[0]


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getresult', methods=['GET', 'POST'])
def getresult():
    if request.method == 'POST':
        text = request.form['text']
        rs = estimate(text)
        return str(rs)


if __name__ == '__main__':
    app.run(debug=True)
