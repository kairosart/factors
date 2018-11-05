from flask import Flask, render_template, session
import random
from sklearn import datasets, svm

app = Flask(__name__)

# Add these extra two lines
app.secret_key = 'your secret'
app.config['SESSION_TYPE'] = 'filesystem'

def factors(num):
    return [x for x in range(1, num+1) if num%x==0]

@app.route('/')
def home():
    n = random.randint(2, 10000)
    return render_template(
    # name of template
    "index.html",
    # now we pass in our variables into the template
    random_num=n, 
    )

@app.route('/factors/<int:n>', methods=['POST', 'GET'])
def factors_display(n):
	return render_template(
    # name of template
	"factors.html",
    # now we pass in our variables into the template
	number=n, 
	factors=factors(n) 
	)

# Load Dataset from scikit-learn.
digits = datasets.load_digits()

@app.route('/predict')
def hello():
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    prediction = clf.predict(digits.data[-1:])

return jsonify({'prediction': repr(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

