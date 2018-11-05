from flask import Flask, render_template, session, jsonify, request, flash
from form import StartValuesForm

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

@app.route('/predict', methods=['POST', 'GET'])
def hello():
    
    return render_template(
    # name of template
	"strategy.html",
    # now we pass in our variables into the template
    start_val = 100000,
    symbol = "AMZN",
    commission = 0.00,
    impact = 0.0,
    num_shares = 1000
    )  

@app.route('/form', methods = ['GET', 'POST'])
def introStartValues():
    form = StartValuesForm()
   
    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('startValuesForm.html', form = form)
        else:
            return render_template('success.html')
    elif request.method == 'GET':
        return render_template('startValuesForm.html', form = form)



if __name__ == '__main__':
    app.run(host='0.0.0.0')

