from flask import Flask, render_template, session, jsonify, request, flash
from form import StartValuesForm
import pandas as pd
import numpy as np  
import random
import datetime as dt
from sklearn import datasets, svm

# To fetch data
from pandas_datareader import data as pdr   
import fix_yahoo_finance as yf  
yf.pdr_override()  

from util import create_df_benchmark, get_data, fetchOnlineData

app = Flask(__name__)


# Add these extra two lines
app.secret_key = 'your secret'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/')
def home():
    return render_template(
    # name of template
    "index.html"
    )


# Load Dataset from scikit-learn.
digits = datasets.load_digits()

# Show values
@app.route('/showvalues', methods=['POST', 'GET'])
def showvalues():
    # Specify the start and end dates for this period.
    start_d = dt.datetime(2008, 1, 1)
    #end_d = dt.datetime(2018, 10, 30)
    yesterday = dt.date.today() - dt.timedelta(1)
    
    # Get portfolio values from Yahoo
    symbol = request.form['symbol']
    portf_value = fetchOnlineData(start_d, yesterday, symbol)
    
    return render_template(
    # name of template
	"showvalues.html",
        
    # now we pass in our variables into the template
    start_val = request.form['start_val'],
    symbol = request.form['symbol'],
    commission = request.form['commission'],
    impact = request.form['impact'],
    num_shares = request.form['num_shares'],
    start_date = start_d, 
    end_date = yesterday,
    tables=[portf_value.to_html(classes=symbol)],
    titles = ['na', 'Price']    
    )  

# Initial form to get values
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

