import pandas as pd
from util import symbol_to_path
from flask_wtf import FlaskForm
from wtforms import IntegerField, DecimalField, SubmitField, RadioField, SelectField
from wtforms import validators


# Form class
class StartValuesForm(FlaskForm):
    start_val = IntegerField('Initial Capital', [validators.DataRequired("Please enter a value.")])
    symbol = SelectField('Stock Symbol')
    commission = DecimalField('Commision')
    impact = DecimalField('Impact')
    num_shares = IntegerField('Shares number', [validators.DataRequired("Please enter number of shares.")])
    submit = SubmitField("Send")
    forecast = RadioField('Forecast type', choices = [('forecastprices','Price'),('showvalues','Price movements')], default='Price')

    def get_tickers(setf, filename):
        df = pd.read_csv(symbol_to_path(filename), usecols=['Symbol'])
        return df