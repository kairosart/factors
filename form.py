from datetime import date, datetime

import pandas as pd
from util import symbol_to_path
from flask_wtf import FlaskForm
from wtforms import IntegerField, DecimalField, SubmitField, RadioField, SelectField, StringField, DateField
from wtforms import validators


def get_tickers(filename):
    df = pd.read_csv(symbol_to_path(filename), usecols=['Symbol'])
    return df

# Price movements Form class
class StartValuesForm(FlaskForm):
    start_val = IntegerField('Initial Capital', [validators.DataRequired("Please enter a value.")])
    symbol = SelectField('Stock Symbol')
    commission = DecimalField('Commision')
    impact = DecimalField('Impact')
    num_shares = IntegerField('Shares number', [validators.DataRequired("Please enter number of shares.")])
    submit = SubmitField("Send")
    forecast = RadioField('Forecast type',
                          choices = [('forecastprices','Price'),('showvalues','Price movements')],
                          default='forecastprices')


class pricesForecast(FlaskForm):
    forecastDate = DateField('Forecast Date',
                             format='%m/%d/%Y',
                             default=datetime.today(),
                             validators=[validators.DataRequired()])
    symbol = SelectField('Stock Symbol', coerce=int)
    modelSelection = SelectField('Select model')
    forecastTime = SelectField('Forecast')
    lookback = SelectField('Lookback')
    submit = SubmitField("Send")


