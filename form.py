from datetime import date, datetime

import pandas as pd
from util import symbol_to_path
from flask_wtf import FlaskForm
from wtforms import IntegerField, DecimalField, SubmitField, RadioField, SelectField, HiddenField, DateField
from wtforms import validators, ValidationError
import datetime as dt

def get_tickers(filename):
    df = pd.read_csv(symbol_to_path(filename), usecols=['Symbol'])
    return df


# Trading agent
class StartValuesForm(FlaskForm):
    d2 = dt.date.today() - dt.timedelta(days=365)
    loookbakc_date = DateField('Lookback Date (mm/dd/yyyy)',
                             format='%m/%d/%Y',
                             default=d2,
                             validators=[validators.DataRequired()])
    start_val = IntegerField('Initial Capital', [validators.DataRequired("Please enter a value.")])
    symbol = SelectField('Stock Symbol')
    commission = DecimalField('Commision')
    impact = DecimalField('Impact')
    num_shares = IntegerField('Shares number', [validators.DataRequired("Please enter number of shares.")])
    submit = SubmitField("Send")





# Price forecasting
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


