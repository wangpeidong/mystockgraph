print(f"__file__ = {__file__:<35} | __name__ = {__name__:<20} | __package__ = {str(__package__):<20}")

import pandas_datareader.data as web
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.DARKLY])
app.title = 'Stock Graph'

app.layout = html.Div([
		html.H1('Stock Graph', style = {'textAlign': 'center'}),

		html.Div([
			html.Div(['Symbol to graph:', 
					 dcc.Input(id = 'symbol', value = '^GSPC', type = 'text', style = {'display': 'block'})],
					 style = {'textDecoration': 'underline'}),

			html.Div(['Price to graph:', 
					 dcc.Dropdown(id = 'price',
						options = [{'label': i, 'value': i} for i in ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Ohlc', 'Forecast']],
						value = 'Adj Close',
						style = {'color': 'black'}
					 )],
					 style = {'textDecoration': 'underline'}),
			
			dbc.Button('Submit', id = 'submit', color = 'primary', className = 'mr-1', style = {'marginTop': 5}),
		], 
		style = {'display': 'inline-grid', 'marginLeft': 8}),

		html.Div([
			html.Div(id = 'output-graph')
		],
		style = {'height': '100%'})
	]
)

# Save previous symbol and data set, in case
# sysmbol is the same as prevous one, return
# prevoius data set directly.
prev_symbol = ''
prev_df = None

def data_regression(df):
	df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
	df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

	forecast_col = 'Adj Close'
	df.fillna(value = -99999, inplace = True)
	forecast_out = 10 # 10 days
	df['label'] = df[forecast_col].shift(-forecast_out)

	X = np.array(df.drop(['label'], 1))
	X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X_2nd_lately = X[-forecast_out:] 
	X = X[:-forecast_out * 2] # Feature
	y = np.array(df['label'][:-forecast_out * 2]) # Label

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	clf = LinearRegression() # Classifier
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)

	last_col = len(df.columns)
	df['Forecast'] = np.nan

	# Comparing set with Adj Close using 2nd X_lately period feature
	comparing_set = clf.predict(X_2nd_lately) 
	for idx, val in enumerate(comparing_set):
		df.iloc[-forecast_out + idx, last_col] = val

	# Future set based on X_lately period feature
	forecast_set = clf.predict(X_lately)
	last_date = df.iloc[-1].name
	last_unix = last_date.timestamp()
	one_day = 86400 # seconds in a day
	next_unix = last_unix + one_day
	for val in forecast_set:
		next_date = datetime.datetime.fromtimestamp(next_unix)
		# exclude weekend
		while next_date.weekday() > 4:
			next_unix += 86400
			next_date = datetime.datetime.fromtimestamp(next_unix)

		next_unix += 86400
		df.loc[next_date, 'Forecast'] = val

	return accuracy, df

# Retrieve symbol price data set from Yahoo
def retrieve_dataset(symbol):
	global prev_symbol, prev_df
	if prev_symbol != '' and prev_symbol == symbol and prev_df is not None:
		return prev_df

	start = datetime.datetime(2015, 1, 1)
	end = datetime.datetime.now()
	df = web.DataReader(symbol, 'yahoo', start, end)
	df.reset_index(inplace = True)
	df.set_index('Date', inplace = True)

	df['5-DMA'] = df['Adj Close'].rolling(window = 5, min_periods = 0).mean()
	df['30-DMA'] = df['Adj Close'].rolling(window = 30, min_periods = 0).mean()

	prev_symbol = symbol
	prev_df = df

	return df

def create_fig_layout(fig, symbol):
	# centered title, left aligned by default
	fig.update_layout(title_text = symbol, title_x = 0.5, 
				plot_bgcolor = '#222', paper_bgcolor = '#222', font = dict(color = 'orangered'),
				xaxis = dict(
					rangeselector = dict(
						buttons = list([
							dict(count = 5,
								 label = '1w',
								 step = 'day',
								 stepmode = 'backward'),
							dict(count = 10,
								 label = '2w',
								 step = 'day',
								 stepmode = 'backward'),
							dict(count = 1,
								 label = '1m',
								 step = 'month',
								 stepmode = 'backward'),
							dict(count = 3,
								 label = '3m',
								 step = 'month',
								 stepmode = 'backward'),
							dict(count = 6,
								 label = '6m',
								 step = 'month',
								 stepmode = 'backward'),
							dict(count = 1,
								 label = 'YTD',
								 step = 'year',
								 stepmode = 'todate'),
							dict(count = 1,
								 label = '1y',
								 step = 'year',
								 stepmode = 'backward'),
							dict(step = 'all')
						])
					),
					rangeslider_visible = False
	))
	# rangebreak does not work with update_layout somehow
	fig.update_xaxes(
		rangebreaks = [
				dict(bounds = ['sat', 'mon'])
			]
	)
	return fig

def go_figure(df, symbol, price):
	fig = make_subplots(
		rows = 5, cols = 1, 
		specs = [
					[{'rowspan': 4}],
					[None],
					[None],
					[None],
					[{}]
				],
		shared_xaxes = True, 
		vertical_spacing = 0.02)

	if price == 'Ohlc':
		g1 = go.Candlestick(
					x = df.index, 
					open = df['Open'],
					high = df['High'],
					low = df['Low'],
					close = df['Close'],
					name = 'Ohlc')
	elif price == 'Forecast':
		accuracy, df = data_regression(df.copy())
		print(f'LinearRegression accuracy: {accuracy}')
		g1 = go.Scatter(x = df.index, y = df["Adj Close"], name = "Adj Close", marker = dict(color = 'blue'))
		g1_forecast = go.Scatter(x = df.index, y = df["Forecast"], name = "Forecast", marker = dict(color = 'red'))
		fig.add_trace(g1_forecast, row = 1, col = 1)
	else:
		g1 = go.Scatter(x = df.index, y = df[price], name = price, marker = dict(color = 'blue'))
		if (price == 'Adj Close'):
			g1_d30 = go.Scatter(x = df.index, y = df['30-DMA'], name = '30-DMA', marker = dict(color = 'red'))
			g1_d5 = go.Scatter(x = df.index, y = df['5-DMA'], name = ' 5-DMA', marker = dict(color = 'green'), fill='tonexty')
			fig.add_trace(g1_d30, row = 1, col = 1)
			fig.add_trace(g1_d5, row = 1, col = 1)

	fig.add_trace(g1, row = 1, col = 1)
	g2 = go.Bar(x = df.index, y = df['Volume'], name = 'Volume', marker = dict(color = 'white'))
	fig.add_trace(g2, row = 5, col = 1)

	return create_fig_layout(fig, symbol)

@app.callback(
	Output(component_id = 'output-graph', component_property = 'children'),
	[Input('submit', 'n_clicks')], 
	state = [State(component_id = 'symbol', component_property = 'value'), State(component_id = 'price', component_property = 'value')]
)
def update_output_graph(n_clicks, symbol, price):
	if n_clicks is None or symbol == '' or price == '':
		return
	print(f'You entered symbol {symbol}, price {price}, and clicked {n_clicks} times')
	try:
		df = retrieve_dataset(symbol)

		return dcc.Graph(
			id = 'stock-graph',
			figure = go_figure(df, symbol, price),
			style = {'backgroundColor': 'black', 'color': 'white', 'height': 800}
		)
	except Exception as e:
		print(f'[update_output_graph] exception: {str(e)}')

if __name__ == '__main__':
	app.run_server(debug = True)

print("---end of main---")    