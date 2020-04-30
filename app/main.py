print(f"__file__ = {__file__:<35} | __name__ = {__name__:<20} | __package__ = {str(__package__):<20}")

import pandas_datareader.data as web
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'Stock Graph'

app.layout = html.Div([
	html.H1('Stock Graph'),

	html.Div([
		html.Div('Symbol to graph:'),
		dcc.Input(id = 'symbol', value = '', type = 'text'),
		html.Div('Price to graph:'),
		dcc.Dropdown(id = 'price',
			options = [{'label': i, 'value': i} for i in ['High', 'Low', 'Open', 'Close', 'Adj Close', 'All']],
			value = 'Adj Close'
		),
		html.Div('_____________'),
		html.Button('Submit', id = 'submit'),
	], 
	style = {'width': '13%', 'display': 'inline-block'}),

	html.Div(id = 'output-graph'),
])

# Retrieve symbol price data set from Yahoo
def retrieve_dataset(symbol):
	start = datetime.datetime(2015, 1, 1)
	end = datetime.datetime.now()
	df = web.DataReader(symbol, 'yahoo', start, end)
	df.reset_index(inplace = True)
	df.set_index('Date', inplace = True)
	return df

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

	if price == 'All':
		g1 = go.Candlestick(
					x = df.index, 
					open = df['Open'],
					high = df['High'],
					low = df['Low'],
					close = df['Close'],
					name = 'Price')
	else:
		g1 = go.Scatter(x = df.index, y = df[price], name = price)
	fig.add_trace(g1, row = 1, col = 1)

	g2 = go.Bar(x = df.index, y = df['Volume'], name = 'Volume')
	fig.add_trace(g2, row = 5, col = 1)

	# centered title, left aligned by default
	fig.update_layout(title_text = symbol, title_x = 0.5)
	fig.update(layout_xaxis_rangeslider_visible=False)
	return fig

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
			figure = go_figure(df, symbol, price)
		)
	except Exception as e:
		print(f'[update_output_graph] exception: {str(e)}')

if __name__ == '__main__':
	app.run_server(debug = True)

print("---end of main---")    