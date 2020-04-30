print(f"__file__ = {__file__:<35} | __name__ = {__name__:<20} | __package__ = {str(__package__):<20}")

# Refer to below URL on the steps to deploy to heroku
# https://dash.plotly.com/deployment
from app.main import app

if __name__ == "__main__":
    app.run_server(debug = True)
    
print("---end of wsgi---")    