import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yaml
#LSTM Model Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout




def Predictor():
    # Title
    st.title('Stock Market Analysis Predictor App')
    # Subheader
    st.subheader('This app allows you to analyze the stock market and predict the future price of a stock of your choice.')
    # Add Image
    st.image('https://miro.medium.com/v2/resize:fit:720/format:webp/1*NpT5pyemQQsGEHXbfS51Zw.png')

    #Take the input from the user about start and end date
    #Sidebar
    st.sidebar.header('User Input Features')

    start_date = st.sidebar.date_input("Start date", date(2014, 12, 10))
    end_date = st.sidebar.date_input("End date", date.today()) 

    #Add ticker symbol list
    ticker_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'INTC', 'CMCSA', 'PEP', 'CSCO', 'AVGO', 'TMUS', 'QCOM', 'TXN', 'CHTR', 'AMGN', 'SBUX', 'GILD', 'MDLZ', 'FISV', 'INTU', 'BKNG', 'ISRG', 'ADP', 'VRTX', 'REGN', 'AMD', 'MU', 'ATVI', 'ILMN', 'CSX', 'ADSK', 'ADI', 'BIIB', 'LRCX', 'MELI', 'AMAT', 'ADP', 'CTSH', 'NXPI', 'WBA', 'EBAY', 'MAR', 'WDAY', 'KLAC', 'EXC', 'ROST', 'ORLY', 'EA', 'LULU', 'SNPS', 'MNST', 'KHC', 'XEL', 'IDXX', 'CDNS', 'DOCU', 'PCAR', 'ANSS', 'ALXN', 'PAYX', 'SIRI', 'XLNX', 'VRSK', 'NTES', 'CPRT', 'FAST', 'DLTR', 'INCY', 'CTAS', 'SWKS', 'MXIM', 'CERN', 'CHKP', 'BMRN', 'TCOM', 'ULTA', 'FOXA', 'FOX', 'NTAP', 'SGEN', 'VRSN', 'WDC', 'CDW', 'CTXS', 'MXIM', 'KLAC', 'CTSH', 'CDNS']

    # if st.sidebar.button('Choose a Ticker'):
    ticker = st.sidebar.selectbox('Select Ticker Symbol', ticker_list)
        

    #Get the data from yahoo finance library

    data = yf.download(ticker,start= start_date,end= end_date)

    #Add date as column in the dataframe
    data.insert(0, 'Date', data.index, True)
    data.reset_index(drop=True, inplace=True)
    st.write("Data from start data", start_date, "to end date", end_date)
    st.write(data)
    #Plot the data
    st.title("Data Visualization")
    st.subheader("Plot the data")
    fig = px.line(data, x='Date', y=data.columns, title='Price of '+ticker, width=800, height=600)
    st.plotly_chart(fig)
    #Select the column to be used for prediction
    # Get the list of columns
    columns = list(data.columns[1:])

    # Find the index of 'Close' in the list
    default_index = columns.index('Close') if 'Close' in columns else 0

    # Create the selectbox with 'Close' selected by default
    column = st.selectbox('**Note:** Select Column to be used for prediction', columns, index=default_index)

    # Selecting the data
    data = data[['Date', column]]
    #Plot the data
    st.write(data)

    #ADF Test for Stationarity
    st.subheader("ADF Test for Stationarity")
    st.write("p-value: ", sm.tsa.stattools.adfuller(data[column])[1]<0.05)
    p_value = sm.tsa.stattools.adfuller(data[column])[1]
    st.write("p-value: ", p_value < 0.05)
    if p_value < 0.05:
        st.write("The series is stationary.")
    else:
        st.write("The series is not stationary.")
    decomposition = seasonal_decompose(data[column], model='additive', period=12)
    st.write(decomposition.plot())
    #Make same plot using plotly
    TrendFig = px.line(x=data['Date'], y=decomposition.trend, title='Trend', width=800, height=600)
    TrendFig.update_traces(line_color="red")
    st.plotly_chart(TrendFig)

    SeasonalityFig = px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', width=800, height=600)
    SeasonalityFig.update_traces(line_color="green")
    st.plotly_chart(SeasonalityFig)
    ResidualFig = px.line(x=data['Date'], y=decomposition.resid, title='Residual', width=800, height=600)
    ResidualFig.update_traces(line_color="red", line_dash="dot")
    st.plotly_chart(ResidualFig)

    #Lets run the SARIMA model
    # User input for three parameters of model and seasonal order
    p = st.slider('Select the value of p', 0, 5, 2)
    q = st.slider('Select the value of q', 0, 5, 1)
    d = st.slider('Select the value of d', 0, 5, 2)

    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

    #Model 
    model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))

    model = model.fit()

    #Print the summary of the model
    st.subheader("Model Summary")
    st.write(model.summary())
    st.write('____')

    st.write('<p style="color:Blue; font-size: 50px font-weight: bold;">Model Prediction</p>', unsafe_allow_html=True)

    #Predict the future values
    forcast_period = st.number_input('Select the number of days for prediction', 1, 365, 10)
    #predict the future values
    prediction = model.get_prediction(start=len(data), end=len(data)+forcast_period)
    prediction = prediction.predicted_mean
    prediction.name = 'Prediction'
    prediction = pd.DataFrame(prediction)
    st.write(prediction)

    #Add index to the prediction
    prediction.index = pd.date_range(start=data['Date'].iloc[-1], periods=len(prediction), freq='D')
    prediction = pd.DataFrame(prediction)
    prediction.insert(0, 'Date', prediction.index, True)
    prediction.reset_index(drop=True, inplace=True)
    st.write("Prediction", prediction)
    st.write("__________________________")

    st.write('<p style="color:Blue; font-size: 50px font-weight: bold;">Prediction Visualization</p>', unsafe_allow_html=True)

    #lets plot the prediction
    fig=go.Figure()
    #Add actual data to the plot
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual Data', line=dict(color='blue')))
    #Add prediction to the plot
    fig.add_trace(go.Scatter(x=prediction['Date'], y=prediction['Prediction'], mode='lines', name='Prediction', line=dict(color='red')))

    # Display the figure in Streamlit
    st.plotly_chart(fig)
    #Set the title and axis labels
    st.write("__________________________")
    #Calculate prediction using LSTM model
    st.write('<p style="color:Blue; font-size: 50px font-weight: bold;">LSTM Model Prediction</p>', unsafe_allow_html=True)
    #Create a new dataframe
   # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Prepare the data for LSTM
    n_input = 14  # number of steps
    n_features = 1  # number of features
    generator = TimeseriesGenerator(data_normalized, data_normalized, length=n_input, batch_size=6)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(LSTM(100, activation='relu'))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    # model.fit(generator, epochs=200, batch_size=8)  # Increase epochs and decrease batch size
    # model.save('lstm_model.h5')
    
    model = load_model('lstm_model.h5')

    # Predict the future values
    pred_list = []

    batch = data_normalized[-n_input:].reshape((1, n_input, n_features))

    for i in range(forcast_period):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

    # Transform the predicted values back to the original scale
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list), index=pd.date_range(start=data['Date'].iloc[-1], periods=forcast_period, freq='D'))

    # Add 'Date' column to the prediction dataframe
    df_predict.reset_index(inplace=True)
    df_predict.columns = ['Date', 'Prediction']

    st.write("LSTM Prediction", df_predict)
    
    st.write('<p style="color:Blue; font-size: 50px font-weight: bold;">LSTM Prediction Visualization</p>', unsafe_allow_html=True)

    # Let's plot the LSTM prediction
    fig = go.Figure()

    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual Data', line=dict(color='blue')))

    # Add LSTM prediction to the plot
    fig.add_trace(go.Scatter(x=df_predict['Date'], y=df_predict['Prediction'], mode='lines', name='LSTM Prediction', line=dict(color='red')))

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)
            


def Simulator():


    if "window_size" not in st.session_state:
        st.session_state.window_size = 365


    def calc_portfolio(x, num_holds):
        pf = 0
        for k, v in num_holds.items():
            pf += x[f"Close_{k}"] * v
        return pf


    def calc_stock(num_holds, stocks, ratio, portfolio):
        # calc portfolio value
        stocks["Close_Portfolio"] = stocks.apply(
            lambda x: calc_portfolio(x, num_holds), axis=1
        )
        # convert to int
        for k in num_holds.keys():
            stocks[f"Close_{k}"] = np.floor(
                pd.to_numeric(stocks[f"Close_{k}"], errors="coerce")
            ).astype("Int64")
        stocks["Close_Portfolio"] = np.floor(
            pd.to_numeric(stocks["Close_Portfolio"], errors="coerce")
        ).astype("Int64")
        # recent value ratio
        recent_valid_index = (
            stocks.dropna(subset=["Close_Portfolio"]).tail(1).index.values[0]
        )
        # recent value percent
        recent_values = []
        recent_ratios = []
        for k in portfolio["ticker"].keys():
            value = stocks.loc[recent_valid_index, f"Close_{k}"] * num_holds[k]
            ratio = value / stocks.loc[recent_valid_index, "Close_Portfolio"] * 100
            value = round(value, 2)
            ratio = round(ratio, 2)
            recent_values.append(value)
            recent_ratios.append(ratio)
        ratio = pd.DataFrame(
            data={
                "ticker": portfolio["ticker"].keys(),
                "latest_value_sum": recent_values,
                "ratio_percent": recent_ratios,
            }
        )
        ratio["type"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["type"])
        ratio["detail"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["detail"])
        ratio["sector"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["sector"])
        ratio["num_holds"] = ratio.ticker.apply(lambda x: num_holds[x])
        ratio["latest_value"] = ratio.latest_value_sum / ratio.num_holds

        # calc sharpe ratio
        sharpe = stocks.loc[:, ["Date", "Close_Portfolio"]]
        sharpe = sharpe.dropna(subset=["Close_Portfolio"])
        sharpe["lag_1d"] = sharpe["Close_Portfolio"].shift(1)
        sharpe["rate_change"] = np.log(sharpe.Close_Portfolio / sharpe.lag_1d)
        sharpe["one_year_mean"] = sharpe["rate_change"].rolling(252).mean()
        sharpe["one_year_std"] = sharpe["rate_change"].rolling(252).std()
        sharpe["sharpe_ratio"] = sharpe.one_year_mean / sharpe.one_year_std
        sharpe["sharpe_ratio_annual"] = sharpe["sharpe_ratio"] * 252**0.5
        sharpe = sharpe.dropna(subset=["sharpe_ratio_annual"])
        sharpe = sharpe.loc[:, ["Date", "sharpe_ratio_annual"]]

        return sharpe, stocks, ratio


    def layout_input(sharpe, stocks, ratio, portfolio):
        num_holds = {}
        for t in ratio.ticker:
            num_holds[t] = st.number_input(t, 0, key=t)
        num_holds = {k: float(v) for k, v in num_holds.items()}
        if st.button("Calculate"):
            sharpe, stocks, ratio = calc_stock(num_holds, stocks, ratio, portfolio)
        return sharpe, stocks, ratio


    # comment off cache to avoid warning at streamlit cloud
    # @st.cache
    def read_stock_data_from_local():
        stocks = pd.read_pickle("data/stocks.pkl")
        ratio = pd.read_pickle("data/ratio.pkl")
        sharpe = pd.read_pickle("data/sharpe.pkl")
        return sharpe, stocks, ratio


    def plot_stock_data(df: pd.DataFrame, window: int, x: str, y: str, title: str) -> None:
        """plot line chart"""
        # drop nan values
        df = df.dropna(subset=y)
        df = df.tail(window)
        if window > 400:
            df = df[::2]
        fig = px.line(df, x=x, y=y)
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)


    def layout_plots(sharpe, stocks, ratio) -> None:
        """manipulate plots layout"""
        col1, col2 = st.columns(2)

        # plot portfolio values by time series
        # with col1:
        col0, col1, col2, col3, col4 = st.columns([6, 1, 1, 1, 1])
        with col0:
            st.write("Portfolio Overall Performance")
        with col1:
            if st.button("3Year", key="portfolio_3year"):
                st.session_state.window_size = 1080
        with col2:
            if st.button("Year", key="portfolio_year"):
                st.session_state.window_size = 360
        with col3:
            if st.button("Quarter", key="portfolio_quarter"):
                st.session_state.window_size = 90
        with col4:
            if st.button("Month", key="portfolio_month"):
                st.session_state.window_size = 30
        plot_stock_data(
            df=stocks,
            window=st.session_state.window_size,
            x="Date",
            y="Close_Portfolio",
            title="Portfolio Transition",
        )

        # plot sharpe ratio
        st.write("Portfolio Annual Sharpe Ratio (Baseline = 1)")
        fig = px.line(
            sharpe.tail(st.session_state.window_size), x="Date", y="sharpe_ratio_annual"
        )
        fig.add_hline(1, line_color="red")
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

        # plot portfolio share by pie chart
        fig = px.sunburst(
            ratio,
            path=["type", "ticker"],
            values="ratio_percent",
            title="Portfolio Recent Value Ratio",
        )
        st.plotly_chart(fig, use_container_width=True)

        # table for each stock stat
        st.write("Portfolio Detail")
        st.table(
            ratio.loc[
                :,
                [
                    "ticker",
                    "detail",
                    "type",
                    "sector",
                    "num_holds",
                    "latest_value",
                    "latest_value_sum",
                    "ratio_percent",
                ],
            ]
        )

        for ticker, detail in zip(ratio.ticker.values, ratio.detail.values):
            col0, col1, col2, col3, col4 = st.columns([6, 1, 1, 1, 1])
            with col0:
                st.write(f"{ticker}: {detail}")
            with col1:
                if st.button("3Year", key=f"{ticker}_3year"):
                    st.session_state.window_size = 1080
            with col2:
                if st.button("Year", key=f"{ticker}_year"):
                    st.session_state.window_size = 360
            with col3:
                if st.button("Quarter", key=f"{ticker}_quarter"):
                    st.session_state.window_size = 90
            with col4:
                if st.button("Month", key=f"{ticker}_month"):
                    st.session_state.window_size = 30
            plot_stock_data(
                df=stocks,
                window=st.session_state.window_size,
                x="Date",
                y=f"Close_{ticker}",
                title=f"{ticker}: {detail}",
            )


    if __name__ == "__main__":

        # general layout settings
        
        st.title("ETF Portfolio Simulator")

        # load sharpe, stocks, and ratio pickles
        sharpe, stocks, ratio = read_stock_data_from_local()

        # load portfolio settings
        with open("portfolio.yaml", "rb") as file:
            portfolio = yaml.safe_load(file)

        # layout sidebar
        with st.sidebar:
            st.write("Input Portfolio To Simulate")
            sharpe, stocks, ratio = layout_input(sharpe, stocks, ratio, portfolio)

        # layout plots
        layout_plots(sharpe, stocks, ratio)


def main():
    st.set_page_config(page_title="My App", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")

    st.title('Trading Insights Analysis App')

    st.markdown('## Navigation')
    selected_page = st.selectbox('Go to:', ['Home', 'Simulator', 'Predictor'])
    
    if 'Simulator' in selected_page:
        Simulator()

    if 'Predictor' in selected_page:
        Predictor()


if __name__ == "__main__":
    main()