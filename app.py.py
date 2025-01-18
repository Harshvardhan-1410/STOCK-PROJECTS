import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st



from yahoo_fin import stock_info as si


# Set the title of the app
st.title("STOCK TREND PREDICTOR")

# Get user input for stock ticker
USER_INPUT = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data based on user input
df = si.get_data(USER_INPUT, start_date='2010-01-01', end_date='2023-12-31')

df=df.reset_index()
df=df.rename(columns={'index':'date'})
df=df.drop(columns=['date','adjclose'])

# Display the fetched data
st.write(df)

#visualizations
st.subheader('closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)


st.subheader('closing Price vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(ma200,'r')
plt.plot(df.close,'b')
st.pyplot(fig)


#splitting  data into testing and training

data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)






#load model

model = load_model('my_model.keras')


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test = np.array(x_test), np.array(y_test)

#making predictions

y_predicted = model.predict(x_test)
scaler.scale_

scale_factor = 1/0.00646057
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


#final fig
st.subheader("Predicted vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)