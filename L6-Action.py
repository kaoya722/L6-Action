import pandas as pd

###数据加载
train = pd.read_csv('D:/Python/TEST/train.csv')
# print(train)

###转换为pandas中时间日期格式
train['Datetime'] = pd.to_datetime(train['Datetime'])
train.index = train['Datetime']
train.drop(['ID', "Datetime"], axis=1, inplace=True)
# print(train)

### 按天采样
daily_train = train.resample('D').sum()
# print(daily_train)
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train['Count']
daily_train.drop(['Count'], axis=1, inplace=True)
# print(daily_train)
from fbprophet import Prophet
###创建模型
m =Prophet(yearly_seasonality=True,seasonality_prior_scale=0.1)
m.fit(daily_train)
###预测未来7个月
future=m.make_future_dataframe(periods=213)
print(future)
forecast=m.predict(future)
# print(forecast)
