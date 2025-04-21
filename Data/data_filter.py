import pandas as pd

df = pd.read_csv('btcusd_bitstamp_1min_2012-2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df[df['timestamp'] >= '2017-01-01']  # 新增过滤条件
df.set_index('timestamp', inplace=True)
daily_df = df.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

daily_df.reset_index(inplace=True)
daily_df['timestamp'] = daily_df['timestamp'].dt.date

daily_df = daily_df[daily_df['timestamp'] >= pd.to_datetime('2017-01-01').date()]

daily_df.to_csv(
    'bitcoin_daily_since_2017.csv',
    index=False,
)

print(f"data range：{daily_df['timestamp'].min()} 至 {daily_df['timestamp'].max()}")