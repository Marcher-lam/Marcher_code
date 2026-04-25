# 假设数据文件为CSV格式，包含columns：['datetime'，'open'，'high'，'low'，'close'，'volume']  
df = pd.read_csv('minute_data.csv', parse_dates=[[‘datetime’])  
df = df.sort_values('datetime').reset_index.drop=True)
