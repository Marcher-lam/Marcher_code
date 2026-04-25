# 获取近一年内所有A股的日线数据  
end_date = datetime today().strftime('%Y%m%d')  
start_date = (datetime today() - timedelta(days=365)).strftime('%Y%m%d')  
df = pro.daily( trades_date=start_date, end_date=end_date) 
```

获取涨跌停数据

```python
limit df = pro.stk_limit-trade_date=start_date, 
```

end_date $\equiv$ end_date)

合并涨停信息

```python
df = df.merge limit df[['ts_code', 'trade_date', 'up_limit'], on=['ts_code', 'trade_date'], how='left')  
df['is_limit_up'] = df['close'] >= df['up_limit'] 
```

统计连板数

```python
df = df.sort_values(['ts_code', 'trade_date'])  
df['limit_up_count'] =  
df.groupby('ts_code') ['is_limit_up'].cumsum() 
```

找出5连板及以上个股及其首板日期

```python
def find_first_board(group):
    boards = group[group['is_limit_up}]
    if len(boards) >= 5:
        return boards.iloc[0]
    return None
first Boards =
df.groupby('ts_code').applyfind_first_board).dropna()
first Boards = first Boards[['ts_code', 'trade_date', 'close',
'turnover_rate', 'total_mv']) 
```
