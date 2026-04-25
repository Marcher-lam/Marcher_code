# 假设 df 已经包含所有特征和目标变量 'future_return'   
features = ['turnover_rate', 'amplitude', 'volume_ratio', 'close.bias', 'volatility', 'capital_flow', 'corr_vol_price', 'tail_strength', 'abnormal_vol_ratio', 'netBIG_order_rate']   
X = df[features].fillna(0)   
y = df['future_return']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
```

随机森林  
```python
rf = RandomForestRegressor(n_estimators=100, random_state=42)  
rf.fit(X_train, y_train)  
rfImportances = rf.feature-importances_
