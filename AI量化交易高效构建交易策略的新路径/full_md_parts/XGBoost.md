# XGBoost  
xgb = XGB regressor(n_estimators=100, random_state=42)  
xgb.fit(X_train, y_train)  
xgbImportances = xgb feature-importances_ 
```

输出特征重要性排序   
（2）可视化  
importances $=$ pd.DataFrame({ 'feature': features, 'RF Importance':rf-importances, 'XGB Importance':xgb-importances   
}）.sort_values('XGB Importance',ascending $\equiv$ False) print(importantances)

Python   
import matplotlib.pyplot as plt   
importances.set_index('feature')['RF Importance', 'XGB-importance'].plot(kind $\equiv$ 'barh'，figsize=(10,6)) plt.title('Feature Importance (RF vs XGBoost)') plt.xlabel('Importance') plt.show()

（3）构建非线性收益预测模型

- 以XGBoost为例，利用筛选后的重要特征构建未来收益的预测模型。  
- 输出回测结果、信号生成逻辑。

Python   
from sklearn.metrics import mean_squared_error   
y_pred $\equiv$ xgb.predict(X_test)   
mse $\equiv$ mean_squared_error(y_test,y_pred)   
print(f'XGBoost Test MSE:{mse:.6f}')

生成买入信号（如预测收益大于0，则买入）df_test $\equiv$ dfiloc[y_test.index].copy()

```python
df_test['pred_return'] = y_pred  
df_test['signal'] = (df_test['pred_return'] > 0).astype(int) 
```
