#y：未来3个季度净利润变动方向（1=上修，0=持平，-1=下修） $\mathrm{X} = \mathrm{df\_fin}[[\mathrm{'impact\_feature'}',\mathrm{'roe\_chg'},\mathrm{'cashflow\_chg'},$ 'r_and_d_ratio']].values   
y $=$ df_fin['profit_revision'] #需提前构造好标签   
X_train，X_test，y_train，y_test $=$ train_test_split(X,y,   
test_size=0.2, random_state=42)   
clf $=$ LogisticRegression(multi_class='multinomial'，

```python
max_iter=200)  
clf.fit(X_train, y_train)  
print("模型训练准确率：", clf.score(X_test, y_test)) 
```
