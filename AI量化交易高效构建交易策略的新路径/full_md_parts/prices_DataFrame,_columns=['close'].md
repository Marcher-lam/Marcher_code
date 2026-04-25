# prices: DataFrame, columns=['close'] 
```

1. 状态与动作空间离散化

```python
state_bins = {
    'volatility': np-percentile(state['volatility'],
[33, 66])
    'liquidity': np-percentile(state['liquidity'],
[33, 66])
} def get_state(row):
    v_bin = np.digital(row['volatility'],
state_bins['volatility']
    l_bin = np.digital(row['liquidity'],
state_bins['liquidity']
    return (v_bin, l_bin) 
```

![](images/80c20b7ef1b62cdd61b7b1154271ed7aa4b71b7cf21e995b91e44fdcc007246c.jpg)
