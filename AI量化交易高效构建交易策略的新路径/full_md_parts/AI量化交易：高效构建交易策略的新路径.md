# AI量化交易：高效构建交易策略的新路径

![](images/8f380be3b78bf2a1d710dfc784cc823c29c15febaea67c4135c6764146ebafb0.jpg)  
图3.5 数据分析与模型的构建流程

代码示例：计算基础特征。

Python   
import numpy as np   
def calc_features(df): #换手率 df['turnover_rate'] $=$ df['vol']/df['float_share'] #振幅 df['amplitude'] $=$ (df['high']-df['low'])/ df['pre_close'] #量比 df['volume_ratio'] $=$ df['vol']/ df['vol'].rolling(5).mean() #收盘价与均价偏离度 df['avg_price'] $=$ df['amount']/df['vol'] df['close.bias'] $=$ (df['close']-df['avg_price'])/ df['avg_price'] #价格波动率 df['volatility'] $=$ df['close'].rolling(10).std() #资金流向（假设有大单成交额字段） df['capital_flow'] $=$ df['big_order_amount']- df['small_order_amount'] #量价相关系数 df['corr_vol_price'] $=$ df['vol'].rolling(10).corr(df['close']) #省略其他特征 return df
