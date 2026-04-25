#df_zt:columns $\equiv$ ['date'，'code'，'name'，'zt_time'，'amount'， 'tags'，'net_inflow'，'zt_count'] #tags为分号分隔的题材标签，如"机器人；人工智能"   
def classify_zt_reasons(df_zt):   
    ""归类涨停原因，统计各题材的涨停家数与封单总额   
    ""topic_count $=$ defaultdict(int) topic_amount $=$ defaultdict(float) for _,row in df_zt,iterrrows():

![](images/d40298079f3ea91f4c80f570e12b3466ce344298400f3f4c9e318c4e30d1222f.jpg)  
图3.14 涨停股筛选与归因分析流程

（2）识别新题材首板股  
```python
tags = str(row['tags'].split ';'')
for tag in tags:
    if tag:
        topic_count[U] += 1
        topic_amount[U] += row['amount']
