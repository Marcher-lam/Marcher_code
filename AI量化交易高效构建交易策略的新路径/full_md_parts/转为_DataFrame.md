# 转为 DataFrame
df_topi = pd.DataFrame({
    '题材': list(topi_count.keys'),
    '涨停家数': list(topi_count.values'),
    '封单总额': [topic_amount[k] for k in topic_count.keys()
})
df_topi = df_topi.sort_values(['涨停家数', '封单总额'],
ascending=False)
return df_topi
