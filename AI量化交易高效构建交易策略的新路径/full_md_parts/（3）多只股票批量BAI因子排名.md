# （3）多只股票批量BAI因子排名

假设有多只A股的LV2数据，批量计算过去5日BAI因子排名前10的股票：

Python   
import glob   
results $= []$ for file in glob.glob('orderbook_data/\*.csv'): stock_code $\equiv$ file.split('/')[-1].split（.）[0] df $\equiv$ pd.read_csv(file，parse_dates $\equiv$ ['timestamp']) df['BAI'] $=$ (df['bid1_vol']-df['ask1_vol'])/ (df['bid1_vol'] $^+$ df['ask1_vol']+1e-6) mean_bai $=$ df['BAI'].mean() results.append(['code':stock_code,'mean_bai':mean_bai})   
top10 $=$ sorted(results, key $\equiv$ lambda x:- abs(x['mean_bai'])[:10] print('过去5日BAI因子排名前10的A股：')   
for item in top10: print(item)
