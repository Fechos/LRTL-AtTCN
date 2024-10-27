import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt

data = pd.read_csv('data/final_data/train_W_No0.csv', parse_dates=['date'], index_col='date')
columns = data.columns
print(columns)
for column in columns:
    if column != 'OT':
        # 平稳性检验（ADF检验）
        def adf_test(series, signif=0.05, name=''):
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            print(f'ADF检验结果（{name}）：p-value = {p_value:.4f}')
            return p_value < signif
        # ggg
        
        # 检验数据的平稳性
        print(" 检查数据的平稳性：")
        is_stationary_temperature = adf_test(data[column], name=column)
        is_stationary_OT = adf_test(data['OT'], name='OT')
        
        # 如果不平稳，需要进行差分处理
        if not is_stationary_temperature:
            data[column] = data[column].diff().dropna()
            is_stationary_temperature = adf_test(data[column].dropna(), name=f'{column} (Differenced)')
        if not is_stationary_OT:
            data['OT'] = data['OT'].diff().dropna()
            is_stationary_OT = adf_test(data['OT'].dropna(), name='OT (Differenced)')
        
        # Granger 因果分析
        max_lag = 4  # 设置最大滞后阶数
        print("\n进行Granger 因果分析：")
        granger_result = grangercausalitytests(data[['OT', column]].dropna(), max_lag)
        
        # 解释结果
        print("\nGranger 因果分析结果解释：")
        for i in range(1, max_lag + 1):
            f_test_p_value = granger_result[i][0]['ssr_ftest'][1]
            if f_test_p_value < 0.05:
                print(f"滞后{i}阶下，{column} Granger 导致 OT，p-value = {f_test_p_value:.4f}")
            else:
                print(f"滞后{i}阶下，{column} 不 Granger 导致 OT，p-value = {f_test_p_value:.4f}")


