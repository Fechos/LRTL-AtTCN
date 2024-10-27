import pandas as pd

for i in range(20):
    filename = 'processed_data/train_W_No' + str(i) + '.csv'
    # 读取数据
    df = pd.read_csv(filename)

    # 指定列名
    columns = ['temperature', 'dew_point', 'humidity', 'air_pressure', 'wind_speed', 'OT']

    # 使用 for 循环对指定列执行检查和处理
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 标识异常值
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        # 删除标识为异常值的行
        df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        # 线性插值
        df[column] = df[column].interpolate(method='linear')

        # 截断
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        # 将处理后的数据写入新的 CSV 文件
    cleaned_filename = filename.replace('processed_data', 'final_data')
    df.to_csv(cleaned_filename, index=False)


