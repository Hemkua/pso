import pandas as pd

# 读取输入文件
df = pd.read_csv('gamma和c.csv')

# 提取第一列和第二列中的小数
col1 = df['A'].str.extract(r'(\d+\.\d+)', expand=False)
col2 = df['B'].str.extract(r'(\d+\.\d+)', expand=False)

# 将结果保存到输出文件
output = pd.DataFrame({'A': col1, 'B': col2})
output.to_csv('output1.csv', index=False)
