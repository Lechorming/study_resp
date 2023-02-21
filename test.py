import pandas as pd

# 读取数据
data = pd.read_csv("test.csv")

# 获取数据集中的含有类别变量的列的列名，并用列表表示
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

# 移除含有类别变量的列
drop_data = data.select_dtypes(exclude=['object'])

print(drop_data);
