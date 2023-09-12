import numpy as np

# 指定.npy文件的路径
file_path = r'data\data-used-by-train-val-test\train\0002_01.npy'

# 使用np.load()加载.npy文件
loaded_data = np.load(file_path)

# 现在，loaded_data变量中包含了.npy文件中的数据
print(loaded_data)  #  just ----list
 
