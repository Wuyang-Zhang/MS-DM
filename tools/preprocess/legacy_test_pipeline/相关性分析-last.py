import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import PySide2

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


'''
读取txt文件中的每一行数据。
将每一行数据按照空格进行切割。
取出切割后的列表中倒数第二个和最后一个元素，并将其转换为整数类型。
将取出的整数存储在列表中，用于后续的相关性分析。
'''
# real  pre

# txt 形式 ：
#   img_20156235   8548   5698
#   img_20152515   8548   5698
#   ......

data = []  # 存储提取出的数据

with open(r'test_images-pre-result-full\result\wf_result_logger_ours.txt', 'r') as f:
    for line in f:
        line_data = line.strip().split(' ')
        num1, num2 = float(line_data[-2]), float(line_data[-1])
        data.append((num1, num2))

# 进行相关性分析，这里以计算相关系数为例

data = np.array(data)
# 将列表转为numpy数组，并进行拟合
slope, intercept, r_value, p_value, std_err = stats.linregress(data[:, 0], data[:, 1])


correlation_matrix = np.corrcoef(data[:, 0], data[:, 1])
# 计算相关系数r
correlation_xy = correlation_matrix[0,1]
r = correlation_xy

# 绘制拟合曲线和散点图
plt.plot(data[:, 0], data[:, 1], 'o', label='original data')
plt.savefig(r'test_images-pre-result-full\result\wf-original-data.png')
plt.plot( data[:, 0], intercept + slope*np.array( data[:, 0]), 'r', label='fitted line')
plt.savefig(r'test_images-pre-result-full\result\wf-fitted-line.png')
plt.legend()
plt.show()

print(f"拟合曲线的斜率为: {slope}")
print(f"拟合曲线的截距为: {intercept}")
print(f"相关系数r为: {r}")

with  open (r"test_images-pre-result-full\result\wf-r2.txt",'w') as  f :
    f.write(f'y = {slope}x + {intercept}\n ')
    f.write(f'相关系数r为: {r}')






#=========================================================================

data1 = []  # 存储提取出的数据

with open(r'test_images-pre-result-full\result\ff-result_logger_ours.txt', 'r') as f:
    for line in f:
        line_data = line.strip().split(' ')
        num1, num2 = float(line_data[-2]), float(line_data[-1])
        data1.append((num1, num2))



# 进行相关性分析，这里以计算相关系数为例

data1= np.array(data1)
# 将列表转为numpy数组，并进行拟合
slope, intercept, r_value, p_value, std_err = stats.linregress(data1[:, 0], data1[:, 1])


correlation_matrix = np.corrcoef(data1[:, 0], data1[:, 1])
# 计算相关系数r
correlation_xy = correlation_matrix[0,1]
r = correlation_xy

# 绘制拟合曲线和散点图
plt.plot(data1[:, 0], data1[:, 1], 'o', label='original data')
plt.savefig(r'test_images-pre-result-full\result\ff-original-data.png')
plt.plot( data1[:, 0], intercept + slope*np.array( data1[:, 0]), 'r', label='fitted line')
plt.savefig(r'test_images-pre-result-full\result\ff-fitted-line.png')
plt.legend()
plt.show()

print(f"拟合曲线的斜率为: {slope}")
print(f"拟合曲线的截距为: {intercept}")
print(f"相关系数r为: {r}")

with  open (r"test_images-pre-result-full\result\ff-r2.txt",'w') as  f :
    f.write(f'y = {slope}x + {intercept}\n ')
    f.write(f'相关系数r为: {r}')
