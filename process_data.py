import numpy as np

# 初始化用于存储数据的列表
acc_list = []
macro_f1_list = []

# 读取文件并处理数据
with open('res.txt', 'r') as file:
    lines = file.readlines()

# 假设文件中每项数据都是由空格分隔的，而且我们只关心数字部分
for i, line in enumerate(lines):
    # 移除不需要的行尾字符，然后分割字符串
    numbers = line.strip().split()
  
    # 只有当我们有两部分才处理，忽略其它格式的行
    if len(numbers) == 3:
        seed, run_name, num = numbers
        num = float(num)  # 转换为浮点数
        if i < 5:
            acc_list.append(num)
        elif i > 5 and i < 11:
            macro_f1_list.append(num)

# 计算平均值和标准差
acc_mean = np.mean(acc_list)
acc_std = np.std(acc_list)

macro_f1_mean = np.mean(macro_f1_list)
macro_f1_std = np.std(macro_f1_list)

# 打印结果
# print(acc_list)
# print(macro_f1_list)
# print(micro_f1_list)

print(f'Accuracy {round(acc_mean*100, 2)}+-{round(acc_std*100, 2)}')
# 输入顺序换了，所以得改
print(f'Macro F1 - Mean: {round(macro_f1_mean*100, 2)}+-{round(macro_f1_std*100, 2)}')