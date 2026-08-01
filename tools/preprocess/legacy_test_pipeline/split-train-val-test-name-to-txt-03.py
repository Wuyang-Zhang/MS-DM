import os
import random
import shutil


# 指定图片文件夹路径和三个txt文件名
img_folder_path = r"data\raw_test\fruit_fly\images"
txt1_name = r"data\raw_test\fruit_fly\train.txt"
txt2_name = r"data\raw_test\fruit_fly\val.txt"
txt3_name = r"data\raw_test\fruit_fly\test.txt"


# 记住 生成果蝇 第二次 不要运行这个文件 需要 train 和 val 的名称对应上 这个会生成随机名称 对不上

# 定义要移动的文件和目标文件夹的路径
# source_file_path = r'test_images-pre-result-full\images'
# destination_folder_path = r'data\images'

# 使用shutil.copy2()函数复制文件
# shutil.copy2(source_file_path, destination_folder_path)




# 打开三个txt文件，用于写入图片文件名
with open(txt1_name, 'w') as txt1, open(txt2_name, 'w') as txt2, open(txt3_name, 'w') as txt3:
    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(img_folder_path):
        # 只处理图片文件
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 去掉文件后缀，得到不包含后缀的文件名
            name = os.path.splitext(filename)[0]
            # 生成一个随机数，决定将文件写入哪个txt文件
            rand_num = random.random()
            if rand_num <= 1:
                txt3.write(name + "\n")


print('=====done======')
