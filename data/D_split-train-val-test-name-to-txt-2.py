import os
import random
import shutil


# 指定图片文件夹路径和三个txt文件名
img_folder_path = r"data\temp"
txt1_name = r"data\train.txt"
txt2_name = r"data\val.txt"
txt3_name = r"data\test.txt"


# 定义要移动的图片文件文件夹的路径
# source_file_path = r'H:\Pest_monitoring_program\image_prepocessing\a_split_overlapp_result\cross_validation\val_output1\img'
# destination_folder_path = r'data\images'

 
# 打开验证集txt文件，用于写入图片文件名
with   open(txt2_name, 'w') as txt2:
    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(img_folder_path):
        # 只处理图片文件
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 去掉文件后缀，得到不包含后缀的文件名
            name = os.path.splitext(filename)[0]
            # 如果文件中有空格" "替换成 "_"
            name = name.replace(" ", "_") 

            # 生成一个随机数，决定将文件写入哪个txt文件
            rand_num = random.random()
            # 只改写验证集
            if rand_num < 1.0:
                txt2.write(name + "\n")


print('=====done======')