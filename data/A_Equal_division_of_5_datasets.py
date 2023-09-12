# 将文件夹内的图片 顺序打乱后平均分成五分  要求设置随机种子

import os
import random
import shutil

# 设置随机种子以确保可复现的随机结果
random_seed = 42
random.seed(random_seed)

# 原始图片文件夹路径
original_folder = r"H:\Pest_monitoring_program\cross_validationForTrain\raw\img_split"

# 打乱图片文件顺序
image_files = os.listdir(original_folder)
random.shuffle(image_files)

# 创建五个目标文件夹，用于存放分割后的图片
output_folders = [f"cross_validationForTrain/ValOutput{i}/img" for i in range(1, 6)]

K_floder_num = 5

# 计算每个目标文件夹应有的图片数量
num_images = len(image_files)
num_per_folder = num_images // K_floder_num
assert num_per_folder*5 <= num_images 
print(f"总共有{num_images}张图片，每一个文件夹里有{num_per_folder}个文件")

# 分割图片并复制到目标文件夹
for i in range(K_floder_num):

    print(f"现在正在迭代第{i+1}次循环")


    if i == 0 :  # 修改数据以训练5折
        print("============================================================")
        print(f"这里生成的是5折交叉验证的第{i+1}折的验证集的数据")
        print("============================================================")


        # 这里是根据第几次随机抽取的训练集图片
        start_index = i * num_per_folder
        end_index = start_index + num_per_folder
        if i == len(output_folders) - 1:
            end_index = num_images  # 最后一个文件夹包含余下的图片
        selected_images = image_files[start_index:end_index]
        


        for image in selected_images:

            source_path = os.path.join(original_folder, image)
            WFsource_path = os.path.join(original_folder.replace("img_split","wf_split"), image.replace(".jpg",".txt"))
            FFsource_path = os.path.join(original_folder.replace("img_split","ff_split"), image.replace(".jpg",".txt"))
            target_path = os.path.join(r"data\images", image)
            WFtarget_path = os.path.join(r"data\mats", image.replace(".jpg",".txt"))
            FFtarget_path = os.path.join(r"data\mats-1", image.replace(".jpg",".txt"))
            shutil.copyfile(source_path, target_path)
            shutil.copyfile(FFsource_path, FFtarget_path)
            shutil.copyfile(WFsource_path, WFtarget_path)