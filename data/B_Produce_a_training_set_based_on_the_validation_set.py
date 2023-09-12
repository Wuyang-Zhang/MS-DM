# 根据验证集制作训练集

# 有两个文件夹，并且想要遍历第一个文件夹，将.jpg文件复制到第三个文件夹，但只有当.jpg文件不在第二个文件夹中时才复制，可以使用以下Python代码来实现：


import os
import shutil




source_folder =  r"H:\Pest_monitoring_program\cross_validationForTrain\ed\img_split"  
FFsource_folder =  r"H:\Pest_monitoring_program\cross_validationForTrain\ed\ff_split"  
WFsource_folder =  r"H:\Pest_monitoring_program\cross_validationForTrain\ed\wf_split" 

output_folders = r"data\images"
WFoutput_folders = r"data\mats"
FFoutput_folders = r"data\mats-1"

# 列出第一个文件夹中的所有图片文件文件
files_in_source = os.listdir(source_folder)
print(f"训练集验证集一共有{len(os.listdir(output_folders) )}张验证的图片")

# 遍历第一个文件夹中的文件
for file_name in files_in_source:
    if file_name.endswith(".jpg"):
        source_file_path = os.path.join(source_folder, file_name)
        target_file_path = os.path.join(output_folders, file_name)

        WFsource_file_path = os.path.join(WFsource_folder, file_name.replace(".jpg", ".txt"))
        WFtarget_file_path = os.path.join(WFoutput_folders, file_name.replace(".jpg", ".txt"))

        FFsource_file_path = os.path.join(FFsource_folder, file_name.replace(".jpg", ".txt"))
        FFtarget_file_path = os.path.join(FFoutput_folders, file_name.replace(".jpg", ".txt"))

        
        # 检查.jpg文件是否不在第二个文件夹中
        if not os.path.exists(os.path.join(output_folders, file_name)):
            # 复制文件到第三个文件夹
            shutil.copyfile(source_file_path, target_file_path)
            shutil.copyfile(WFsource_file_path, WFtarget_file_path)
            shutil.copyfile(FFsource_file_path, FFtarget_file_path)           
            print(f"复制 {file_name} 到第文件夹")

files_in_source = os.listdir(output_folders)
print(f"训练集验证集一共有{len(files_in_source )}张图片")
# 输出完成消息
print(f"训练集验证集补充完成")
