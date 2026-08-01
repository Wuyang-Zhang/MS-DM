import os
import shutil



import os



# 设置新的权限模式
mode = 0o777




# 设置父文件夹路径
parent_folder = r'test_images-pre-result-full\test-all-ori'

# 获取子文件夹列表
# subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
subfolders = os.listdir(parent_folder)

# 遍历子文件夹
for subfolder in subfolders:
    if subfolder == 'test-both':

        # 对于名字为folder1的子文件夹，执行操作1
        # 操作1示例：将子文件夹内的所有文件拷贝到指定文件夹

        for i in os.listdir(os.path.join(parent_folder,subfolder)):

            if i == 'ff':
                # 更改目录的权限
                # os.chmod(os.path.join(parent_folder,subfolder,i), mode)

                shutil.copy(os.path.join(parent_folder,subfolder,i),r'test_images-pre-result-full\point2txt_ff')
                shutil.copy(os.path.join(parent_folder,subfolder,'img'),r'test_images-pre-result-full\test_images_ff')
            elif i == 'WF' :
                shutil.copy(os.path.join(parent_folder,subfolder,i),r'test_images-pre-result-full\point2txt') # wf txt
                shutil.copy(os.path.join(parent_folder,subfolder,'img'),r'test_images-pre-result-full\test_images')
            else:
                pass

    elif subfolder == 'test-ff':

                shutil.copy(os.path.join(parent_folder,subfolder,'FF'),r'test_images-pre-result-full\point2txt_ff') # wf txt
                shutil.copy(os.path.join(parent_folder,subfolder,'img'),r'test_images-pre-result-full\test_images_ff')

    elif subfolder == 'test-wf':
                shutil.copy(os.path.join(parent_folder,subfolder,'WF'),r'test_images-pre-result-full\point2txt') # wf txt
                shutil.copy(os.path.join(parent_folder,subfolder,'img'),r'test_images-pre-result-full\test_images')


print('done')
