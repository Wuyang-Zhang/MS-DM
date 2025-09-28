import json
import os
import scipy.io as scio
import numpy as np
import scipy.io as scio
import os

#============================主要功能及其作用========================================================================
# 生成 .mat
# 文件路径中保证没有空格

# 直接读取json文件中获取point 过于麻烦   --------  将点保存在txt文件中在获取  所以读取的时txt文件
#===================================================================================================================================
# 输出图片的文件夹路径
bug_mat_path = os.path.abspath(r'data\mats')
image_mat = []

image_path = []
#  txt txt txt txt txt 路径！！！！！！！
img_path = os.path.abspath(r'data\mats')
image_json  = os.listdir(img_path)  
root_path =  img_path
# 用来存储所有的points信息
points_store = []
# 用来存储不同图片的信息
points_layer = []

for i in range (0,len(image_json)):

    # 获取每个txt文件的路径    
    print(image_json[i])
    txt_path = os.path.join(root_path,image_json[i])

    with open(txt_path, 'r') as load_f:  # encoding='UTF-8'
        lines = load_f.readlines()   #每次读取一行内容 
        for line in lines:
            point_item = []
            c_x = float(line.strip().split(' ')[0]) 
            c_y = float(line.strip().split(' ')[1])
            point_item.append(c_x)
            point_item.append(c_y)
            points_store.append(point_item)

        # 将每一张图片中的信息作为一个新维度存储起来 方便取出
        points_store = np.array(points_store)
        points_layer.append([[points_store]])
        # 列表清空 用于存储第二张图片的points信息
        points_store = []

    image_num_mat = image_json[i].split('.')[0]

    #print(image_num_mat)
    image_mat = image_num_mat + '.mat'
    mat_path = os.path.join(bug_mat_path,image_mat)  # 构建每一张mat的路径 跟源程序格式相同哦
    # print(mat_path)
    points_layer[-1] = np.array(points_layer[-1])
    # 如果文件中有空格" "替换成 "_"
    mat_path = mat_path.replace(" ", "_")
    print(mat_path)
    scio.savemat(mat_path,{'image_info': points_layer[i]})

#-------------------------------------------------------------
# 官方给出的 形式为 # gt['image_info'][0,0] = [[(array([[   ]]))]]   自己制作为 [list([[[168.11594202898553, 614.4927536231885]]])] ok改好了


print('---------image_num_mat保存已完成--------------------')

