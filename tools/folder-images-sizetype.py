# -*- coding:utf-8 -*-

# 需要知道这些图片尺寸大小有几类


import os
from PIL import Image
import pandas as pd

wh_list = []

root_path = input('请输入 图片 所在地址:')
suffix = ['.jpg', '.png','jpeg' , 'bmp']   # 如果图片后缀比较多（png jpg jpeg bmp），可以在 suffix里进行添加


img_list = os.listdir(root_path )

for img in img_list:
    img = Image.open(os.path.join(root_path,img))
    w,h =  img.size[0],img.size[1]
    wh_list.append((w,h))


wh_list_deduplicate = list(set(wh_list))  # 去重
wh_list_deduplicate.sort(key = wh_list.index)  # 通过列表中索引（index）的方法保证去重后的顺序不变

print( '一共有' , len(wh_list_deduplicate) , '类型的图片size尺寸')
print('======种类有========')
for i , wwhh in enumerate(wh_list_deduplicate) :
    print(i,':'  ,wwhh)



print('==done===')
