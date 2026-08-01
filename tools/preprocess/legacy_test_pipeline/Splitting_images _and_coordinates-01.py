from PIL import Image
import os
import  cv2

# ok
# 作用 :
#       将大分辨率的图片 分割成小的 同时分割注释信息（txt文件 内涵xy坐标信息）  不同尺寸 分割比例不同


#-----------------input-------------------------------------------------
path_txt = r'data\raw_test\whitefly\annotations'
image_path = r'data\raw_test\whitefly\images'
path_txt_save = r'data\raw_test\whitefly\tiles\annotations'
save_path_img = r"data\raw_test\whitefly\tiles\images"
#----------------------------------------------------------------------------





txt_list = os.listdir(path_txt)
total_num_crop = 0

# 定义分割后的块数
split_num = {
    (3264, 2448): 3,
    (2448,3264 ): 3,
    (3468, 4624): 4,
    (4624,3468 ): 4,
    (6936, 9248): 8,
    (9248,6936 ): 8,
    (3472, 4624): 4,
    (4624, 3472): 4,
    (6944, 9248): 8,
    (9248, 6944): 8,
}



for txt in txt_list:

    txt_name =txt.split('.')[0]




    # 每一张图片txt分割单独保存
    if not os.path.exists (os.path.join(save_path_img,txt_name)):
        os.mkdir(os.path.join(save_path_img,txt_name))

    if not os.path.exists (os.path.join(path_txt_save,txt_name)):
        os.mkdir(os.path.join(path_txt_save,txt_name))


    #---------------------------------------------------
    # 替换时 除了其他位置不要与替换位置内容相同！！！！S
    # 全局变量和局部变量不要一样 易导致路径叠加  ！！！！
    if os.path.exists(os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','JPG')):
        img_path  = os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','JPG')  # 结合图片尺寸分组进行分割
    else:
        img_path  = os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','jpg')

    # img_path  = os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','JPG')  # 结合图片尺寸分组进行分割


    img = Image.open(img_path)
    size_img = img.size  # type : tuple  # （左上角坐标(x,y)，右下角坐标（x+w，y+h）

    # print('size_img :' , size_img)

    w = size_img[0]   # 和列表一样，也是一个有序的存储数据的容器；可以通过下标来获取元素
    h = size_img[1]

    '''
    #------根据不同size进行分割----------------------------
    if w == 3472 or w == 4624 :
        x_num = 4
        y_num = 4
        total_num_crop = total_num_crop +  y_num * x_num
    if w == 9248 or w == 6944 :
        x_num = 8
        y_num = 8
        total_num_crop = total_num_crop +  y_num * x_num
    if w == 3264 or w == 2448 :
        x_num = 3
        y_num = 3
        total_num_crop = total_num_crop +  y_num * x_num
    '''


    img_size = tuple(cv2.imread(img_path).shape[:2])


    # if img_size in split_num:

    # or any other default value

    if img_size in split_num:


        num = split_num[img_size]

        x_num = y_num = num

        x = 0
        y = 0


        w = int(size_img[0] / x_num)
        h = int(size_img[1] / y_num)

        for k in range(x_num):  # 注意这里是从上到下，再从左到右裁剪的
            for v in range(y_num):


                #-------------------保存图片的位置以及图片名称----------------------------------------------------------
                region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
                region.save(os.path.join(save_path_img,txt_name) + '/' +f'{txt_name}_'+'%d%d' % (v, k) + '.jpg')
                #-------------------------------------------------------------------------------------------#


                # ---------------------------保存txt 的位置以及txt名称----------------------------------------
                with  open (os.path.join(path_txt_save,txt_name) + '\\'+f'{txt_name}_'+'%d%d' % (v, k) + '.txt','w') as f:

                    with open(os.path.join(path_txt,txt)) as t_f:
                        lines = t_f.readlines()   #每次读取一行内容

                        for line in lines:

                            # print( 'line: ', line)
                            # print('line.strip(): ',line.strip())  # txt文件 出现空白行 末尾有换行字符

                            c_x = line.strip().split(' ')[0]
                            c_y = line.strip().split(' ')[1]

                            if float(c_x) > w * k and float(c_x) <= w * ( k + 1 ) and float(c_y) > h * v and float(c_y) <= h * ( v + 1 ) :

                                c_x = float(c_x) - w * k
                                c_y = float(c_y) -  h * v
                                f.write('{} {}'.format(c_x,c_y))
                                # f.write(f'{c_x}')
                                # f.write(' ')
                                # f.write(f'{c_y}')
                                f.write('\n')

    else:
        print('===========该尺寸没有预先定义======================')

#==================================ff==================================================
 #-----------------input-------------------------------------------------
path_txt = r'data\raw_test\fruit_fly\annotations'
image_path = r'data\raw_test\fruit_fly\images'
path_txt_save = r'data\raw_test\fruit_fly\tiles\annotations'
save_path_img = r"data\raw_test\fruit_fly\tiles\images"
#----------------------------------------------------------------------------



txt_list = os.listdir(path_txt)
total_num_crop = 0

# 定义分割后的块数
split_num = {
    (3264, 2448): 3,
    (2448,3264 ): 3,
    (3468, 4624): 4,
    (4624,3468 ): 4,
    (6936, 9248): 8,
    (9248,6936 ): 8,
    (3472, 4624): 4,
    (4624, 3472): 4,
    (6944, 9248): 8,
    (9248, 6944): 8,
}



for txt in txt_list:

    txt_name =txt.split('.')[0]




    # 每一张图片txt分割单独保存
    if not os.path.exists (os.path.join(save_path_img,txt_name)):
        os.mkdir(os.path.join(save_path_img,txt_name))

    if not os.path.exists (os.path.join(path_txt_save,txt_name)):
        os.mkdir(os.path.join(path_txt_save,txt_name))


    #---------------------------------------------------
    # 替换时 除了其他位置不要与替换位置内容相同！！！！S
    # 全局变量和局部变量不要一样 易导致路径叠加  ！！！！
    if os.path.exists(os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','JPG')):
        img_path  = os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','JPG')  # 结合图片尺寸分组进行分割
    else:
        img_path  = os.path.join(path_txt,txt).replace(path_txt ,image_path).replace('txt','jpg')


    img = Image.open(img_path)
    size_img = img.size  # type : tuple  # （左上角坐标(x,y)，右下角坐标（x+w，y+h）

    # print('size_img :' , size_img)

    w = size_img[0]   # 和列表一样，也是一个有序的存储数据的容器；可以通过下标来获取元素
    h = size_img[1]

    '''
    #------根据不同size进行分割----------------------------
    if w == 3472 or w == 4624 :
        x_num = 4
        y_num = 4
        total_num_crop = total_num_crop +  y_num * x_num
    if w == 9248 or w == 6944 :
        x_num = 8
        y_num = 8
        total_num_crop = total_num_crop +  y_num * x_num
    if w == 3264 or w == 2448 :
        x_num = 3
        y_num = 3
        total_num_crop = total_num_crop +  y_num * x_num
    '''


    img_size = tuple(cv2.imread(img_path).shape[:2])


    # if img_size in split_num:

    # or any other default value

    if img_size in split_num:


        num = split_num[img_size]

        x_num = y_num = num

        x = 0
        y = 0


        w = int(size_img[0] / x_num)
        h = int(size_img[1] / y_num)

        for k in range(x_num):  # 注意这里是从上到下，再从左到右裁剪的
            for v in range(y_num):


                #-------------------保存图片的位置以及图片名称----------------------------------------------------------
                region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
                region.save(os.path.join(save_path_img,txt_name) + '/' +f'{txt_name}_'+'%d%d' % (v, k) + '.jpg')
                #-------------------------------------------------------------------------------------------#


                # ---------------------------保存txt 的位置以及txt名称----------------------------------------
                with  open (os.path.join(path_txt_save,txt_name) + '\\'+f'{txt_name}_'+'%d%d' % (v, k) + '.txt','w') as f:

                    with open(os.path.join(path_txt,txt)) as t_f:
                        lines = t_f.readlines()   #每次读取一行内容

                        for line in lines:

                            # print( 'line: ', line)
                            # print('line.strip(): ',line.strip())  # txt文件 出现空白行 末尾有换行字符

                            c_x = line.strip().split(' ')[0]
                            c_y = line.strip().split(' ')[1]

                            if float(c_x) > w * k and float(c_x) <= w * ( k + 1 ) and float(c_y) > h * v and float(c_y) <= h * ( v + 1 ) :

                                c_x = float(c_x) - w * k
                                c_y = float(c_y) -  h * v
                                f.write('{} {}'.format(c_x,c_y))
                                # f.write(f'{c_x}')
                                # f.write(' ')
                                # f.write(f'{c_y}')
                                f.write('\n')

    else:
        print('===========该尺寸没有预先定义======================')


# print ('一共处理了' , total_num_crop, '张')   # 和下一步重叠了


print("处理完毕")
