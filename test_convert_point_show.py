import argparse
import torch
import os
import numpy as np
import datasets.crowd_test as crowd
# from models import vgg19   # 修改
# from models import vgg19
from models.models_aspp_cmba_conv_ff  import  vgg19
from PIL import Image
# from scipy.ndimage.measurements import center_of_mass

'''
scipy.ndimage.measurements.center_of_mass 是一个用于计算 N 维数组的质心的函数，
它返回的是一个包含每个维度的质心坐标的元组。对于二维数组，返回的元组形状为 (y, x)，表示纵坐标和横坐标。

'''
#-----------------------------------------------------------------------
# add
import logging    #---------1
# ok 定比例拼接

# 配置日志记录器
# logging.basicConfig(filename='result-log.txt', level=logging.INFO)     #-------2
#------------------------------------------------------------------------------------------
def center_of_mass(array):
    # 计算每个维度的质心坐标
    coords = np.indices(array.shape)
    total_mass = np.sum(array)
    y_center = np.sum(coords[0] * array) / total_mass
    x_center = np.sum(coords[1] * array) / total_mass
    # 返回质心坐标元组
    return (y_center, x_center)





parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,   # 可以测试 哪个尺度最好
                    help='the crop size of the train image')
# 修改
parser.add_argument('--model-path', type=str, 
                    default=r'ckpts\input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0-v3 fusion\best_model_16.pth',  # 可在文件夹里找到预训练文件 /  改为自己训练好的
                    help='saved model path')
# 秀发该
parser.add_argument('--data-path', type=str,
                    default=r'test_images-pre-result-full\data-used-by-train-val-test',    # 数据集可以替换为自己的
                    help='saved model path')

parser.add_argument('--dataset', type=str, default='qnrf',   # 仿照他的格式来就行
                    help='dataset name: qnrf, nwpu, sha, shb')
# 修改                   
parser.add_argument('--pred-density-map-path', type=str, default=r'test_images-pre-result-full\result_point',    # 预测密度图保存路径 
                    help='save predicted density maps when pred-density-map-path is not empty.')

parser.add_argument('--if-val', type=bool, default= True,    # 预测密度图保存路径 
                    help='save predicted density maps when pred-density-map-path is not empty.')
parser.add_argument('--wf-txt', type=str, default= r'test_images-pre-result-full\map_out\detection_wf',    # 预测密度图保存路径 
                    help='save predicted density maps when pred-density-map-path is not empty.')
parser.add_argument('--ff-txt', type=str, default= r'test_images-pre-result-full\map_out\detection_ff',    # 预测密度图保存路径 
                    help='save predicted density maps when pred-density-map-path is not empty.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path

per_img_files = []

# 测试 两种结果的测试显示 



# 遍历文件夹中的所有子文件夹 包括子文件夹的子文件夹
# 如果只需要遍历文件夹的子文件夹 os.listdir即可



for  subdirname in os.listdir(data_path): 

    LEN_NUM = []

    for i in os.listdir(os.path.join(data_path, '{}'.format(subdirname))):
        if i.endswith('.jpg') or  i.endswith('.JPG'):
            LEN_NUM.append(i)

    full_img_list = []  # 用来存放每个文件夹处理完的分割的图片方便按照规则拼接到一起
    full_img_path_list = [] # 用来存放每个文件夹处理完的分割的图片路径名字方便按照规则拼接到一起
    
    real_count = 0
    pre_count = 0

    real_count1 = 0
    pre_count1 = 0

    # print('subdirname',subdirname)


    if args.dataset.lower() == 'qnrf':   # 在'data/QNRF-Train-Val-Test' 数据集中取出 测试用到的数据
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, '{}'.format(subdirname)), crop_size, 8, method='val')

    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,num_workers=0, pin_memory=True)


    if args.pred_density_map_path:  # save predicted density maps when pred-density-map-path is not empty
        import cv2
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)

    model = vgg19()
    model.to(device)

    model.load_state_dict(torch.load(model_path, device))  # 直接使用预训练好的权重


    model.eval()
    image_errs = []
    image_errs1 = []

    i = 0
    # for inputs, count,name in dataloader:
    for inputs, count,count1, name in dataloader:


        print('name : ',f"{str(name[0])}".split('.')[0])

        f = open(os.path.join(args.wf_txt,f"{str(name[0])}").split('.')[0]+'.txt','w')

        f1 = open(os.path.join(args.ff_txt,f"{str(name[0])}").split('.')[0]+'.txt','w')
        # print('name',name)

        #-------------------------------读取原图--------------------------------------
        # print(os.path.join(data_path, '{}'.format(str(subdirname)),'{}.jpg'.format(str(name[0]))))
        img = cv2.imread(os.path.join(data_path, '{}'.format(str(subdirname)),'{}.jpg'.format(str(name[0]))))
        #----------------------------------------------------------------------

        ori_w,ori_h ,_= img.shape# 获取宽高


        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'

        with torch.set_grad_enabled(False):  # 关闭自动求导模式
            outputs, Probability_map,outputs1, Probability_map1 = model(inputs)   # return mu, mu_normed  # 返回密度图 和 归一化操作的概率图


        img_err = count[0].item() - torch.sum(outputs).item()  # 计算预测的密度与真实密度之间的误差。
        img_err1 = count1[0].item() - torch.sum(outputs1).item()



        # 结果计数
        real_count +=  count[0].item()
        real_count1 +=  count1[0].item()

        pre_count += torch.sum(outputs).item()
        pre_count1 += torch.sum(outputs1).item()

        image_errs.append(img_err)
        image_errs1.append(img_err1)

        if args.pred_density_map_path:

            vis_img = outputs[0, 0].cpu().numpy()  # 提取第一个图像的密度图，并将其转换为 NumPy 数组。
            vis_img1 = outputs1[0, 0].cpu().numpy()

            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)

            vis_img1 = (vis_img1 - vis_img1.min()) / (vis_img1.max() - vis_img1.min() + 1e-5)
            vis_img1 = (vis_img1 * 255).astype(np.uint8)

            #------------------------add-------------------------------------------------
            # 查看输出的图
            # 查看不为0的像素个数
            # print('vis_img', vis_img )  # , 'vis_img.size', vis_img.size()
            nonzero_count = np.sum(vis_img != 0)
            nonzero_count1 = np.sum(vis_img1 != 0)
            # print("不为0的像素个数为 ：",nonzero_count)
            vis_img_copy = vis_img.copy()
            vis_img_copy1 = vis_img1.copy()
            #-------------------------------------------------------------------------------


            #--------------------------------------resize 到 同一 尺寸---------------------------------------------------
            w , h  = vis_img.shape[:2]  # ：获取热力图的宽和高
            # print(w , h )   # 240 180    # 下采样8倍
            # print(ori_w , ori_h ) # 1920 1440
            size = (int(h),int(w))   # 将热力图的宽和高赋值给变量 size
            img = cv2.resize(img,size)  # 将原始图像 img 调整为与热力图大小相同



            wide_ration =  ori_w / w
            hight_ration =  ori_h/ h 



            #--------------------------------add---------------------------------------------------------
            # 目的:   是使其输出的密度图以固定的颜色显示到原图上
            
            # special_vis_img = img  # ！！！！ 直接复制 相同的内存地址 会改变原来的数据
            special_vis_img = img.copy()  # 拷贝原始图像数据
            
            # 生成一个与img大小相同的全零二维数组 
            mask = np.zeros_like(special_vis_img[:, :, 0])
            mask1 = np.zeros_like(special_vis_img[:, :, 0])

            # 用于存放点坐标
            mask2 = np.zeros_like(special_vis_img[:, :, 0])
            mask3 = np.zeros_like(special_vis_img[:, :, 0])


            # 将arr1中不为0的像素位置在mask中标记为1
            # mask[vis_img_copy != 0] = 1 
            # mask1[vis_img_copy1 != 0] = 1 
            mask[vis_img_copy >= 10] = 1 
            mask1[vis_img_copy1 >= 10] = 1 

            # cv2.imwrite(r'mask.jpg',mask1)

            #==============================-转换为点表示=============================================================

            radius = 2  # 决定消除冗余点的半径大小
            points = cv2.dilate(mask, np.ones((radius, radius)))
            points = cv2.erode(points, np.ones((radius, radius)))
            # points, _ = cv2.connectedComponents(points.astype(np.uint8))  
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(points.astype(np.uint8), connectivity=4)
            # print('retval',retval)   
            # print('labels',labels)    
            # print('stats',stats)     # [[    0     0   180   240 43200]]
            # print('centroids',centroids)           
            '''
                cv2.connectedComponents函数需要一个二进制图像作为输入，其中像素值为非零表示前景（即要标记的对象），像素值为零表示背景。
                函数会将前景像素组成的连通块（connected component）标记为不同的整数值（从1开始），并将背景像素标记为0。
                因此，返回的标记后的图像中，每个像素都表示对应像素所属的连通块的整数标记值。

                返回的标记后的图像赋值给points变量，而下划线(_)则表示忽略了返回的第二个数组。
            '''
            #  python test_convert_point_show.py 
            if len(centroids) !=0:
                for centroid in centroids :
                    
                    mask2[int(centroid[1]), int(centroid[0])] = 1 


                    x1 = int(centroid[0]) * wide_ration - 16
                    # if x1 < 0 :
                        # x1 = 0


                    y1 = int(centroid[1]) * hight_ration - 16
                    # if y1 < 0:

                        # y1 = 0


                    x2 = int(centroid[0]) * wide_ration + 16
                    # if x2 > w :
                       #  x2 = w
                  
                    y2 = int(centroid[1]) * hight_ration + 16
                    # if y2 > h:
                        # y2 =h
                    # cv2.rectangle(special_vis_img, (x1, y1), (x2, y2), (255,0, 0), 1) 




                    #===============================放到txt===========================================

                   
                    f.write('1 1')  
                    f.write(' ') 
                    # f.write(f'{x1*wide_ration} {y1*hight_ration} {x2*wide_ration} {y2*hight_ration}') 
                    f.write(f'{x1} {y1} {x2} {y2}') 
                    f.write('\n') 


                    cv2.rectangle(special_vis_img, (int(centroid[0])-4, int(centroid[1])-4), (int(centroid[0])+4, int(centroid[1])+4), (0, 0, 255), 1)   
            f.close()
            #=======================================================  



            radius1 = 2  # 决定消除冗余点的半径大小
            points1 = cv2.dilate(mask1, np.ones((radius1, radius1)))
            points1 = cv2.erode(points1, np.ones((radius1, radius1)))
            # points1, _ = cv2.connectedComponents(points1.astype(np.uint8))    
            retval, labels, stats, centroids1 = cv2.connectedComponentsWithStats(points1.astype(np.uint8), connectivity=4) # centroids [[ 89.5 119.5]]
            # centroids : 返回的是连通区域的质心。
            # stats ：stats会包含5个参数分别为x,y,h,w,s。
            # 分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；
            # s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
            # labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
            # retval : 返回值是连通区域的数量。


            if len(centroids1) !=0:
                for centroid in centroids1 :
                    mask3[int(centroid[1]), int(centroid[0])] = 1  
                    x1 = int(centroid[0]) * wide_ration - 32

                    y1 = int(centroid[1]) * hight_ration - 32



                    x2 = int(centroid[0]) * wide_ration + 32
                
                    y2 = int(centroid[1]) * hight_ration + 32

                    # cv2.rectangle(special_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1) 

                    f1.write('2 1')  
                    f1.write(' ') 
                    # f1.write(f'{x1*wide_ration} {y1*hight_ration} {x2*wide_ration} {y2*hight_ration}')
                    f1.write(f'{x1} {y1} {x2} {y2}')
                    f1.write('\n')          
                    cv2.rectangle(special_vis_img, (int(centroid[0])-6, int(centroid[1])-6), (int(centroid[0])+6, int(centroid[1])+6), (255, 255, 0), 1)  
            f1.close()
        #===================================================================


            # 将mask中的1对应的像素在img中设为蓝色 
            special_vis_img[np.where(mask2 == 1)] = [255, 0, 0]
            # print(len(special_vis_img[np.where(mask2 == 1)]))
            special_vis_img[np.where(mask3 == 1)] = [0,255, 0]  



            full_img_list.append(special_vis_img) # 存放已经处理完的图片
            

            full_img_path_list.append(os.path.join(data_path, '{}'.format(str(subdirname)),'{}.jpg'.format(str(name[0]))))

            if len(full_img_list) == len(LEN_NUM):
                # print('ok')

                full_image = 0

                row_list = []
                col_list = []

                for i, item in enumerate(full_img_list):
                    row_rol = full_img_path_list[i].split('_')[-1].split('.')[0]
                    row = int(row_rol[0])
                    col = int(row_rol[1])
                    row_list.append(row)
                    col_list.append(col)

                row_num = max(row_list) + 1
                col_num = max(col_list) + 1

                full_img_list_array = [[0 for x in range(col_num)] for y in range(row_num)]

                for i, item in enumerate(full_img_list):
                    row_rol = full_img_path_list[i].split('_')[-1].split('.')[0]
                    row = int(row_rol[0])
                    col = int(row_rol[1])
                    full_img_list_array[row][col] = item

                # print(full_img_list_array)

                row_images = []
                for i in range(row_num):
                    col_images = []
                    for j in range(col_num):
                        col_images.append(full_img_list_array[i][j])
                    row_images.append(np.concatenate(col_images, axis=1))
                
                full_image = np.concatenate(row_images, axis=0)

                #---------------------------------------------------------------------------
                # 不能画矩形框
                '''
                width,height = full_image.shape[:2]
                # 定义矩形框的位置和大小
                x, y, w, h = int(width*0.8), int(height*0.95), int(width*0.2), int(height*0.05)
                # 绘制矩形框
                cv2.rectangle(full_image, (x, y), (x+w, y+h), (255, 0, 0), -1)
                # 在矩形框内添加文本
                text = 'predict ： {}'.format(pre_count)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x, text_y = int(x+w/2-text_size[0]/2), int(y+h/2+text_size[1]/2)
                cv2.putText(full_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                '''

                cv2.imwrite(os.path.join(args.pred_density_map_path, str(subdirname) + '_result_full.jpg'),full_image)
                
                print('已经保存至',os.path.join(args.pred_density_map_path, str(subdirname) + '_result_full.jpg'))


                with open (os.path.join(args.pred_density_map_path,'wf_result_logger_ours'+'.txt'),'a') as result_record:

                    result_record.write('{} {} {}'.format(subdirname,real_count,pre_count))
                    result_record.write('\n')

                with open (os.path.join(args.pred_density_map_path,'ff-result_logger_ours'+'.txt'),'a') as result_record:

                    result_record.write('{} {} {}'.format(subdirname,real_count1,pre_count1))
                    result_record.write('\n')

            #---------------------------------------------------------------------------------------------------------------------------
            # 将二维数组进行扩展 使其shape与三维数组形同
            vis_img_expanded = np.expand_dims(vis_img,axis=2)

            superimposed_img2 = vis_img_expanded * 0.9 + img # 将两张图像进行叠加，其中vis_img是密度图，img是原始图像。将vis_img和原始图像img相乘并加上0.9，这样密度图的颜色更加鲜明
            imgs = np.hstack([ img,superimposed_img2])
            '''
            两个图像数组（即 `img` 和 `superimposed_img2`）水平拼接在一起，并将结果赋值给变量 `imgs`
            方便对比
            '''


            

    

    image_errs = np.array(image_errs)
    image_errs1 = np.array(image_errs1)

    wfmse = np.sqrt(np.mean(np.square(image_errs)))
    wfmae = np.mean(np.abs(image_errs))

    ffmse = np.sqrt(np.mean(np.square(image_errs1)))
    ffmae = np.mean(np.abs(image_errs1))
    print('{}'.format(str(subdirname)),'wf-pre_count is : {}'.format(pre_count),'wf-real_count is :{}'.format(real_count))
    print('{}'.format(str(subdirname)),'ff-pre_count is : {}'.format(pre_count1),'ff-real_count is :{}'.format(real_count1))
    print('{}: wf mae {}, wf mse {}\n'.format(model_path, wfmae, wfmse))
    print('{}: ff mae {}, ff mse {}\n'.format(model_path, ffmae, ffmse))


print('=====预测已完成======')