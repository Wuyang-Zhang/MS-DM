from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
    func: generate the discrete map.
    points: [num_gt, 2], for each row: [width, height]
    """
    # 创建一个形状为 (im_height, im_width) 的全零的离散地图
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]  # 获取地图的高度和宽度
    num_gt = points.shape[0]  # 获取点的数量

    if num_gt == 0:
        return discrete_map  # 如果没有点，则返回全零的地图

    # 快速创建离散地图的方法
    points_np = np.array(points).round().astype(int)  # 将点坐标四舍五入并转换为整数

    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))  # 计算每个点的高度坐标
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))  # 计算每个点的宽度坐标

    # 使用 scatter_add_ 函数将点添加到离散地图中
    p_index = torch.from_numpy(p_h * im_width + p_w)  # 计算点在一维数组中的索引
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index.long(), src=torch.ones(im_width * im_height)).view(im_height, im_width).numpy()

    ''' slow method
    # 慢速方法（注释掉的部分）：逐个点将其加入到离散地图中
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''

    assert np.sum(discrete_map) == num_gt  # 断言：离散地图中的值总和应等于点的数量

    return discrete_map



class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints, keypoints1):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)

        # 必须大于512才能裁剪
        # self.c_size = 512
 
        assert st_size >= self.c_size 
        
        assert len(keypoints) >= 0
        assert len(keypoints1 )>=0
        # if st_size >= self.c_size :

        '''
        i：通常表示图像的垂直方向（行）的起始位置（或裁剪窗口的左上角 y 坐标）。
        j：通常表示图像的水平方向（列）的起始位置（或裁剪窗口的左上角 x 坐标）。
        h：通常表示裁剪窗口的高度（垂直方向上的大小）。
        w：通常表示裁剪窗口的宽度（水平方向上的大小）。

        这些变量的具体值是通过调用 random_crop 函数来获取的，它可能是一个用于随机裁剪图像的函数。
        通常情况下，random_crop 函数会随机生成合适的 i、j、h 和 w 值，以便对输入图像进行裁剪，以满足特定的需求或数据增强操作。
        '''
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)

        
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                    (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        if len(keypoints1) > 0:
            keypoints1 = keypoints1 - [j, i]
            idx_mask1 = (keypoints1[:, 0] >= 0) * (keypoints1[:, 0] <= w) * \
                    (keypoints1[:, 1] >= 0) * (keypoints1[:, 1] <= h)
            keypoints1 = keypoints1[idx_mask1]
        else:
            keypoints1 = np.empty([0, 2]) 

        # else:

        #     img=img
        #     keypoints = keypoints
        #     keypoints1 = keypoints1
        #     w,h = wd,ht



        gt_discrete = gen_discrete_map(h, w, keypoints)
        #  add
        gt_discrete1 = gen_discrete_map(h, w, keypoints1)

        down_w = w // self.d_ratio
        down_h = h // self.d_ratio

        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        # add
        gt_discrete1 = gt_discrete1.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))

        assert np.sum(gt_discrete) == len(keypoints)
        # add
        assert np.sum(gt_discrete1) == len(keypoints1)


        # 进行图片翻转

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        # add
        if len(keypoints1) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete1 = np.fliplr(gt_discrete1)
                keypoints1[:, 0] = w - keypoints1[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete1 = np.fliplr(gt_discrete1)
        gt_discrete1 = np.expand_dims(gt_discrete1, 0)


        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(),torch.from_numpy(keypoints1.copy()).float(), torch.from_numpy(
            gt_discrete1.copy()).float()


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method

        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))

        print('该数据集 训练集 数量 number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]

        # print('读取图片路径为，', img_path)

        gd_path = img_path.replace('jpg', 'npy')


        img = Image.open(img_path).convert('RGB')

        if self.method == 'train':


            if os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and not os.path.exists(gd_path):

                keypoints1 = np.load(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy'))
                # keypoints = np.zeros((img.size[0],img.size[1]),dtype= np.float32)
                keypoints = np.zeros((0,2))
                # 增加第二个关键点返回值
                return self.train_transform(img, keypoints, keypoints1)  
            
            elif not os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and os.path.exists(gd_path): 

                keypoints = np.load(gd_path)
                # keypoints1 =  np.zeros((img.size[0],img.size[1]),dtype= np.float32)
                keypoints1 = np.zeros((0,2))
                return self.train_transform(img, keypoints, keypoints1) 
            
            elif os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and os.path.exists(gd_path):  
                keypoints = np.load(gd_path)
                keypoints1 = np.load(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy'))
                return self.train_transform(img, keypoints, keypoints1) 
            
            else:
                # keypoints1 =  np.zeros((img.size[0],img.size[1]),dtype= np.float32)
                # keypoints = np.zeros((img.size[0],img.size[1]),dtype= np.float32)
                keypoints = np.zeros((0,2))
                keypoints1 = np.zeros((0,2))
                return self.train_transform(img, keypoints, keypoints1) 
        
        elif self.method == 'val': 

            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]

            if os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and not os.path.exists(gd_path):
                keypoints1 = np.load(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy'))
                # print('wf:' ,0,'ff',len(keypoints1))
                return img, 0, len(keypoints1),name  
            
            elif not os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and os.path.exists(gd_path):  
                keypoints = np.load(gd_path)
                # print('wf:' ,len(keypoints),'ff',0)
                return img,len(keypoints), 0,name 
            
            elif os.path.exists(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy')) and os.path.exists(gd_path):  
                keypoints = np.load(gd_path)
                keypoints1 = np.load(img_path.replace('data-used-by-train-val-test','data-used-by-train-val-test-another').replace('jpg', 'npy'))
                # print('wf:' ,len(keypoints),'ff',len(keypoints1))
                return img,len(keypoints), len(keypoints1),name 
            
            else:
                return img,0, 0,name 




class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('该数据集 验证的数量 是 number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name


class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground-truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()
