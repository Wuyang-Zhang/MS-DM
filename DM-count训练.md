## 数据生成



### 输入：

将 图片 ---datasets\imgs和 txt 标注文件--- datasets\imgs和annotation 

### 操作



首先要确保图片被分割成868 * 1156 和上海数据集相同的尺寸



首先判断所有图片的尺寸，方便进行分割

run 

F:\002Patrol_inspection_topic\data_processing_demo\Folder-Images-SizeType\folder-images-sizetype-v1.py

获得以下种类

======种类有========
0 : (3264, 2448)
1 : (3468, 4624)
2 : (6936, 9248)
3 : (3472, 4624)
4 : (4624, 3472)
5 : (6944, 9248)
6 : (9248, 6944)



1. 训练时 去除没有目标的图片和对应的txt信息！！！

run 

F:\002Patrol_inspection_topic\data_processing_demo\Remove-Empty-Target-Image-based-on-point-information\Remove-empty-target-image-based-on-point-information-v1.py



空文件直接删除即可

此步骤需查看路径的准确性



2. 根据 data\generate_mat.py生成mat文件 到data文件夹里的mats  folder -----data\mats



3. 建立data\images  同---datasets\imgs



4. 然后在data\data-used-by-train-val-test folder 里 分别建立 train test val folder



5. run data\split-train-val-test-name-to-txt.py 生成 对应的  train test val.txt

注意：

该名称为绝对路径 如果该根目录移动的话是需要换的



6. run  preprocess\preprocess_dataset_nwpu.py 

```python
i = i.strip().split(',')[0] 
```

 因为有的图片路径中包含空格 所以这里以空格分割改为逗号分割

运行之后data\data-used-by-train-val-test\test 没有.npy   ._densitymap.npy



没关系 可以以后再生成  训练只用到 train  val文件中的东西

## train

**!!!!! 使用环境pytorch**



## 开始训练

run  train.py



## 报错



warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

**错误原因：**

python版本问题，python3.5 支持 upsample 函数，python3.6 不支持 upsample 函数

***假如我们忽略这个警告，会导致实验效果降低，简单来说，这个警告一定要改\***



**解决办法**：

在model.py里面

```python
from torch.nn import functional as F


        # x = F.upsample_bilinear(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
```



## test

run   preprocess\preprocess_dataset_nwpu-test.py  得到test文件的  .npy  _density_map.npy文件

这个因为**不用** 单独测试 所以不需要运行



修改修炼好的权重路径

```python
parser.add_argument('--model-path', type=str, default=r'',   help='saved model path')
```



修改包含test文件的父文件夹

```python
parser.add_argument('--data-path', type=str,
                    default=r'data\data-used-by-train-val-test',   
                    help='saved model path')
```



密度图结果保存位置

```python
parser.add_argument('--pred-density-map-path', type=str, default=r'predicted_result', help='save predicted density maps when pred-density-map-path is not empty.')

```



run  test.py 文件  

如果想生成test文件里的真值文件 

运行 preprocess\preprocess_dataset_nwpu-test.py 即可 不需修改路径 但一般情况下不生成test文件了，重新测试为主  把data\split-train-val-test-name-to-txt.py 比例改变即可 



## 多张图训练测试拼接

重新测试文件在 test_images-pre-result-full folder里



**直接放进去即可**  ，不需要任何改动 最后可直接在result文件夹里查看效果图

**输入：**

​	est_images   : 存放用于测试的图片
​	json : 存放对应的json文件



1. 将坐标点转换成txt文件-----------------》test_images-pre-result-full\point2txt

   run  test_images-pre-result-full\json2point-v1.py 



2. 分割坐标和图片  

     ------------------------------test_images-pre-result-full\annotation

   -------------------------------test_images-pre-result-full\images
   
   run  test_images-pre-result-full\Splitting_images _and_coordinates.py



3. 将测试的图片姓名写入txt文件 方便生成真值图----------test_images-pre-result-full\test_txt

   run test_images-pre-result-full\split-train-val-test-name-to-txt.py



4. 升mat 文件 生成真值图 -------------test_images-pre-result-full\mats

   run test_images-pre-result-full\generate_mat.py



5. 生成真值图  

   run test_images-pre-result-full\preprocess_dataset_nwpu-test.py



6. 进行结果测试   ----------------》  test_images-pre-result-full\result

   run test-v1.py

## 相关性分析  使用often 环境

相关性分析.py

得到拟合图和相关系数r

相关系数(r)和截距(b)是线性回归的两个重要输出结果。

相关系数(r)是衡量两个变量之间线性关系强度的指标，取值范围在-1到1之间。当r为正时，说明两个变量呈正相关，r的值越接近1，正相关性越强；当r为负时，说明两个变量呈负相关，r的值越接近-1，负相关性越强；当r为0时，说明两个变量没有线性关系。

截距(b)是回归模型的截距项，是回归直线与y轴交点的值。在一元线性回归中，它代表当自变量取值为0时，因变量的预测值。

斜率代表的是拟合曲线的斜率，也称为回归系数或权重。在这里，斜率代表的是x和y之间的线性关系的强度，具体来说，它代表着y的单位变化量（即y轴上的变化量）对x的单位变化量（即x轴上的变化量）的影响程度。斜率越大，说明x和y之间的线性关系越强，即y对x的影响越大。

## 可视化

使用netron即可

### CBAMB

#### 修改 model文件 加入CMBA 模块 并权重初始化 应该是model.py  或models-v2.py

**error:** UserWarning: nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")

```python
# x = F.upsample_bilinear(x, scale_factor=2)
x = F.interpolate(x, scale_factor=2)
```



#### run   D:\download\DM-Count-master\DM-Count-master\Modify-the-weights-save-the-new.py

根据已经训练好的 权重  生成新的权重文件   **"pretrained_models\new_ok_model_qnrf.pth"**

#### 然后使用冻结训练 微调的方法  在train_helper.py or train_helper--v2.py

```python
    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch

            #======================add=======================================
            if epoch <= 500:
                # Freeze layers except CBAM module after 500 epochs
                for name, param in self.model.named_parameters():
                    if "cbam" not in name:
                        param.requires_grad = False
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            else:
                # Unfreeze all layers after 500 epochs
                for name, param in self.model.named_parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            #====================================================================
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
```

####  修改train.py 文件的 

```python
parser.add_argument('--resume', default=r'D:\download\DM-Count-master\DM-Count-master\pretrained_models\model_qnrf.pth', type=str,
                        help='the path of resume training model')
```

## **目录结构详解**

datasets\crowd_test.py   用来test文件 生成数据集



##**相应的修改**##**

### 改了相应的预测回归层  只有一个密度图 但是两个通道  最后分

models_2_desity.py
pretrained_models\new_2_density_model_qnrf.pth
train_helper_2_density.py
train_2_density.py


最后只运行  train_2_density.py 即可

#-----------------------------------------------------------------------------------

### 原来从reg_layer那就开始改的的

models.py
train_helper.py
pretrained_models\new_ok_model_qnrf.pth
train.py
最后只运行  train.py 即可

## 结构目录一览

![image-20230913094521452](DM-count%E8%AE%AD%E7%BB%83.assets/image-20230913094521452.png)

![image-20230913094540221](DM-count%E8%AE%AD%E7%BB%83.assets/image-20230913094540221.png)