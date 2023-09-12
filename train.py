import argparse
import os
import torch
from train_helper import Trainer


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default=r'data\data-used-by-train-val-test', help='data path')     # 数据存放文件夹
    # add
    parser.add_argument('--data-dir_1', default=r'data\data-used-by-train-val-test-another', help='data path')     # 数据存放文件夹

    parser.add_argument('--dataset', default='qnrf', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    
    # 要求使用反斜杠符
    pretrained_models_dict = {

        "1": 'pretrained_models/new_ok_model_qnrf.pth',  # MA-DM + Data_equalization

    }


    parser.add_argument('--resume', default=pretrained_models_dict["1"], type=str,  # pretrained_models\model_qnrf.pth
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=200,   # 最大epoch
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')   # 每训练 n 个epoch 开始验记录
    parser.add_argument('--val-start', type=int, default=5,  
                        help='the epoch start to val')     # 50轮之后开始验证
    parser.add_argument('--batch-size', type=int, default=10,   # 可以修改
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=0,  # Caught IndexError in DataLoader worker process 0.
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=512,    # 图片尺寸
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')   
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')  # 计算距离是否归一化
    parser.add_argument('--num-time', type=int, default=1, help='ff loss')  # 计算距离是否归一化
    parser.add_argument('--tensorboard_dir', default='./runs', help='path where to save, empty for no saving')   
    parser.add_argument('--output_dir', default='./log',help='path where to save, empty for no saving') 
    args = parser.parse_args()


    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 512
        # args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()  # Set parameters and Prepare data set
    trainer.train()  # start train
