import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime

from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from models import vgg19
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

def train_collate(batch):

    '''
    这段代码定义了一个自定义的collate_fn函数，用于在数据加载器中处理批量数据。
    在这个函数中，输入的batch参数是一个列表，其中每个元素都是数据集中的一个样本。每个样本由三个部分组成：图像数据、点数据和离散地图数据。
    为了将这些样本组合成一个批次，首先将它们转置，然后使用torch.stack()函数将所有图像数据、离散地图数据组合成张量。点数据是一个不固定长度的列表，因此保留它作为一个列表。
    最后，将组合后的图像数据、点数据和离散地图数据作为元组返回。
    请注意，此代码段只是一个示例，具体实现可能因您的数据集结构和需求而有所不同。您需要根据自己的数据集和训练需求，对该函数进行必要的调整和修改。
    
    '''
    transposed_batch = list(zip(*batch))

    # stack expects each tensor to be equal size, but got [3, 512, 512] at entry 0 and [3, 384, 1920] at entry 1
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    # add
    points1 = transposed_batch[3]

    gt_discretes = torch.stack(transposed_batch[2], 0)
    # add
    gt_discretes1 = torch.stack(transposed_batch[4], 0)
    return images, points, gt_discretes,points1,gt_discretes1


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # add
        self.combined_dataloader = {}

    def setup(self):
        args = self.args
        sub_dir = 'input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

        self.save_dir = os.path.join('ckpts', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')

        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        self.logger_wf = log_utils.get_logger(os.path.join(self.save_dir, 'wf-train-{:s}.log'.format(time_str)))
        self.logger_ff = log_utils.get_logger(os.path.join(self.save_dir, 'ff-train-{:s}.log'.format(time_str)))

        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8

        #------------------------------------------------------------------------------------------------
        if args.dataset.lower() == 'qnrf':

            print( '数据集1的路径是 ：' ,(os.path.join(args.data_dir, 'train')   ))
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
            print( '数据集1 的长度是 : ', len(self.datasets))

      
        #+--------------------------------------------------------------------------
                             
        else:
            raise NotImplementedError
        
        #--------------------------------------------------------------------
        

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          # shuffle=False,    # 不打乱      
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        


        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                               args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        run_log_name = os.path.join(args.output_dir, 'run_log.txt')
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            # self.train_eopch()
            # if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                # self.val_epoch()

            # def train_eopch(self):
            args = self.args
            epoch_ot_loss = AverageMeter()
            epoch_ot_obj_value = AverageMeter()
            epoch_wd = AverageMeter()
            epoch_count_loss = AverageMeter()
            epoch_tv_loss = AverageMeter()
            epoch_loss = AverageMeter()
            epoch_mae = AverageMeter()
            epoch_mse = AverageMeter()
            epoch_start = time.time()


            epoch_ot_loss_wf = AverageMeter()
            epoch_ot_obj_value_wf = AverageMeter()
            epoch_wd_wf = AverageMeter()
            epoch_count_loss_wf = AverageMeter()
            epoch_tv_loss_wf = AverageMeter()
            epoch_loss_wf = AverageMeter()
            epoch_mae_wf = AverageMeter()
            epoch_mse_wf = AverageMeter()
            epoch_start_wf = time.time()

            epoch_ot_loss_ff = AverageMeter()
            epoch_ot_obj_value_ff = AverageMeter()
            epoch_wd_ff = AverageMeter()
            epoch_count_loss_ff = AverageMeter()
            epoch_tv_loss_ff = AverageMeter()
            epoch_loss_ff = AverageMeter()
            epoch_mae_ff = AverageMeter()
            epoch_mse_ff = AverageMeter()
            epoch_start_ff = time.time()



            self.model.train()  # Set model to training mode

            #===================================================
            # the logger writer
            writer = SummaryWriter(args.tensorboard_dir)
            #===================================================

            for step, (inputs, points, gt_discrete,points1, gt_discrete1) in enumerate(self.dataloaders['train']):
                



                inputs = inputs.to(self.device)
                # inputs1 = inputs1.to(self.device)

                gd_count = np.array([len(p) for p in points], dtype=np.float32)  # 白色
                # print('1batch 真值1的个数',gd_count)
                # add
                gd_count1 = np.array([len(p) for p in points1], dtype=np.float32)
                # print('1batch 真值2的个数',gd_count1)
                

                points = [p.to(self.device) for p in points]
                # add
                points1 = [p.to(self.device) for p in points1]

                gt_discrete = gt_discrete.to(self.device) 
                # add
                gt_discrete1 = gt_discrete1.to(self.device)  

                N = inputs.size(0)  # Batch


                with torch.set_grad_enabled(True):

                    # 在此处修改  将第二个输出结果的特征图及其归一化结果增加
                    outputs, outputs_normed, outputs1, outputs_normed1 = self.model(inputs)
                    # print('outputs的尺寸() :',outputs.size())
                    # print('outputs1的尺寸 :',outputs1.size())


                    #------------------------------------Compute OT loss---------------------------------------------------------
                    ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points) 
                    # print('ot_的 loss,', ot_loss)
                    # add
                    ot_loss1, wd1, ot_obj_value1 = self.ot_loss(outputs_normed1, outputs1, points1)
                    # print('ot_的loss1,', ot_loss1)
                    
                    ot_loss = ot_loss * self.args.wot
                    # add
                    ot_loss1 = ot_loss1 * self.args.wot
                    # add 计算总的 ot_loss
                    ot_loss_all = ot_loss + ot_loss1

                    ot_obj_value = ot_obj_value * self.args.wot
                    # add
                    ot_obj_value1 = ot_obj_value1 * self.args.wot
                    # add 计算总的 ot_obj_value
                    ot_obj_value_all = ot_obj_value + ot_obj_value1

                    # add 计算总的 wd
                    wd_all = wd + wd1
                    
                    
                    epoch_ot_loss.update(ot_loss_all.item(), N)
                    epoch_ot_obj_value.update(ot_obj_value_all.item(), N)
                    epoch_wd.update(wd_all, N)

                    epoch_ot_loss_wf.update(ot_loss.item(), N)
                    epoch_ot_obj_value_wf.update(ot_obj_value.item(), N)
                    epoch_wd_wf.update(wd, N)

                    epoch_ot_loss_ff.update(ot_loss1.item(), N)
                    epoch_ot_obj_value_ff.update(ot_obj_value1.item(), N)
                    epoch_wd_ff.update(wd1, N)


                    #----------------------------------------Compute count_loss.----------------------------------------------------------
                    count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                        torch.from_numpy(gd_count).float().to(self.device))
                    # print('一种类的数数损失,', count_loss)
                    # add
                    count_loss1 = self.mae(outputs1.sum(1).sum(1).sum(1),
                                        torch.from_numpy(gd_count1).float().to(self.device))
                    # print('2种类的数数损失,', count_loss1)
                    # 计算总的count_loss
                    count_loss_all = count_loss + count_loss1
                    # print(f'总种类的数数损失 {N} ,', count_loss_all)

                    epoch_count_loss.update(count_loss_all.item(), N)

                    #----------------------------------------Compute TV loss.-------------------------------------------------------------------
                    gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(  
                        2).unsqueeze(3)
                    
                    # add
                    gd_count_tensor1 = torch.from_numpy(gd_count1).float().to(self.device).unsqueeze(1).unsqueeze( 
                        2).unsqueeze(3)
                    

                    gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                    # add
                    gt_discrete_normed1 = gt_discrete1 / (gd_count_tensor1 + 1e-6)


                    tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                        1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                    
                    # add
                    tv_loss1 = (self.tv_loss(outputs_normed1, gt_discrete_normed1).sum(1).sum(1).sum(
                        1) * torch.from_numpy(gd_count1).float().to(self.device)).mean(0) * self.args.wtv
                    
                    # add 计算总的tv_loss
                    tv_loss_all = tv_loss + tv_loss1

                    epoch_tv_loss.update(tv_loss_all.item(), N)


                    loss_wf = ot_loss + count_loss+ tv_loss
                    loss_ff = ot_loss1 + count_loss1+ tv_loss1


                    loss =  loss_wf + args.num_time * loss_ff # 修改
                    # print('args.num_time :', args.num_time)
                    # loss =  loss_wf +  loss_ff # 修改
                    # loss = ot_loss_all + count_loss_all + tv_loss_all
                    # print('总损失:', loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                    # print('种类一 预测 ',pred_count)
                    # add
                    pred_count1 = torch.sum(outputs1.view(N, -1), dim=1).detach().cpu().numpy()
                    # print('种类一 预测:',pred_count1)
                    pred_err = pred_count - gd_count
                    # print('总预测 :',pred_err)
                    # add
                    pred_err1 = pred_count1 - gd_count1
                    # print('pred_err1:',pred_err1)
                    # add 计算总的预测误差
                    pred_err_all = pred_err + pred_err1

                    # print('pred_err_all:',pred_err_all)

                    epoch_loss.update(loss.item(), N)

                    '''    
                    avg:2.521906090797878
                    count:12148
                    sum:30636.11519101262

                    val:2.472342014312744

                    '''
                    epoch_mse.update(np.mean(pred_err_all * pred_err_all), N)
                    epoch_mae.update(np.mean(abs(pred_err_all)), N)

                    epoch_loss_wf.update(loss_wf.item(), N)
                    epoch_mse_wf.update(np.mean(pred_err * pred_err), N)
                    epoch_mae_wf.update(np.mean(abs(pred_err)), N)  

                    # .item() 方法 用于从张量（tensor）中提取单个标量值。
                    epoch_loss_ff.update(loss_ff.item(), N)
                    epoch_mse_ff.update(np.mean(pred_err1 * pred_err1), N)
                    epoch_mae_ff.update(np.mean(abs(pred_err1)), N)              





            # ===============================================================================
            # 可视化
            # error:
                # writer.add_scalar('metric/loss/epoch_loss', epoch_loss , step) ====> 'AverageMeter' object has no attribute 'cpu'
                # print(epoch_loss)   ----> <__main__.AverageMeter object at 0x7de030c67f40>
                # print(epoch_loss.val)   ----> 3.140000104904175
            # ===============================================================================
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, epoch_mae.val))
                    log_file.write("metric/mse@{}: {}".format(step, epoch_mse.val))



                writer.add_scalar('metric/mae/epoch_mae', epoch_mae.val, step)
                writer.add_scalar('metric/mae/epoch_mae_wf', epoch_mae_wf.val, step)
                writer.add_scalar('metric/mae/epoch_mae_ff', epoch_mae_ff.val, step)
                # Got <class 'dict'>, but expected numpy array or torch tensor.
                writer.add_scalars('metric/mae/epoch_mae_Compare', { 
                    "mae.val" : epoch_mae.val,
                    "mae_wf.val" : epoch_mae_wf.val,
                    "mae_ff.val" : epoch_mae_ff.val}, step)

                writer.add_scalar('metric/mse/epoch_mse', epoch_mse.val, step)
                writer.add_scalar('metric/mse/epoch_mse_wf', epoch_mse_wf.val, step)
                writer.add_scalar('metric/mse/epoch_mse_ff', epoch_mse_ff.val, step)
                writer.add_scalars('metric/mse/epoch_mse_Compare', { 
                    "mse.val" : epoch_mse.val,
                    "mse_wf.val" : epoch_mse_wf.val,
                    "mse_ff.val" : epoch_mse_ff.val}, step)
                
                writer.add_scalar('metric/loss/epoch_loss', epoch_loss.val , step)
                writer.add_scalar('metric/loss/epoch_loss_wf', epoch_loss_wf.val, step)
                writer.add_scalar('metric/loss/epoch_loss_ff', epoch_loss_ff.val, step)
                writer.add_scalars('metric/loss/epoch_loss_Compare', { 
                    "loss.val" : epoch_loss.val,
                    "loss_wf.val" : epoch_loss_wf.val,
                    "loss_ff.val" : epoch_loss_ff.val}, step)



                step += 1


            # ===============================================================================

            self.logger.info(
                'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
                'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                    .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                            epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                            np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                            time.time() - epoch_start))
            
            self.logger_wf.info(
                'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
                'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                    .format(self.epoch, epoch_loss_wf.get_avg(), epoch_ot_loss_wf.get_avg(), epoch_wd_wf.get_avg(),
                            epoch_ot_obj_value_wf.get_avg(), epoch_count_loss_wf.get_avg(), epoch_tv_loss_wf.get_avg(),
                            np.sqrt(epoch_mse_wf.get_avg()), epoch_mae_wf.get_avg(),
                            time.time() - epoch_start))
            
            self.logger_ff.info(
                'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
                'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                    .format(self.epoch, epoch_loss_ff.get_avg(), epoch_ot_loss_ff.get_avg(), epoch_wd_ff.get_avg(),
                            epoch_ot_obj_value_ff.get_avg(), epoch_count_loss_ff.get_avg(), epoch_tv_loss_ff.get_avg(),
                            np.sqrt(epoch_mse_ff.get_avg()), epoch_mae_ff.get_avg(),
                            time.time() - epoch_start))
            
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)


            #=================================================================================
            # val stage
            #=================================================================================

            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
    # def val_epoch(self):
                args = self.args
                epoch_start = time.time()
                self.model.eval()  # Set model to evaluate mode

                epoch_res_wf = []
                epoch_res_ff = []
                epoch_res_all = []
                # 修改 
                for (inputs, count,count1, name )in self.dataloaders['val']:
                    inputs = inputs.to(self.device)
                    assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
                    with torch.set_grad_enabled(False):


                        # 修改
                        outputs, _ ,outputs1,_ = self.model(inputs)

                        res = count[0].item() - torch.sum(outputs).item()
                        # add
                        res1 = count1[0].item() - torch.sum(outputs1).item()
                        # add 计算总的误差
                        res_all = res + res1
                        
                        epoch_res_wf.append(res)
                        epoch_res_ff.append(res1)
                        epoch_res_all.append(res_all)

                epoch_res_all = np.array(epoch_res_all)
                mse_all = np.sqrt(np.mean(np.square(epoch_res_all)))
                mae_all = np.mean(np.abs(epoch_res_all))

                epoch_res_wf = np.array(epoch_res_wf)
                mse_wf = np.sqrt(np.mean(np.square(epoch_res_wf)))
                mae_wf = np.mean(np.abs(epoch_res_wf))

                epoch_res_ff = np.array(epoch_res_ff)
                mse_ff = np.sqrt(np.mean(np.square(epoch_res_ff)))
                mae_ff = np.mean(np.abs(epoch_res_ff))

                self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                                .format(self.epoch, mse_all, mae_all, time.time() - epoch_start))
                
                self.logger_wf.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                                .format(self.epoch, mse_wf, mae_wf, time.time() - epoch_start))
                
                self.logger_ff.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                                .format(self.epoch, mse_ff, mae_ff, time.time() - epoch_start))

                model_state_dic = self.model.state_dict()
                if (2.0 * mse_all + mae_all) < (2.0 * self.best_mse + self.best_mae):
                    self.best_mse = mse_all
                    self.best_mae = mae_all
                    self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                            self.best_mae,
                                                                                            self.epoch))
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
