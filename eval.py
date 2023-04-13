

import os
import time
from data.robotcar import RobotCar
from network.hypliloc import HypLiLoc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys
DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.options import Options
from torchvision import transforms
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from torch.utils.data import DataLoader
from tools.options import Options
opt = Options().parse()
from tools.utils import set_seed
set_seed(7)







def main(epoch='0'):


    opt = Options().parse()
    device = "cuda"


    # model for HypLiLoc
    model = HypLiLoc()
    model.eval()


    # criterion functions
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error

    # transform
    tforms = []
    tforms.append(transforms.ToTensor())



    # Load the dataset
    test_set = RobotCar(data_dir=opt.data_dir ,training=False)
    print(test_set)


    # pose mean and std
    pose_m = test_set.mean_t
    pose_s = test_set.std_t




    test_loader = DataLoader(test_set, batch_size=opt.batchsize_test, shuffle=False, 
        num_workers=opt.nThreads, 
        pin_memory=True,
    )



    pred_poses_list = []
    targ_poses_list = []
    

    # load weights
    model.to(device)
    weights_filename = 'logs/{:s}_{:s}_'.format(opt.dataset, 'full8')  + opt.model+'_False/models/epoch_'+str(epoch)+'.pth.tar'




    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded weights from {:s}'.format(weights_filename))
    else:
        print('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)



    # inference loop
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, feed_dict in pbar:

        # 3D point cloud
        lidar_float32 = feed_dict['lidar_float32'].to(device)
        # 2D spherical projection
        projected_lidar_float32 = feed_dict['projected_lidar_float32'].to(device)
        # ground truth pose
        pose_float32 = feed_dict['pose_float32'].to(device)
        target = pose_float32.clone().detach()

        


        with torch.set_grad_enabled(False):
            output_dict = model(lidar_float32, projected_lidar_float32)
            output_fusion  = output_dict['output_fusion']


        # we use fusion pose as the final pose
        output = output_fusion

        

        s = output.size() # [b,6]
        output = output.cpu().detach().numpy().reshape((-1, s[-1]))
        target = target.cpu().detach().numpy().reshape((-1, s[-1]))

        # exp the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        q = [qexp(p[3:]) for p in target]
        target = np.hstack((target[:, :3], np.asarray(q)))

        # un-normalize the predicted and target translations
        output[:, :3] = (output[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m



        for each_output in output:
            pred_poses_list.append(each_output)

        for each_target in target:
            targ_poses_list.append(each_target)


    # convert list to array
    def compute_error(pred_poses_list, targ_poses_list, fig_name):
        pred_poses = np.vstack(pred_poses_list)
        targ_poses = np.vstack(targ_poses_list)
        

        # calculate errors
        t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
        t_median_error = np.median(t_loss)
        q_median_error = np.median(q_loss)
        t_mean_error = np.mean(t_loss)
        q_mean_error = np.mean(q_loss)



        if opt.save_fig:
            real_pose = (pred_poses[:, :3] - pose_m) / pose_s
            gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
            plt.plot(real_pose[:, 1], real_pose[:, 0], color='red', linewidth=0.5)


            # colorful line
            norm = plt.Normalize(t_loss.min(), t_loss.max())
            norm_y = norm(t_loss)
            plt.scatter(gt_pose[:, 1], gt_pose[:, 0], c=norm_y, cmap='jet',linewidths=1)
            plt.xlabel('x [km]')
            plt.ylabel('y [km]')
            plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
            image_filename = osp.join(
                osp.expanduser(opt.results_dir), '{:s}_{:s}.png'.format(str(epoch),fig_name))
            plt.savefig(image_filename)
            plt.close()

        return t_mean_error, q_mean_error, t_median_error, q_median_error




    t_mean_error, q_mean_error, t_median_error, q_median_error = compute_error(
        pred_poses_list, targ_poses_list, '')
        
    return (
        t_mean_error, q_mean_error, t_median_error, q_median_error
    )
    




if __name__ == '__main__':

    tq_mean_error_best = [10000., 10000., 0, 10000.]
    tq_median_error_best = [10000., 10000., 0, 10000.]


    txt = []
    for i in range(80,81):
        epoch = str(i)
        weights_filename = 'logs/{:s}_{:s}_'.format(opt.dataset, 'full8')  + opt.model+'_False/models/epoch_'+str(epoch)+'.pth.tar'
        if not os.path.exists(weights_filename):
            continue
        
        
        t_mean_error, q_mean_error, t_median_error, q_median_error = main(epoch)



        txt.append('mean:   t {:3.2f} m  q {:3.2f}'.format(t_mean_error, q_mean_error))
        print('mean:   t {:3.2f} m  q {:3.2f}'.format(t_mean_error, q_mean_error))
        txt.append('median: t {:3.2f} m  q {:3.2f}'.format(t_median_error, q_median_error))
        print('median: t {:3.2f} m  q {:3.2f}'.format(t_median_error, q_median_error))


        if t_mean_error+q_mean_error<tq_mean_error_best[-1]:
            tq_mean_error_best = [t_mean_error, q_mean_error, int(epoch), t_mean_error+q_mean_error]
        if t_median_error+q_median_error<tq_median_error_best[-1]:
            tq_median_error_best = [t_median_error, q_median_error, int(epoch), t_median_error+q_median_error]


        txt.append('tq mean   best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_mean_error_best))
        txt.append('tq median best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_median_error_best))



        print('epoch: {:s}'.format(epoch))
        txt.append('epoch: {:s}'.format(epoch))


        print('{:s}'.format(opt.scene))
        txt.append('{:s}'.format(opt.scene))

        print('----------------')
        txt.append('----------------')


        import datetime
        if not os.path.exists('results_eval'): os.mkdir('results_eval')
        with open('results_eval/results_eval_{:s}.txt'.format(opt.scene),'w') as f:
            for each_txt in txt:
                f.writelines(each_txt)
                f.writelines('\n')
            
            f.writelines('\n')
            f.writelines(str(datetime.datetime.now()))