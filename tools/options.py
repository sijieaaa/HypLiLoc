import argparse
import os
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):

        # path to your Radar_RobotCar folder
        # e.g. /home/workstation/Radar_RobotCar/
        self.parser.add_argument('--data_dir', type=str, default='/data/sijie/lidar/Radar_RobotCar/')



        self.parser.add_argument('--cuda', type=str, default='0')
        self.parser.add_argument('--nThreads', type=int, default=6)
        self.parser.add_argument('--resume_epoch', type=int, default=-1)
        self.parser.add_argument('--batchsize', type=int, default=32) 
        self.parser.add_argument('--batchsize_test', type=int, default=1) # test
        self.parser.add_argument('--divide_factor', type=float, default=1)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--lidar_model', type=str, default='PointNet2ClsSsgCPP')
        self.parser.add_argument('--model', type=str, default='ResNet34')
        self.parser.add_argument('--gattnorm', type=str, default='ln')
        self.parser.add_argument('--gattactivation', type=str, default='relu')
        self.parser.add_argument('--dataset', type=str, default='RobotCar')



        # full6   full7   full8    full9   
        self.parser.add_argument('--scene', type=str, default='full8')
        self.parser.add_argument('--grid_size', type=int, default=1)
        self.parser.add_argument('--grid_scale', type=float, default=0.1)
        self.parser.add_argument('--feat_dim', type=int, default=512)



        self.parser.add_argument('--save_out', type=bool, default=False)
        self.parser.add_argument('--save_fig', type=bool, default=True)
        self.parser.add_argument('--save_image', type=bool, default=False)


        self.parser.add_argument('--gpus', type=str, default='-1')
        self.parser.add_argument('--seed', type=int, default=7)
        self.parser.add_argument('--lstm', type=bool, default=False)
        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')


        self.parser.add_argument('--beta', type=float, default=-3.0)
        self.parser.add_argument('--gamma', type=float, default=-3.0, help='only for AtLoc+ (-3.0)')
        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')

        





    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpus.split(',')
        self.opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpus.append(id)



        self.opt.exp_name = '{:s}_{:s}_{:s}_{:s}'.format(self.opt.dataset, self.opt.scene, self.opt.model, str(self.opt.lstm))
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
        self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])

        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda)


        return self.opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)