
import os
from data.robotcar import RobotCar
from network.hypliloc import HypLiLoc
import torch
import sys
import time
import os.path as osp
from tqdm import tqdm
from tools.utils import AtLocCriterion
from torch.utils.data import DataLoader
import eval
from tools.utils import load_state_dict
from tools.utils import set_seed
from tools.options import Options
opt = Options().parse()
set_seed(7)







def main():
    print('device:{:s}'.format(opt.cuda))
    # Config
    device = "cuda"

    model = HypLiLoc()
    param_list = [{'params': model.parameters()}]
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)


    # Optimizer
    param_list = [{'params': model.parameters()}]
    if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
        print('learn_beta')
        param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
    if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
        print('learn_gamma')
        param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
    optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    print(opt.projected_lidar_type)
    print(opt.bev_type)


    # Load the dataset
    train_set = RobotCar(data_dir=opt.data_dir ,training=True)


    print(train_set)


    train_loader = DataLoader(
        train_set, 
        batch_size=opt.batchsize, 
        shuffle=True,
        num_workers=opt.nThreads,
        pin_memory=True,
    )



    # resume from epoch
    if opt.resume_epoch>0:
        weights_filename = 'logs/{:s}_{:s}_'.format(opt.dataset,opt.scene)  + opt.model+'_False/models/epoch_'+str(opt.resume_epoch)+'.pth.tar'
        if osp.isfile(weights_filename):
            checkpoint = torch.load(weights_filename, map_location=device)
            load_state_dict(model, checkpoint['model_state_dict'])
            print('Resume weights from {:s}'.format(weights_filename))
        else:
            print('Could not load weights from {:s}'.format(weights_filename))
            sys.exit(-1)



    model.to(device)
    train_criterion.to(device)
    experiment_name = opt.exp_name

    tq_mean_error_best = [10000., 10000., 0, 10000.]
    tq_median_error_best = [10000., 10000., 0, 10000.]




    t0 = time.time()
    model.train()
    for epoch in range(opt.resume_epoch+1, opt.epochs):
        txt = []


        pbar = tqdm(enumerate(train_loader))
        for batch_idx, feed_dict in pbar:

            lidar_float32 = feed_dict['lidar_float32'].to(device)
            projected_lidar_float32 = feed_dict['projected_lidar_float32'].to(device)
            pose_float32 = feed_dict['pose_float32'].to(device).detach()
            target = pose_float32.clone().detach()
                


            with torch.set_grad_enabled(True):

                output_dict = model(lidar_float32, projected_lidar_float32) 

                output_lidar = output_dict['output_lidar']
                output_prj = output_dict['output_prj']
                output_fusion  = output_dict['output_fusion']



                loss_tmp = train_criterion(output_lidar, target)
                loss_tmp += train_criterion(output_prj, target)
                loss_tmp += train_criterion(output_fusion, target)

                
                pbar.set_description(desc='loss:{:.4f}'.format(loss_tmp.item()))



            loss_tmp.backward()
            optimizer.step()
            optimizer.zero_grad()
            now_lr = optimizer.param_groups[0]["lr"]


        lr_scheduler.step()

        # save weights
        filename = osp.join(opt.models_dir, 'epoch_{:s}.pth.tar'.format(str(epoch)))
        checkpoint_dict = {
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(), 
            'criterion_state_dict': train_criterion.state_dict()
        }
        torch.save(checkpoint_dict, filename)


        # test
        if True:
            t_mean_error, q_mean_error, t_median_error, q_median_error = eval.main(str(epoch))

            def update_txt(txt, t_mean_error, q_mean_error, t_median_error, q_median_error, 
                tq_mean_error_best, tq_median_error_best):
                txt.append('mean:   t {:3.2f} m  q {:3.2f}'.format(t_mean_error, q_mean_error))
                txt.append('median: t {:3.2f} m  q {:3.2f}'.format(t_median_error, q_median_error))
                if t_mean_error+q_mean_error<tq_mean_error_best[-1]:
                    tq_mean_error_best = [t_mean_error, q_mean_error, epoch, t_mean_error+q_mean_error]
                if t_median_error+q_median_error<tq_median_error_best[-1]:
                    tq_median_error_best = [t_median_error, q_median_error, epoch, t_median_error+q_median_error]
                txt.append('tq mean   best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_mean_error_best))
                txt.append('tq median best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_median_error_best))

                return txt, tq_mean_error_best, tq_median_error_best
            

            txt, tq_mean_error_best, tq_median_error_best = update_txt(
                txt, 
                t_mean_error, q_mean_error, t_median_error, q_median_error,
                tq_mean_error_best, tq_median_error_best)


        # print   
        txt.append('Train/test {:s}\t Epoch {:d}\t Lr {:f}\t Time {:.2f}'.format(
            experiment_name, epoch, now_lr, time.time()-t0))
        txt.append('-----------------------')   


        # print and save txt
        f = open('results.txt', 'a')
        for info in txt:
            print(info)
            f.write(info)
            f.write('\n')
        f.close()

        
        f = open('results/results_{:s}_{:d}.txt'.format(opt.bev_type, opt.bev_resize_size), 'a')
        for info in txt:
            f.write(info)
            f.write('\n')
        f.close()

        
        
        t0 = time.time()




if __name__ == '__main__':

    set_seed(7)

    if osp.exists('results.txt'):
        os.remove('results.txt')
    if osp.exists('results/results_{:s}_{:d}.txt'.format(opt.bev_type, opt.bev_resize_size)):
        os.remove('results/results_{:s}_{:d}.txt'.format(opt.bev_type, opt.bev_resize_size))

    main()