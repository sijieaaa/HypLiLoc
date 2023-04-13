
import os

import torch
from torch import nn
import scipy.linalg as slin
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe
import numpy as np
import sys

from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
from torchvision.datasets.folder import default_loader
from collections import OrderedDict
from tools.options import Options
import random
import torchvision.transforms.functional as TVF
import os.path as osp
import shutil


opt = Options().parse()

class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
        return loss


class AtLocGridCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocGridCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        assert len(pred.shape)==3
        assert len(targ.shape)==3
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :, :3], targ[:, :, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, :, 3:], targ[:, :, 3:]) + self.saq
        return loss




class AtLocPlusCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, 
        learn_beta=True, learn_gamma=True):
        super(AtLocPlusCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        # absolute pose loss
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
                   torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq

        # get the VOs
        # pred_vos = calc_vos_simple(pred)
        # targ_vos = calc_vos_simple(targ)
        pred_vos = calc_vos_simple_batch(pred)
        targ_vos = calc_vos_simple_batch(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = (torch.exp(-self.srx) * self.t_loss_fn(pred_vos[:, :3], targ_vos[:, :3]) + self.srx
                #   + torch.exp(-self.srq) * self.q_loss_fn(pred_vos[:, 3:], targ_vos[:, 3:]) + self.srq
            )

        # # ---- qexp torch
        # pred_q = qexp_torch(pred[:,3:])
        # targ_q = qexp_torch(targ[:,3:])

        # # ---- calc vos q
        # pred_vos_q = calc_vos_q(pred_q)
        # targ_vos_q = calc_vos_q(targ_q)

        # vo_loss_q = torch.exp(-self.srq) * self.q_loss_fn(pred_vos_q, targ_vos_q) + self.srq
        

        # total loss
        loss = abs_loss + vo_loss
        # loss = abs_loss + vo_loss + vo_loss_q

        return loss













class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img

def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

def qexp_torch(q):
    n = torch.linalg.norm(q,dim=1)
    # q = (torch.cos(n), torch.sinc(n/torch.pi)*q)
    w = torch.cos(n).unsqueeze(1)
    xyz = torch.sinc(n/torch.pi).unsqueeze(1)
    xyz = torch.cat([xyz,xyz,xyz], dim=-1)
    xyz = xyz*q
    q = torch.hstack([w, xyz])
    return q


def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def calc_vos_simple_batch(poses):
    # poses_withoutlast = poses[:-1,:]
    vos = poses[1:,:] - poses[:-1,:]
    # vos = []
    # for p in poses:
    #     pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
    #     vos.append(torch.cat(pvos, dim=0))
    # vos = torch.stack(vos, dim=0)
    return vos



def calc_vos_q(poses_q):
    vos_q = []
    for i, q in enumerate(poses_q):
        if i==0:
            None
        elif i>0:
            q1 = poses_q[i-1,[1,2,3,0]].clone()
            q2 = poses_q[i  ,[1,2,3,0]].clone()
            q12 = quat_product(quat_inverse(q1), q2)[[3,0,1,2]]
            vos_q.append(q12)
    vos_q = torch.stack(vos_q, dim=0)
    return vos_q


def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    assert poses_in.shape[-1]==12
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out







def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict,strict=False)





def semantic_augmentation(images, semantic_masks, mode):

    num_samples = len(images)
    images_ref = torch.clone(images)

    semantic_masks = semantic_masks>0

    semantic_masks = semantic_masks.unsqueeze(1).expand_as(images)

    if mode == 'shuffle':
        ref_ids = torch.randint(0, num_samples, size=[num_samples])
        images_auged = images_ref*(~semantic_masks) + images_ref[ref_ids]*semantic_masks

    elif mode == 'add':   

        if random.random()<opt.mask_flip_rate:
            ref_ids = torch.randint(0, num_samples, size=[num_samples])
            images_auged = images_ref*TVF.hflip(~semantic_masks[ref_ids]) + TVF.hflip(images_ref[ref_ids])*TVF.hflip(semantic_masks[ref_ids])
        else:
            ref_ids = torch.randint(0, num_samples, size=[num_samples])
            images_auged = images_ref*(~semantic_masks[ref_ids]) + images_ref[ref_ids]*semantic_masks[ref_ids]  



    elif mode == 'shuffle_add':
        ref_ids = torch.randint(0, num_samples, size=[num_samples])
        images_auged = images_ref*(~semantic_masks) + images_ref[ref_ids]*semantic_masks
        

        ref_ids = torch.randint(0, num_samples, size=[num_samples])
        images_auged = images_auged*(~semantic_masks[ref_ids]) + images_ref[ref_ids]*semantic_masks[ref_ids]


    elif mode == 'add_shuffle':
        ref_ids = torch.randint(0, num_samples, size=[num_samples])
        images_auged = images_ref*(~semantic_masks[ref_ids]) + images_ref[ref_ids]*semantic_masks[ref_ids]

        ref_ids = torch.randint(0, num_samples, size=[num_samples])
        images_auged = images_auged*(~semantic_masks) + images_ref[ref_ids]*semantic_masks
        

        


    return images_auged






def set_seed(seed=7):
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(7)




def mkdir_custom(folder):
    if osp.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)





def quat_conjugation(quat):
    """
    Returns the conjugation of input batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        Conjugation of a unit quaternion is equal to its inverse.        
    """
    inv = quat.clone()
    inv[...,:3] *= -1
    return inv

def quat_inverse(quat):
    """
    Returns the inverse of a batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        - Inverse of null quaternion is undefined.
        - For unit quaternions, consider using conjugation instead.        
    """
    return quat_conjugation(quat) / torch.sum(quat**2, dim=-1, keepdim=True)

def quat_product(p, q):
    """
    Returns the product of two quaternions.

    Args:
        p, q (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L153
    batch_shape = p.shape[:-1]
    assert q.shape[:-1] == batch_shape
    p = p.reshape(-1, 4)
    q = q.reshape(-1, 4)
    product = torch.empty_like(q)
    product[..., 3] = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], axis=-1)
    product[..., :3] = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      torch.cross(p[..., :3], q[..., :3], dim=-1))
    return product.reshape(*batch_shape, 4)





    
