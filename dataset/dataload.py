import numpy as np
import torch


def dataloader(args, fold, config=None, phase=None):

    cond_path = '/DataPath/'
    idxd_path = '/IndexPath/rp%d_f%d.npz' % (args.rp, args.fold)

    conn_dict = np.load(cond_path)
    indx_dict = np.load(idxd_path)

    conn_d = conn_dict['fc']
    conn_l = conn_dict['label']
    conn_l[conn_l == -1] = 0
    trnidx = indx_dict['trn_idx']
    validx = indx_dict['val_idx']
    tstidx = indx_dict['tst_idx']

    conn_d = torch.tensor(conn_d, dtype=torch.float32)
    conn_l = torch.from_numpy(conn_l).long()

    train_loader = [conn_d[trnidx], conn_l[trnidx]]
    val_loader = [conn_d[validx], conn_l[validx]]
    test_loader = [conn_d[tstidx], conn_l[tstidx]]

    return train_loader, val_loader, test_loader
