import torch as th
import torch.optim as optim
import torch.utils.data as torch_data
import random
from models.LSTM_M import LSTMModel as LSTM_M
from torch.utils.data.sampler import SequentialSampler
from dataloader import Data_Set, Loader
import datetime
from tqdm import tqdm
import argparse
import os
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pprint as ppr

global_log_file = None

def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


def mse(pred, label):
    loss = (pred - label)**2
    return th.mean(loss)

def mape(pred, label):
    diff = ((pred - label)/label).abs()
    return 100. * th.mean(diff)

def mae(pred, label):
    loss = (pred - label).abs()
    return th.mean(loss)

def rmse(pred, label):
    loss = (pred - label)**2
    return th.sqrt(th.mean(loss))

def TIC(pred, label):
    part1 = th.sqrt(th.mean((pred-label)**2))
    part2 = th.sqrt(th.mean((label)**2)) + th.sqrt(th.mean((pred)**2))
    return  part1/part2


def metric(pred, label):
    return mse(pred, label), mae(pred, label), rmse(pred, label), mape(pred, label), TIC(pred, label)

class Trainer(object):
    def __init__(self, nnet, optimizer, scheduler, module,
                 checkpoint, gpuid, train_loader, val_loader, test_loader, total_epoch, resume=None, args=None):
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        self.module = module

        self.scheduler = scheduler
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.total_epoch = total_epoch
        self.cur_epoch = 0
        self.no_impr = 0
        self.best_epoch = 0
        self.scheduler.best = 100
        self.best_score = th.inf

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            print("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            model2_dict = nnet.state_dict()
            state_dict = {k: v for k, v in cpt["model_state_dict"].items() if k in model2_dict.keys()}
            model2_dict.update(state_dict)
            nnet.load_state_dict(model2_dict)
            self.nnet = nnet.to(self.device)
        else:
            self.nnet = nnet.to(self.device)


    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))
        if not best and self.cur_epoch != None:
            th.save(
                cpt,
                os.path.join(self.checkpoint,
                             "Epoch{:d}.pt.tar".format(int(self.cur_epoch))))

    def run(self):
        with th.cuda.device(self.gpuid[0]):
            pprint("Set train mode...")
            for epoch in range(self.total_epoch):
                self.nnet.train()
                # train
                self.cur_epoch = epoch+1
                self.nnet.train()

                t_loss = 0
                for step, (target, st_s, target_label) in enumerate(tqdm(self.train_loader)):
                    train_target = target.to(self.device)
                    train_st_s_embedding = st_s.to(self.device)
                    train_target_label = target_label.to(self.device)
                    self.optimizer.zero_grad()
                    loss,_ = self.nnet(train_target, train_st_s_embedding,  train_target_label)
                    t_loss += loss

                    loss.backward()
                    self.optimizer.step()

                pprint('Train EPOCH {:d}, avg Loss={:.6f} (lr={:.3e})'.format( self.cur_epoch, t_loss / (step+1),self.optimizer.param_groups[0]["lr"]))

                t_loss,scores = 0,[]
                self.nnet.eval()
                with th.no_grad():
                    for step, (target, st_s, target_label) in enumerate(self.val_loader):
                        val_target = target.to(self.device)
                        val_st_s_embedding = st_s.to(self.device)
                        val_target_label = target_label.to(self.device)
                        loss, out = self.nnet(val_target, val_st_s_embedding, val_target_label)

                        t_loss += loss

                        scores.append(metric(out.cpu(), target_label.cpu())[0])

                avg_loss = t_loss / (step + 1)
                pprint('Eval EPOCH {:d}, avg Loss={:.6f}'.format(self.cur_epoch, avg_loss))
                score = np.average(scores)


                self.nnet.eval()
                t_loss = 0
                with th.no_grad():
                    for step, (target, st_s, target_label) in enumerate(self.test_loader):
                        test_target = target.to(self.device)
                        test_st_s_embedding = st_s.to(self.device)
                        test_target_label = target_label.to(self.device)
                        loss, out = self.nnet(test_target, test_st_s_embedding, test_target_label)

                        t_loss += loss

                avg_loss = t_loss / (step + 1)
                pprint('Test EPOCH {:d}, avg Loss={:.6f}'.format(self.cur_epoch, avg_loss))

                if score > self.best_score:
                    self.no_impr += 1
                else:
                    self.best_score = score
                    self.no_impr = 0
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(best=True)

                self.scheduler.step(score)
                sys.stdout.flush()
                self.save_checkpoint(best=False)

                if self.no_impr == 5:
                    pprint(
                        "Stop training cause no impr for {:d} epochs".format(
                            self.no_impr))
                    break

            pprint("Training Finish! | best epoch:{:d}".format(self.best_epoch))

            # test
            cpt = th.load(self.checkpoint+'/best.pt.tar',map_location='cpu')
            self.nnet.load_state_dict(cpt["model_state_dict"])
            pprint("Load checkpoint from {}, epoch {:d}".format(self.checkpoint+'/best.pt.tar', cpt["epoch"]))
            self.nnet.eval()
            preds = []
            trues = []

            with th.no_grad():
                for step, (target, st_s, target_label) in enumerate(self.test_loader):
                    test_target = target.to(self.device)
                    test_st_s_embedding = st_s.to(self.device)
                    test_target_label = target_label.to(self.device)
                    _, out = self.nnet(test_target, test_st_s_embedding, test_target_label)

                    preds.append(out.detach().cpu())
                    trues.append(test_target_label.detach().cpu())
            preds = th.cat(preds, 0)
            trues = th.cat(trues, 0)
            mse_n, mae_n, rmse_n, mape_n, tic_n = metric(preds, trues)
            pprint(
                'Test mse={:.6f}, Test mae={:.6f}, Test rmse={:.6f}, Test mape={:.2f}%, Test TIC={:.6f}'.
                    format(mse_n, mae_n, rmse_n, mape_n, tic_n))



        return mse_n, mae_n, rmse_n, mape_n, tic_n

def data_split(full_list, ratio1, ratio2, shuffle=False):
    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    offset2 = int(n_total * (ratio1 + ratio2))
    if n_total == 0 or offset1 < 1 or offset2 < 1:
        return [], [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3

def main(args):
    global global_log_file
    global_log_file = args.checkpoint + '/' + 'run.log'
    if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)

    target_datas, st_dic_datas, y_m, y_m_list = Loader(args.mat_path, args.text_path)

    a = len(list(range(0, (len(y_m_list) - args.seq_len - args.pred_len + 1))))
    pprint('total data:{}'.format(a))
    split = [0.8, 0.1, 0.1]
    train_index, val_index, test_index = data_split(list(range(0,(len(y_m_list) - args.seq_len - args.pred_len + 1))),split[0],split[1])

    train_set = Data_Set(target_datas, st_dic_datas, y_m, y_m_list, train_index, args.seq_len, args.pred_len)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=4, pin_memory=True)

    val_set = Data_Set(target_datas, st_dic_datas, y_m, y_m_list, val_index, args.seq_len, args.pred_len)
    val_sc = SequentialSampler(val_set)
    val_loader = torch_data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                       num_workers=4, sampler=val_sc, pin_memory=True)

    test_set = Data_Set(target_datas, st_dic_datas, y_m, y_m_list, test_index, args.seq_len, args.pred_len)

    test_sc = SequentialSampler(test_set)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                        num_workers=4, sampler=test_sc, pin_memory=True)


    if args.module == "LSTM_M":
        model = LSTM_M(args)
    else:
        raise RuntimeError("please check the module name!!!")
    gpuids = tuple(map(int, args.gpus.split(",")))

    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=True)
    trainer = Trainer(model, optimizer, scheduler, args.module,
                      args.checkpoint, gpuids, train_loader, val_loader, test_loader, args.epochs, args.resume, args)

    return trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",type=str,default="0",help="Training on which GPUs ""(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--module",type=str,default="LSTM_M",help="Training module ")
    parser.add_argument("--epochs",type=int,default=100,help="Number of training epochs")
    parser.add_argument("--mat_path",type=str,default="./datasets/Datas.mat")
    parser.add_argument("--text_path",type=str,default="/media/mscarlet/Datasets/datasets_Financial/FOMC")
    parser.add_argument("--resume",type=str,default="",help="Exist model to resume training from")
    parser.add_argument("--checkpoint",type=str,default='./checkpoint/test',help="Directory to dump models")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--topk',type=int, default=9, help='Number of time-aligned textual sentiment series in the TDMC')
    args = parser.parse_args()
    pprint("Arguments in command:\n{}".format(ppr.pformat(vars(args))))
    seed = [2021]
    mses_l, maes_l, rmses_l, mapes_l, tics_l = [], [], [], [], []
    for i in seed:
        print('seed:{}'.format(i))
        fix_seed = i
        random.seed(fix_seed)
        th.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        mses, maes, rmses, mapes, tics = main(args)
        mses_l.append(mses)
        maes_l.append(maes)
        rmses_l.append(rmses)
        mapes_l.append(mapes)
        tics_l.append(tics)
    mean_mse, std_mse = np.mean(mses_l), np.std(mses_l)
    mean_mae, std_mae = np.mean(maes_l), np.std(maes_l)
    mean_rmse, std_rmse = np.mean(rmses_l), np.std(rmses_l)

    f = open("result.txt", 'a')
    setting = '{}_sl{}_pl{}'.format(
        args.module,
        args.seq_len,
        args.pred_len,
        )
    f.write(setting + "  \n")
    f.write('mse_mean:{}, mae_mean:{}, rmse_mean:{}'
            .format(mean_mse, mean_mae, mean_rmse))
    f.write('\n')
    f.write('mse_std:{}, mae_std:{}, rmse_std:{}'
            .format(std_mse, std_mae, std_rmse))
    f.write('\n')
    f.write('\n')
    f.close()