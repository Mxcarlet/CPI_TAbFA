import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import math


class Pro_MLP_for_SimCLR(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=256):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        # self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TDCM(nn.Module):
    def __init__(self, d_model, topk, seq_len):
        super(TDCM, self).__init__()
        self.K = topk
        self.topk = seq_len
        self.queries_project = nn.Linear(d_model, d_model)
        self.keys_project = nn.Linear(d_model, d_model)
        self.values_project = nn.Linear(d_model, d_model)

    def cross_correlation(self, x1, x2):  # Compensate x2 according to x1
        B, L, _ = x1.shape
        _, S, E = x2.shape
        scale = 1. / math.sqrt(E)

        queries = self.queries_project(x1)
        keys = self.keys_project(x2)
        values = self.values_project(x2)

        q_fft = torch.fft.rfft(queries.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 1).contiguous(), dim=-1)
        # v_fft = torch.fft.rfft(values.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        mean_value = torch.mean(corr, dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), self.topk, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(self.topk)], dim=-1)
        tmp_corr = torch.softmax(weights * scale, dim=-1).unsqueeze(1).repeat(1, L, 1)
        t = []
        for i in range(self.topk):
            t.append(torch.roll(values, -int(index[i]), 1) * tmp_corr[..., i].unsqueeze(-1))
        v = torch.cat(t, -1)

        return v


    def forward(self, series, text_series):
        cc = self.cross_correlation(series, text_series)

        return cc[..., :self.K], cc[..., -self.K:]

def generate_binomial_mask(B, T, p=0.05):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class LSTMModel(nn.Module):
    def __init__(self, configs, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.topk = configs.topk
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(hidden_size, 1)

        self.TimeAlign = TDCM(1, self.topk, configs.seq_len)

        self.pre_norm1 = nn.LayerNorm(hidden_size)
        self.temp = 0.7

        self.enc_S = nn.Linear(1, 64)
        self.enc_T = nn.Linear(self.topk, 64)
        self.cls_token_S = nn.Parameter(torch.zeros(1, 1, 1))
        self.cls_token_T = nn.Parameter(torch.zeros(1, 1, self.topk))
        trunc_normal_(self.cls_token_S, std=.02)
        trunc_normal_(self.cls_token_T, std=.02)

        self.enc_S_pro = Pro_MLP_for_SimCLR(in_dim=64, out_dim=64)
        self.enc_T_pro = Pro_MLP_for_SimCLR(in_dim=64, out_dim=64)

    def forward(self, target, st_s_embedding, target_label):
        st_s_embedding = st_s_embedding.unsqueeze(-1)
        # Time Align
        st_s_embedding_c, st_s_embedding_nc = self.TimeAlign(target, st_s_embedding)

        # init
        init = torch.zeros((target.shape[0], self.pred_len, target.shape[2])).float().to(target.device)
        x_init = torch.cat([target, init], dim=1)
        init = torch.zeros((st_s_embedding_c.shape[0], self.pred_len, st_s_embedding_c.shape[2])).float().to(st_s_embedding_c.device)
        st_init = torch.cat([st_s_embedding_c, init], dim=1)

        # cls token embed
        cls_tokens_S = self.cls_token_S.expand(x_init.shape[0], -1, -1)
        cls_tokens_T = self.cls_token_T.expand(st_init.shape[0], -1, -1)
        x_init = torch.cat([cls_tokens_S, x_init], dim=1)
        st_init = torch.cat([cls_tokens_T, st_init], dim=1)
        st_s_embedding_nc = torch.cat([cls_tokens_T, st_s_embedding_nc], dim=1)
        # encoder
        x_init = self.enc_S(x_init)
        st_init = self.enc_T(st_init)
        st_nc_S = self.enc_S(st_s_embedding_nc[..., -1].unsqueeze(-1))  # Series negative samples
        st_nc_T = self.enc_T(st_s_embedding_nc)                         # Text negative samples

        # Feature Align
        rand_idx = np.random.randint(0 + 1, x_init.shape[1] - self.pred_len)

        Pro_S = F.normalize(self.enc_S_pro(x_init[:, rand_idx, :]))
        Pro_T = F.normalize(self.enc_T_pro(st_init[:, rand_idx, :]))
        Pro_S_nc = F.normalize(self.enc_S_pro(st_nc_S[:, rand_idx, :]))
        Pro_T_nc = F.normalize(self.enc_T_pro(st_nc_T[:, rand_idx, :]))

        Pro_S_cls = F.normalize(self.enc_S_pro(x_init[:, 0, :]))
        Pro_T_cls = F.normalize(self.enc_T_pro(st_init[:, 0, :]))
        Pro_S_cls_nc = F.normalize(self.enc_S_pro(st_nc_S[:, 0, :]))
        Pro_T_cls_nc = F.normalize(self.enc_T_pro(st_nc_T[:, 0, :]))

        # decoder
        x_enc = torch.cat([x_init, st_init], -1)
        out, _ = self.rnn(x_enc)
        out = self.pre_norm1(out)

        # out
        pre = self.fc_out(out[:, -self.pred_len:, :])
        # =======================================loss=======================================#
        # MSE Loss
        MSE = torch.nn.MSELoss()
        loss = MSE(pre, target_label)

        # Contrastive Loss
        loss_tsc_l = self.Info_NCE(Pro_S, Pro_S_nc, Pro_T, Pro_T_nc)
        loss_tsc_g = self.Info_NCE(Pro_S_cls, Pro_S_cls_nc, Pro_T_cls, Pro_T_cls_nc)

        loss = loss + loss_tsc_l + loss_tsc_g
        return loss, pre

    def Info_NCE(self, series, series_nc, text, text_nc):
        series_feat_all = torch.cat([series.t(), series_nc.t()], dim=1)     # series samples
        text_feat_all = torch.cat([text.t(), text_nc.t()], dim=1)           # text samples
        sim_s2t_S = series @ text_feat_all / self.temp
        sim_t2s_S = text @ series_feat_all / self.temp

        sim_targets = torch.zeros(sim_s2t_S.size()).to(sim_s2t_S.device)
        sim_targets.fill_diagonal_(1)

        loss_s2t_S = -torch.sum(F.log_softmax(sim_s2t_S, dim=1) * sim_targets, dim=1).mean()  # Info-NCE Loss
        loss_t2s_S = -torch.sum(F.log_softmax(sim_t2s_S, dim=1) * sim_targets, dim=1).mean()  # Info-NCE Loss
        loss_tsc_S = (loss_s2t_S + loss_t2s_S) / 2

        return loss_tsc_S

