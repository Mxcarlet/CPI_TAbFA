import torch.utils.data as torch_data
import torch
import scipy.io as scio
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import nltk
import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import json


def text_Loader(text_path,sentences_len=5,cache=True):
    print('sentences len:{}'.format(sentences_len))
    if not cache:
        tokenizer = AutoTokenizer.from_pretrained("./cache/finbert")
        Sentiment_model = AutoModelForSequenceClassification.from_pretrained("./cache/finbert").to('cuda:0')

        pdf_years = os.listdir(text_path)
        pdf_years.sort(key=lambda x: int(x))

        st_dic = {}
        data_time_list = []
        st_dic_list = []
        for pdf_year in tqdm(pdf_years):
            pdf_dirs = os.listdir(os.path.join(text_path, pdf_year))
            for pdf_dir in pdf_dirs:
                if 'Meeting' in pdf_dir:
                    data_time = pdf_dir.replace('Meeting - ', '').split(' ')
                    day = data_time[1]
                    data_time = data_time[0] + ' ' + data_time[-1]

                    time_format = datetime.datetime.strptime(data_time, '%B %Y')
                    data_time = datetime.datetime.strftime(time_format, '%Y-%m')

                    with open(os.path.join(text_path, pdf_year, pdf_dir, 'key_text_new.txt'), 'r') as f:
                        txt = f.readlines()
                        f.close()
                    sentences = nltk.sent_tokenize(txt[0])

                    sentences_sentiment = []
                    for s in sentences[:sentences_len]:
                        ks = tokenizer(s, return_tensors="pt")
                        for i in ks.keys():
                            ks[i] = ks[i].to('cuda:0')
                        with torch.no_grad():
                            sentences_sentiment.append(Sentiment_model(**ks).logits.cpu())
                    st_dic[data_time] = torch.cat(sentences_sentiment,0)
                    st_dic_list.append(torch.cat(sentences_sentiment, 0))

                    data_time_list.append(data_time)


        torch.save(torch.stack(st_dic_list, 0), './cache/st_tensor.pt')

        with open('./cache/data_time_list.txt', 'w+') as filehandle:
            json.dump(data_time_list, filehandle)
        torch.cuda.empty_cache()
    else:
        st_dic = {}
        with open('./cache/data_time_list.txt', 'r') as filehandle:
            data_time_list = json.load(filehandle)

        st_tensor = torch.load('./cache/st_tensor.pt')
        for i, time in enumerate(tqdm(data_time_list)):
            st_dic[time] = st_tensor[i,...]

    return st_dic

def Loader(data_path, text_path):
    data = scio.loadmat(data_path)
    scaler = StandardScaler()
    rawdatas = data.get('rawdata_org')
    timestamp = data.get('dates_f')
    df = pd.read_csv(data_path.replace('Datas.mat','2021-01.csv')).T.index.values.tolist()
    df.remove('sasdate')
    CPI_index = df.index('CPIAUCSL')
    scaler.fit(rawdatas)
    rawdatas = scaler.transform(rawdatas)
    target_datas = rawdatas[:,CPI_index:CPI_index+1]

    y_m = {}
    y_m_list = []

    for i in range(0, timestamp.shape[0]):
        year = str(1959 + int((timestamp[i][0] - 1959) * 12 / 12))
        month = str(round((timestamp[i][0] - 1959) * 12) % 12)
        if month == '0':
            year = str(int(year) - 1)
            month = '12'
        if int(year) > 2018:
            continue
        if int(year) == 2018 and int(month) > 1:
            continue
        y_m[len(y_m)] = year + '-' + month
        y_m_list.append(year + '-' + month)
    target_datas = target_datas[:len(y_m_list)]
    st_dic_datas = text_Loader(text_path)

    return target_datas, st_dic_datas, y_m, y_m_list

class Data_Set(torch_data.Dataset):
    def __init__(self, target_datas, st_dic_datas, y_m, y_m_list, data_index, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.y_m = y_m
        self.y_m_list = y_m_list
        self.st_dic_datas = st_dic_datas
        self.target_datas = target_datas
        self.data_index = data_index

    def __getitem__(self, index):
        i = self.data_index[index]
        target_datas = torch.tensor(self.target_datas[i:i+self.seq_len,:])
        target_datas_label = torch.tensor(self.target_datas[i+self.seq_len:i+self.seq_len+self.pred_len,:])

        st_s = []
        for t in range(i+1,i+self.seq_len+1):
            s = self.choice_text(t)
            mean_s = s.mean(dim=0)[0]-s.mean(dim=0)[1]
            st_s.append(mean_s.unsqueeze(0))
        st_s = torch.cat(st_s,0)

        return target_datas.float(), st_s.float(), target_datas_label.float()
        # target_datas | doc avg sentiment with timestamp series | target_label

    def __len__(self):
        return len(self.data_index)

    def choice_text(self, index):
        label_datatime = self.y_m_list[index]
        label_year = label_datatime.split('-')[0]
        label_month = label_datatime.split('-')[-1]
        j1 = []
        j2 = []
        for i in self.st_dic_datas.keys():
            if int(i.split('-')[0]) == int(label_year)-1:
                j1.append(i)

            if i.split('-')[0] == label_year:
                if int(i.split('-')[-1]) < int(label_month):
                    j2.append(i)
        j1.sort(key=lambda x: int(x.split('-')[-1]))
        j2.sort(key=lambda x: int(x.split('-')[-1]))
        j = j1 + j2

        return self.st_dic_datas[j[-1]]
