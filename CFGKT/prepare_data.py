import torch.utils.data
import torch.nn.utils
import numpy as np
from torch.utils.data import Dataset

class CFGKT_dataset(Dataset):
    def __init__(self, group, n_ques, n_concept,max_seq):
        self.samples = group
        self.n_ques = n_ques
        self.max_seq = max_seq
        self.data = []
        self.n_concept = n_concept

        for que, concept, ans, timestamp, spendtime in self.samples:
            if len(que) >= self.max_seq:
                self.data.extend([(que[l:l + self.max_seq], ans[l:l + self.max_seq],
                                   concept[l:l + self.max_seq],timestamp[l:l + self.max_seq],
                                   spendtime[l:l + self.max_seq]) for l in range(len(que)) if l % self.max_seq == 0])
            elif len(que) <= self.max_seq and len(que) > 50:
                self.data.append((que, ans, concept,timestamp,spendtime))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        que, ans, concept, timestamp, spendtime = self.data[idx]
        que = np.array(list(map(int, que)))
        ans = np.array(list(map(int, ans)))
        concept = np.array(list(map(int, concept)))
        timestamp = np.array(list(map(int, timestamp)))
        spendtime = np.array(list(map(int, spendtime)))

        skill = np.ones(self.max_seq) * self.n_ques
        skill[:len(que)] = que

        concepts = np.ones(self.max_seq) * self.n_concept
        concepts[:len(que)] = concept

        do_time = np.ones(self.max_seq) * 301
        do_time[:len(spendtime)] = spendtime
        do_time[len(que) - 1] = 301

        time_stamp = np.ones(self.max_seq) * 1441
        time_stamp[:len(timestamp)] = timestamp
        time_stamp[len(que) - 1] = 1441

        time_stamp_tensor = torch.LongTensor(time_stamp)
        sub = torch.cat((torch.LongTensor([0]), time_stamp_tensor[:-1]), dim=-1)
        sub[len(que) - 1] = 1441
        interval = time_stamp_tensor - sub
        interval[len(que) - 1:] = 1441
        interval = interval.clip(0, 1441)


        mask = np.zeros(self.max_seq)
        mask[1:len(ans) - 1] = 0.9
        mask[len(ans) - 1] = 1

        labels = np.ones(self.max_seq) * -1
        labels[:len(ans)] = ans

        qa = np.ones(self.max_seq) * (self.n_concept * 2 + 1)
        qa[:len(concept)] = concepts + ans * self.n_concept
        qa[len(concept) - 1] = self.n_concept * 2 + 1
        return (torch.LongTensor(qa), torch.LongTensor(skill), torch.LongTensor(labels), torch.FloatTensor(mask),
                torch.LongTensor(concepts), torch.LongTensor(do_time), interval)
