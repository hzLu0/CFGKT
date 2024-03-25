import torch.utils.data
import torch.nn.utils
import numpy as np
from torch.utils.data import Dataset

class CFGKT_dataset(Dataset):
    def __init__(self, group, n_skills, n_concept,max_seq):
        self.samples = group
        self.n_skills = n_skills
        self.max_seq = max_seq
        self.data = []
        self.n_concept = n_concept

        for que, exe_cat, ans, timestamp, spendtime in self.samples:
            if len(que) >= self.max_seq:
                self.data.extend([(que[l:l + self.max_seq], ans[l:l + self.max_seq],
                                   exe_cat[l:l + self.max_seq],timestamp[l:l + self.max_seq],
                                   spendtime[l:l + self.max_seq]) for l in range(len(que)) if l % self.max_seq == 0])
            elif len(que) <= self.max_seq and len(que) > 50:
                self.data.append((que, ans, exe_cat,timestamp,spendtime))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content_ids, answered_correctly, exe_category, timestamp, spendtime = self.data[idx]
        content_ids = np.array(list(map(int, content_ids)))
        answered_correctly = np.array(list(map(int, answered_correctly)))
        exe_category = np.array(list(map(int, exe_category)))
        timestamp = np.array(list(map(int, timestamp)))
        spendtime = np.array(list(map(int, spendtime)))

        skill = np.ones(self.max_seq) * self.n_skills
        skill[:len(content_ids)] = content_ids

        concept = np.ones(self.max_seq) * self.n_concept
        concept[:len(content_ids)] = exe_category

        do_time = np.ones(self.max_seq) * 301
        do_time[:len(spendtime)] = spendtime
        do_time[len(content_ids) - 1] = 301

        time_stamp = np.ones(self.max_seq) * 1441
        time_stamp[:len(timestamp)] = timestamp
        time_stamp[len(content_ids) - 1] = 1441

        time_stamp_tensor = torch.LongTensor(time_stamp)
        a = time_stamp_tensor[:-1]
        sub = torch.cat((torch.LongTensor([0]), a), dim=-1)

        sub[len(content_ids) - 1] = 1441
        lagtime = time_stamp_tensor - sub
        lagtime[len(content_ids) - 1:] = 1441
        lagtime = lagtime.clip(0, 1441)


        mask = np.zeros(self.max_seq)
        mask[1:len(answered_correctly) - 1] = 0.9
        mask[len(answered_correctly) - 1] = 1

        labels = np.ones(self.max_seq) * -1
        labels[:len(answered_correctly)] = answered_correctly

        qa = np.ones(self.max_seq) * (self.n_concept * 2 + 1)
        qa[:len(exe_category)] = exe_category + answered_correctly * self.n_concept
        qa[len(exe_category) - 1] = self.n_concept * 2 + 1
        return (torch.LongTensor(qa), torch.LongTensor(skill), torch.LongTensor(labels), torch.FloatTensor(mask),
                torch.LongTensor(concept), torch.LongTensor(do_time), lagtime)