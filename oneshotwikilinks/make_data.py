import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentence_transformers import SentenceTransformer
import json
import os
import torch.nn as nn
from transformers import pipeline
from transformers import DistilBertTokenizer
import numpy as np
import copy
import time

def categoryCount():
    from collections import defaultdict
    import gzip
    import json
        
    counts = {}

    with gzip.open('entityfreq.gz', 'rt') as f:
        for line in f:
            try:
                freq, entity = line.strip().split()
            except:
                continue
            counts[entity] = int(freq)
            
    return counts

def getCategories(threshold):
    import gzip
    import json
    import re
    
    model = SentenceTransformer('all-mpnet-base-v2')
        
    for entity, freq in categoryCount().items():
        if freq >= threshold:
            niceentity = re.sub(r'_', r' ', entity)
            embedcat = model.encode([niceentity])[0]
            yield entity, embedcat

def datasetStats(threshold):
    numclasses = len([ entity for entity, freq in categoryCount().items() if freq >= threshold ])
    return { 'numclasses': numclasses, 'numexamples': threshold * numclasses }
            
def makeData(threshold, categories):
    from collections import defaultdict
    import json
    
    model = SentenceTransformer('all-mpnet-base-v2')
    catcount = defaultdict(int)
    
    with open('shuffled_dedup_entities.tsv') as f:
        batchline, batchencode, batchentity = [], [], []
        for line in f:
            try:
                entity, pre, mention, post = line.strip().split('\t')
            except:
                continue
                
            if entity in categories and catcount[entity] < threshold:
                catcount[entity] += 1
                batchline.append(line)
                batchencode.append(pre)
                batchencode.append(post)
                batchentity.append(entity)

                if len(batchline) == 5:
                    embed = model.encode(batchencode)

                    for n, (line, entity) in enumerate(zip(batchline, batchentity)):
                        embedpre, embedpost = embed[2*n], embed[2*n+1]
                        entityord, entityvec = categories[entity]
                        yield { 'line': line, 
                                'entityord': entityord, 
                                'entityvec': entityvec,
                                'pre': embedpre, 
                                'post': embedpost }

                    batchline, batchencode, batchentity = [], [], []
                
        if len(batchline):
            embed = model.encode(batchencode)

            for n, (line, entity) in enumerate(zip(batchline, batchentity)):
                embedpre, embedpost = embed[2*n], embed[2*n+1]
                entityord, entityvec = categories[entity]
                yield { 'line': line, 
                        'entityord': entityord, 
                        'entityvec': entityvec,
                        'pre': embedpre, 
                        'post': embedpost }

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold):
        from tqdm.notebook import tqdm
        self.labelfeats = { k: (n, v) for n, (k, v) in enumerate(getCategories(threshold)) } 
        Xs = []
        ys = []
        pre_texts = []
        post_texts = []
        text_entities = []
        for n, what in tqdm(enumerate(makeData(threshold, self.labelfeats))):
            pre = torch.tensor(what['pre'])
            post = torch.tensor(what['post'])
            Xs.append(torch.cat((pre, post)).unsqueeze(0))
            ys.append(what['entityord'])

            text_entity, text_pre, text_mention, text_post = what['line'].strip().split('\t')
            text_niceentity = re.sub(r'_', r' ', text_entity)
            
            pre_texts.append(text_pre)
            post_texts.append(text_post)
            text_entities.append(text_niceentity)
            
        self.Xs = torch.cat(Xs, dim=0)
        self.ys = torch.LongTensor(ys)
        self.pre_texts = pre_texts
        self.post_texts = post_texts
        self.text_entities = text_entities
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        # Select sample
        return self.Xs[index], self.ys[index], self.pre_texts[index], self.post_texts[index], self.text_entities[index]

def makeMyDataset(threshold):
    import gzip
    
    foo = MyDataset(threshold)
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'wb') as handle:
        import pickle
        pickle.dump(foo, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
def loadMyDataset(threshold):
    import gzip
    
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as handle:
        import pickle
        return pickle.load(handle)


class Bilinear(torch.nn.Module):
    def __init__(self, dobs, daction, naction, device):
        super(Bilinear, self).__init__()
        
        self.W = torch.nn.Parameter(torch.zeros(dobs, daction, device=device))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Xs, Zs):
        return torch.matmul(torch.matmul(Xs, self.W), Zs.T)
        
    def preq1(self, logits):
        return self.sigmoid(logits)

class RankOneDetset(object):
    def __init__(self, actions):
        self.actions = actions
        self.N, self.K, self.D = actions.shape
        self.device = actions.device
        
        self.batcheye = torch.eye(self.D, device=self.device).unsqueeze(0).expand(self.N, -1, -1)
        self.S = self.batcheye.clone()
        self.Sinv = self.batcheye.clone()
        self.logdetfac = torch.zeros(self.N, device=self.device)
        
    def computePhi(self, i):
        Sinvtopei = self.Sinv[:, i, :]
        return Sinvtopei, self.logdetfac
    
    def updateCoord(self, i, fstar, astar):
        Y = torch.gather(input=self.actions,
                         dim=1,
                         index=astar.reshape(self.N, 1, 1).expand(self.N, 1, self.D)
                        ).squeeze(1)
        Y /= torch.exp(self.logdetfac).reshape(self.N, 1)
        
        u = Y - self.S[:, :, i]
        Sinvu = torch.bmm(self.Sinv, u.unsqueeze(2)).squeeze(2)
        vtopSinv = self.Sinv[:, i, :]
        vtopSinvu = Sinvu[:, i].unsqueeze(1).unsqueeze(2)
        self.Sinv -= (1 / (1 + vtopSinvu)) * torch.bmm(Sinvu.unsqueeze(2), vtopSinv.unsqueeze(1))
        
        self.S[:,:,i] = Y
        thislogdet = 1/self.D * (torch.log(fstar) - self.logdetfac)
        scale = torch.exp(thislogdet).reshape(self.N, 1, 1)
        self.S /= scale
        self.Sinv *= scale
        self.logdetfac += thislogdet
    
def get_embd(dataset):
    temp_d = {}
    for k in dataset.labelfeats.keys():
        id, embd = dataset.labelfeats[k]
        temp_d[id] = torch.FloatTensor(embd)
    entity_embd = []
    for k in sorted(temp_d.keys()):
        entity_embd.append(temp_d[k])
    entity_embd = torch.stack(entity_embd)
    return entity_embd

if __name__ == '__main__':
    datasetStats(2000), datasetStats(1000), datasetStats(100)
    makeMyDataset(2000)
