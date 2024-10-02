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

class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        self.n += 1
        self.sum += other
        self.sumsq += other*other
        return self

    def __isub__(self, other):
        self.n += 1
        self.sum -= other
        self.sumsq += other*other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

    def var(self):
        from math import sqrt
        return sqrt(self.sumsq / max(self.n, 1) - self.mean()**2)

    def semean(self):
        from math import sqrt
        return self.var() / sqrt(max(self.n, 1))

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
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
        
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
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
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
    def __init__(self, threshold, titles, contents, ids, labels):
        from tqdm.notebook import tqdm
        self.labelfeats = { k: (n, v) for n, (k, v) in enumerate(getCategories(threshold, labels, ids)) }
        Xs = []
        ys = []
        text_titles = []
        text_contents = []
        for n, what in tqdm(enumerate(makeData(threshold, self.labelfeats, titles, contents, ids, labels))):
            title = torch.tensor(what['title'])
            content = torch.tensor(what['content'])
            text_title = what['text_title']
            text_content = what['text_content']
            
            text_titles.append(text_title)
            text_contents.append(text_content)
                        
            Xs.append(torch.cat((title, content)).unsqueeze(0))
            cur_ids = what['ids']
            y = []
            for id in cur_ids:
                label = labels[id]
                y.append(self.labelfeats[label][0])
            ys.append(y)
            
        self.Xs = torch.cat(Xs, dim=0)
        self.text_titles = text_titles
        self.text_contents = text_contents
        self.ys = ys
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

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

def get_reward(sample, ys):
    reward = []
    for x,ids in zip(sample, ys):
        if x in ids:
            reward.append(1)
        else:
            reward.append(0)
    reward = torch.tensor(reward, dtype=torch.float32)
    return reward

def pad_collate_fn(batch):
    Xs, text_titles, text_contents, ys = zip(*batch)
    
    max_len = max(len(y) for y in ys)
    
    padded_ys = [y + [-1] * (max_len - len(y)) for y in ys]  # You can use another padding value if 0 is not suitable
    
    Xs = torch.stack(Xs)
    padded_ys = torch.tensor(padded_ys)
    
    return Xs, text_titles, text_contents, padded_ys


class SpannerEG(torch.nn.Module):
    def __init__(self, actions, epsilon, tzero):
        super(SpannerEG, self).__init__()
        
        self.epsilon = epsilon
        self.tzero = tzero
        self.t = 0
        
        with torch.no_grad():
            batchactions = actions.unsqueeze(0)
            self.spanner = self._make_spanner(batchactions)
            
    def _make_spanner(self, actions):
        from math import log

        C = 2
        
        N, K, D = actions.shape
        device = actions.device
        detset = RankOneDetset(actions)
        design = torch.zeros(N, D, device=device).long()
                
        for i in range(D):
            psi, _ = detset.computePhi(i)
            dets = torch.abs(torch.bmm(actions, psi.unsqueeze(2))).squeeze(2)
            fstar, astar = torch.max(dets, dim=1)
            design[:, i] = astar
            detset.updateCoord(i, fstar, astar)
                        
        for _ in range(int(D * log(D))):
            replaced = False
            for i in range(D):
                psi, logdetfac = detset.computePhi(i)
                dets = torch.abs(torch.bmm(actions, psi.unsqueeze(2))).squeeze(2)
                fstar, astar = torch.max(dets, dim=1)
                                
                if torch.any(fstar >= C * torch.exp(logdetfac)):
                    design[:, i] = astar
                    detset.updateCoord(i, fstar, astar)
                    replaced = True
                    break
                    
            if not replaced:
                break
                
        return design

    def sample(self, fhat):
        epsilon = self.epsilon * pow(self.tzero / (self.t + self.tzero), 1/3)
        self.t += 1
        
        exploit = torch.argmax(fhat, dim=1, keepdim=True)
        exploreindex = torch.randint(low=0, high=self.spanner.shape[1], size=(fhat.shape[0], 1), device=fhat.device)
        explore = torch.gather(input=self.spanner[0,:].expand(fhat.shape[0], -1), dim=1, index=exploreindex)
        shouldexplore = (torch.rand(size=(fhat.shape[0], 1), device=fhat.device) < epsilon).long()
        sample = shouldexplore * (explore - exploit) + exploit
        return sample.squeeze(1)

def data(sentences):
    for sentence in sentences:
        yield sentence
        
def get_elements_by_indices(x, y):
    return [x[i] for i in y]
       
def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

def predict_labels(titles, contents, model, tokenizer, device):
    
    predicted_labels = []
    
    input_texts = [f"Title: {title}\nContent: {content}\nTask: Predict the associated label." for title, content in zip(titles, contents)]
    
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)

    output_ids = model.generate(input_ids)

    predicted_labels = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in output_ids
    ]
    return predicted_labels

def language_model_outputs(text_titles, text_contents, sentence_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device):
        batch_size = len(text_titles)
        
        start = time.time()
        
        predicted_entities = predict_labels(text_titles, text_contents, t5_model, t5_tokenizer, device)
        
        now = time.time()
        llm_time = now - start
        
        start = time.time()
        predicted_entities_embd = torch.FloatTensor(entity_model.encode(predicted_entities)).cuda()
        predicted_entities_embd = torch.unsqueeze(predicted_entities_embd, 1).repeat(1, num_entities, 1)
        
        entity_embd = torch.unsqueeze(entity_embd, 0).repeat(batch_size, 1, 1).cuda()
        
        cos_similarities = cos(predicted_entities_embd, entity_embd)
                
        predicted_labels = torch.argmax(cos_similarities, dim=-1)
        
        now = time.time()
        cos_time = now - start

        return predicted_labels.detach(), llm_time, cos_time
        
def learnOnline(dataset, rank, batch_size, cuda, seed, llm_type):
    import time
    torch.manual_seed(seed)
    
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens')
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    labelfeatsdict = { n: v for n, v in dataset.labelfeats.values() }
    labelfeats = [ torch.tensor(labelfeatsdict[n]).float().unsqueeze(0) for n in range(len(labelfeatsdict)) ]
    Zs = torch.cat(labelfeats, dim=0)
    
    if cuda:
        Zs = Zs.cuda()
        
    if rank is not None:
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(Zs, full_matrices=False)
            Zs = U[:, :rank] @ torch.diag(S[:rank])
    
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
            
    avloss_llm, sincelast_llm, avreward_llm, avrewardsincelast_llm, avreward_llm, rewardsincelast_llm = [ EasyAcc() for _ in range(6) ]
    
    avreward_llm_list = []
    avrewardsincelast_llm_list = []
    
    avreward_steps = []
    
    device = torch.device("cuda")

    model_name = "google/flan-t5-" + llm_type
    
    t5_model, t5_tokenizer = load_model_and_tokenizer(model_name, device)

    sentence_model = SentenceTransformer('bert-base-nli-mean-tokens').cuda()
    
    bandit_times = 0
    llm_times = 0
    cos_times = 0
    total_times = 0
    total_start = time.time()
    
    for bno, (Xs, text_titles, text_contents, ys) in enumerate(generator):
        Xs, ys = Xs.to(Zs.device), ys.to(Zs.device)
        start = time.time()

        with torch.no_grad():
            now = time.time()
            bandit_times += (now - start)
            lm_predicted_labels, llm_time, cos_time = language_model_outputs(text_titles, text_contents, sentence_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device = device)
            
            llm_times += llm_time
            cos_times += cos_time
            
            reward = get_reward(lm_predicted_labels, ys).unsqueeze(1).float().to(Zs.device)
            
        with torch.no_grad():
                            
            for pred, avreward, avrewardsincelast in zip([lm_predicted_labels], [avreward_llm], [avrewardsincelast_llm]):
                
                performance = get_reward(pred, ys).unsqueeze(1).float().to(Zs.device)
                avreward += torch.mean(performance)
                avrewardsincelast += torch.mean(performance)
            
        if bno % 100 == 0:
            now = time.time()
            
            avreward_llm_list.append(avreward_llm.mean().item())
            avrewardsincelast_llm_list.append(avrewardsincelast_llm.mean().item())
            
            avreward_steps.append(avreward_llm.n)
            
            sincelast_llm, avrewardsincelast_llm, rewardsincelast_llm = [ EasyAcc() for _ in range(3) ]

    now = time.time()
    
    avreward_llm_list.append(avreward_llm.mean().item())
    avrewardsincelast_llm_list.append(avrewardsincelast_llm.mean().item())

    avreward_steps.append(avreward_llm.n)
    total_times = time.time() - total_start
    
    results = {'steps':avreward_steps, 'avreward_llm':acc_llm_list, 'avrewardsincelast_llm':avrewardsincelast_llm_list}
    
    save_dir = './results/baseline/llm_type_' + llm_type + '/seed_' + str(seed) +'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'results.json', 'w') as fp:
        json.dump(results, fp)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--llm_type', type=str, default='base', help="Supports [base, small, large]")
    args = parser.parse_args()
    
    mydata = loadMyDataset(0)
    
    learnOnline(mydata, rank=50, batch_size=32, cuda=True, seed=args.seed, llm_type = args.llm_type)
    

