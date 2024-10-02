import torch
import argparse
import os
import json

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

def getCategories(threshold, transformer_model):
    from sentence_transformers import SentenceTransformer
    import gzip
    import json
    import re
    
    model = SentenceTransformer(transformer_model)
        
    for entity, freq in categoryCount().items():
        if freq >= threshold:
            niceentity = re.sub(r'_', r' ', entity)
            embedcat = model.encode([niceentity])[0]
            yield entity, embedcat

def datasetStats(threshold):
    numclasses = len([ entity for entity, freq in categoryCount().items() if freq >= threshold ])
    return { 'numclasses': numclasses, 'numexamples': threshold * numclasses }
            
def makeData(threshold, transformer_model, categories):
    from collections import defaultdict
    from sentence_transformers import SentenceTransformer
    import json
    
    model = SentenceTransformer(transformer_model)
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
#             if n >= 1000:
#                 break
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
        # Select sample
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

def makeMyDataset(threshold, transformer_model):
    import gzip
    
    foo = MyDataset(threshold, transformer_model)
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

def learnOnline(dataset, rank, initlr, tzero, epsilon, epsilontzero, batch_size, cuda, seed):
    import time
    
    torch.manual_seed(seed)
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
    model = None
    log_loss = torch.nn.BCEWithLogitsLoss()
        
    avloss, sincelast, avreward, rewardsincelast = [ EasyAcc() for _ in range(4) ]
    
    avreward_bandit_list = []
    avrewardsincelast_bandit_list = []
    avreward_steps = []
    
    total_start = time.time()
        
    for bno, (Xs, text_titles, text_contents, ys) in enumerate(generator):
    
        Xs, ys = Xs.to(Zs.device), ys.to(Zs.device)
        
        if model is None:
            import numpy as np
            model = Bilinear(dobs=Xs.shape[1], daction=Zs.shape[1], naction=Zs.shape[0], device=Zs.device)
            sampler = SpannerEG(actions=Zs, epsilon=epsilon, tzero=epsilontzero)
            opt = torch.optim.Adam(( p for p in model.parameters() if p.requires_grad ), lr=initlr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda t: np.sqrt(tzero) / np.sqrt(tzero + t))
            start = time.time()
            
        opt.zero_grad()
        logit = model.forward(0.0001 * Xs, Zs)

        with torch.no_grad():
            sample = sampler.sample(logit)
            reward = get_reward(sample, ys).unsqueeze(1).float().to(Zs.device)
            
        samplelogit = torch.gather(input=logit, index=sample.unsqueeze(1), dim=1)
        loss = log_loss(samplelogit, reward)
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            avloss += loss
            sincelast += loss
            avreward += torch.mean(reward)
            rewardsincelast += torch.mean(reward)

        if bno % 100 == 0:
            now = time.time()
            
            avreward_bandit_list.append(avreward.mean().item())
            avrewardsincelast_bandit_list.append(rewardsincelast.mean().item())
            avreward_steps.append(avreward.n)
                        
            sincelast, rewardsincelast = [ EasyAcc() for _ in range(3) ]

    now = time.time()

    avreward_bandit_list.append(avreward.mean().item())
    avrewardsincelast_bandit_list.append(rewardsincelast.mean().item())
    avreward_steps.append(avreward.n)
    total_times = time.time() - total_start
    
    results = {'steps':avreward_steps, 'avreward_bandit':avreward_bandit_list, 'avrewardsincelast_bandit':avrewardsincelast_bandit_list, 'total_times':total_times}
        
    save_dir = save_dir = './results/bandit/seed_' + str(seed) +'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'results.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
        
    mydata = loadMyDataset(0)

    learnOnline(mydata, initlr=1/3, tzero=1000, rank=50, epsilon=1, epsilontzero=10, batch_size=32, cuda=True, seed=args.seed)




