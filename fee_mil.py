from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import config

#import reweighting.weight_learner2 as weight_learner
import reweighting
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix






class FeeMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader, True)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #maxpooling_out, _ = self.model.encoder(input.detach().clone()[0])
            #selected_num = torch.argsort(torch.softmax(maxpooling_out, 1)[:, 1], descending=True)
            #loss2 = self.criterion(torch.index_select(maxpooling_out, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)      
            j_0 = torch.softmax(_, 1)
            loss1 = self.criterion(output, targets)
            #fans_loss1 = 1.0 / torch.exp(torch.tensor(loss1.clone().detach().cpu().tolist()).cuda())
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))# * fans_loss1
            #losss3 = self.criterion(torch.index_select(_, dim=0, index=selected_num[-1 :]), targets.repeat(1)) * fans_loss1
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            if epoch <= self.cfgs["asynchronous"]:
                loss = loss2
            else:
                loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        for i, (feature, target, slide_id) in enumerate(loader):
            input = feature.cuda()
            '''
            if if_train:
                self.model.train()
                output, atten_target, atten_score = self.model.encoder.inference(input)  # [B, num_classes]            
                loss = self.criterion(atten_target, target.cuda())
                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()
            else:
            '''
            with torch.no_grad():
                #output = self.model.encoder.inference(input)
                #self.model.eval()
                output, atten_target, atten_score = self.model.encoder.inference(input)
            probs.extend(torch.cat((output, atten_score), 1).detach().cpu().numpy())
            #probs.extend(output.detach().cpu().numpy())
        return np.array(probs)

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                output = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader, False)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred)
            print(epoch, 'Validation:', score)

            torch.cuda.empty_cache()
        

