from typing import List

import torch.nn as nn
import logging
import warnings
import torch
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from paraphrase.utils.data import FewShotDataset, FewShotSSLParaphraseDataset, FewShotSSLFileDataset
from utils.math import euclidean_dist, cosine_similarity
import numpy as np
import collections

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())

class Contrastive_Loss(nn.Module):

    def __init__(self, tau=5.0):
        super(Contrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        M = dot_similarity(x1, x2)/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x)==1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss

class ContrastNet(nn.Module):
    def __init__(self, config_name_or_path, metric="euclidean", max_len=64, super_tau=1.0, unsuper_tau=1.0, task_tau=1.0):
        super(ContrastNet, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.encoder = AutoModel.from_pretrained(config_name_or_path).to(device)
        self.metric = metric
        self.max_len = max_len
        assert self.metric in ('euclidean', 'cosine')
        self.contrast_loss = Contrastive_Loss(super_tau)
        self.unsupervised_loss = Contrastive_Loss(unsuper_tau)
        self.task_loss = Contrastive_Loss(task_tau)

        self.warmed: bool = False

    def encode(self, sentences: List[str]):
        if self.warmed:
            padding = True
        else:
            padding = "max_length"
            self.warmed = True
        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding=padding
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        hidden = self.encoder.forward(**batch).last_hidden_state
        return hidden[:,0,:]

    def pred_proto(self, query, proto): 

        s = dot_similarity(query, proto)
        _, y_pred = s.max(1)

        return y_pred

    def pred_KNN(self, query, support, support_inds, k=1):
        pred_batch = []
        pred_tag = []

        s = dot_similarity(query, support)
        top_v, top_ids = torch.topk(s, k, dim=1)   

        top_v, top_ids = top_v.detach().cpu().numpy(), top_ids.detach().cpu().numpy()
        for i in range(len(query)):
            top_y = support_inds[top_ids[i]]
            y_count = {}
            y_value = {}
            for j in range(len(top_y)):
                index = top_y[j]
                if index not in y_count:
                    y_count[index] = 1
                    y_value[index] = [top_v[i, j]]
                else:
                    y_count[index] += 1
                    y_value[index].append(top_v[i, j])

            sort_y_count = sorted(y_count.items(), key=lambda x: x[1], reverse=True)

            pred_batch.append(sort_y_count[0][0])

        y_pred = torch.tensor(pred_batch).to(device)

        return y_pred  


    def loss(self, sample, supervised_loss_share: float = 0, task_loss_share: float = 0):
        """
        :param supervised_loss_share: share of supervised loss in total loss
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ],
            "x_augment":[
                {
                    "src_text": str,
                    "tgt_texts: List[str]
                }, .
            ]
        }
        :return:
        """
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        support_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long()
        support_inds = Variable(support_inds, requires_grad=False).to(device)
        
        has_augment = 'x_augment' in sample
        has_task_aug = 'task_xs' in sample

        supports = [item["sentence"] for xs_ in xs for item in xs_]
        queries = [item["sentence"] for xq_ in xq for item in xq_]
        x = (supports + queries)*2

        # unsupervised texts and augmentations
        if has_augment:
            n_unsupers_samples = len(sample["x_augment"])
            unsupers_supports = [item["tgt_texts"] for item in sample["x_augment"]]
            unsupers_queries = [item["src_text"] for item in sample["x_augment"]]
            x = x + unsupers_supports + unsupers_queries

        # unsupervised tasks and augmentations
        if has_task_aug:
            task_xs = sample['task_xs']
            n_task = len(task_xs)
            x = x + [item["sentence"] for xs in task_xs for xs_ in xs for item in xs_] + [np.random.choice(item["aug_texts"]) for xs in task_xs for xs_ in xs for item in xs_]

        z = self.encode(x)
        z_dim = z.size(-1)

        z_support = z[:len(supports)]
        z_query = z[len(supports):len(supports) + len(queries)]
        z_support_proto = z_support.view(n_class, n_support, z_dim).mean(dim=[1])

        if has_augment:
            z_unsuper_support = z[(len(supports) + len(queries))*2:(len(supports) + len(queries))*2+len(unsupers_supports)]
            z_unsuper_query = z[(len(supports) + len(queries))*2+len(unsupers_supports):(len(supports) + len(queries))*2+len(unsupers_supports)+len(unsupers_queries)]

        if has_task_aug:
            task_data = z[-(2*n_task*n_class*1):].view(2*n_task, n_class, 1, z_dim)
            task_data = task_data.mean(dim=[2]).mean(dim=[1])

            z_task = task_data[:n_task]
            z_task_aug = task_data[n_task:]

        # supervised contrastive loss
        z_query_in = z_query
        z_support_in = z_support       
        contrast_labels = support_inds.reshape(-1)

        supervised_loss = self.contrast_loss(contrast_labels, z_query_in, z_support_in)
        y_pred = self.pred_proto(z_query, z_support_proto)
        y_pred1 = self.pred_KNN(z_query, z_support, support_inds.reshape(-1), k=1)

        acc_val_supervised = torch.eq(y_pred, support_inds.reshape(-1)).float().mean()
        acc_val1 = torch.eq(y_pred1, support_inds.reshape(-1)).float().mean()

        final_loss = supervised_loss

        # unsupervised contrastive loss
        if has_augment:
            unsupervised_dists = dot_similarity(z_unsuper_query, z_unsuper_support)
            unsupervised_target_inds = torch.range(0, n_unsupers_samples-1).to(device).long()
            unsupervised_loss = self.unsupervised_loss(unsupervised_target_inds, z_unsuper_query, z_unsuper_support)
            _, y_hat_unsupervised = unsupervised_dists.max(1)
            acc_val_unsupervised = torch.eq(y_hat_unsupervised, unsupervised_target_inds.reshape(-1)).float().mean()

            # Final loss
            assert 0 <= supervised_loss_share <= 1
            final_loss = (supervised_loss_share) * final_loss + (1 - supervised_loss_share) * unsupervised_loss


        # task contrastive loss
        task_l = 0. 
        if has_task_aug:
            task_inds = torch.range(0, n_task-1).to(device).long()
            task_loss = self.task_loss(task_inds, z_task, z_task_aug)
            final_loss = final_loss + task_loss_share*task_loss
            task_l = task_loss.item()  
 

        if has_augment:
            return final_loss, {
                "metrics": {
                    "supervised_acc": acc_val_supervised.item(),
                    "unsupervised_acc": acc_val_unsupervised.item(),
                    "supervised_loss": supervised_loss.item(),
                    "unsupervised_loss": unsupervised_loss.item(),
                    "supervised_loss_share": supervised_loss_share,
                    "task_loss": task_l,
                    "final_loss": final_loss.item(),
                },
                "unsupervised_dists": unsupervised_dists,
                "target": support_inds
            }

        return final_loss, {
            "metrics": {
                "acc": acc_val_supervised.item(),
                "acc_knn": acc_val1.item(),
                "supervised_loss": supervised_loss.item(),
                "task_loss": task_l,
                "loss": final_loss.item(),
            },
            "target": support_inds
        }    

    def train_step(self, optimizer, episode, supervised_loss_share: float, task_loss_share: float):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share, task_loss_share=task_loss_share)
        loss.backward()
        optimizer.step()

        return loss, loss_dict


    def test_step(self, dataset: FewShotDataset, n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode = dataset.get_episode()

            with torch.no_grad():
                loss, loss_dict = self.loss(episode, supervised_loss_share=1, task_loss_share=1)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }
