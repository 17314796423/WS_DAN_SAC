import torch
import torch.nn as nn
from utils.utils import sentences_to_indices, read_glove_vecs
import numpy as np
import _pickle as pickle
from utils.lstm import ClassEmbedding
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, input_dim, units, keep_prob):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, units)
        self.dropout = nn.Dropout(
            p=1 - keep_prob)  # Note that PyTorch's dropout rate is the probability of dropping a unit
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SAC_BAP(nn.Module):

    def __init__(self):
        dt = 1024
        de = 1024
        df = 768
        dj = 1024
        super(SAC_BAP, self).__init__()
        self.cls_emb_FC = FC(de, dt, keep_prob=.8)
        self.v_FC = FC(df, dt, keep_prob=.5)
        self.bap_emb_FC = FC(dt, dj, keep_prob=1.0)

    def forward(self, ftm, cls_emb):
        cls_emb = self.cls_emb_FC(cls_emb)
        # cls_emb.register_hook(lambda grad: print(f"Gradient of cls_emb: {torch.abs(torch.round(grad * 100) / 100).sum()}"))
        # print("Gradient of cls_emb.weight: ", self.cls_emb_FC.fc.weight.grad)
        # print("Gradient of cls_emb.bias: ", self.cls_emb_FC.fc.bias.grad)
        v = self.v_FC(ftm)
        # v.register_hook(lambda grad: print(f"Gradient of v: {torch.abs(torch.round(grad * 100) / 100).sum()}"))
        att_w_b_sfx = torch.einsum('bfd,bkd->bfk', (v, cls_emb))
        att_w = F.softmax(att_w_b_sfx, dim=1)
        v = v.permute(0, 2, 1)
        cls_emb = cls_emb.permute(0, 2, 1)
        bap_emb = torch.einsum('bdf,bfk,bdk->bd', (v, att_w, cls_emb))
        bap_emb = self.bap_emb_FC(bap_emb)
        bap_emb = torch.unsqueeze(bap_emb, 1)
        # bap_emb = torch.add(bap_emb, ftm)
        # att_w = tf.math.reduce_sum(att_w, axis=2, keepdims=True)
        return bap_emb, att_w, att_w_b_sfx


class SAC(nn.Module):
    def __init__(self, dataset_name, batch_size, use_gpu, prefix, **kwargs):
        super(SAC, self).__init__()
        dataset_name = dataset_name[0].upper() + dataset_name[1:]
        self.k = 10
        word2index, self.index2word, word2vec = read_glove_vecs(
            prefix + 'data/%s_glove6b_init_300d.npy' % dataset_name,
            prefix + 'data/%s_dictionary.pkl' % dataset_name)
        word2index[self.index2word[0]] = len(word2index)
        add_emb = np.expand_dims(word2vec[0], 0)
        word2vec = np.append(word2vec, add_emb, axis=0)
        word2vec[0] = np.zeros((word2vec.shape[1]))

        classes = np.array(pickle.load(open(prefix + 'data/%s_classes.pkl' % dataset_name, 'rb')))
        self.indices = torch.tensor(sentences_to_indices(classes, word2index, 4))
        self.index2word = np.array(self.index2word)
        self.cls_embedding = ClassEmbedding(word2vec, 1024, 300, use_gpu)
        self.sac_bap = SAC_BAP()
        self.dj = 1024
        self.classes_names = classes
        self.classes = len(classes)
        self.fc_new = nn.Conv2d(self.dj, self.k, kernel_size=1)
        self.initialized = False

    def indices_to_words(self, imgs_indices, ground_truth, Topk_idx):
        res = []
        for bi, indices in enumerate(imgs_indices):
            sentences = []
            # print('================================')
            for i, indice in enumerate(indices):
                # ii = indice.type(torch.int32)
                # non_zero_index = torch.nonzero(ii)[-1][0]
                # trimmed_arr = ii[:non_zero_index + 1]
                # trimmed_arr = trimmed_arr % len(self.index2word)
                # words = self.index2word[trimmed_arr.detach().cpu()]
                sentence = Topk_idx[bi][i]
                # print(self.classes_names[ground_truth[bi]])
                # print('--------------------------------')
                # print(' '.join(words))
                # print(sentence)
                # print('--------------------------------')
                sentences.append(sentence)
            # print('================================')
            res.append(sentences)
        return res

    def forward(self, ftm, logits, inputs, labels):
        if not self.initialized:
            self.indices = self.indices.to(ftm.device).unsqueeze(0).expand(ftm.shape[0], self.classes, 4)
            self.initialized = True
        # 获取每个样本的 top-k 类别的索引
        tmp_topk_cls_indices = torch.topk(logits, k=self.k, dim=-1, sorted=True)[1]
        um = torch.zeros((logits.shape[0], self.classes), device=logits.device)
        Topk_idx = tmp_topk_cls_indices.detach()
        # 扩展 topk_cls_indices 的维度，使其变为 [batch_size, k, 1]
        topk_cls_indices = tmp_topk_cls_indices.unsqueeze(2)
        # 使用广播将 topk_cls_indices 扩展为 [batch_size, k, 3]
        topk_cls_indices = topk_cls_indices.expand(topk_cls_indices.shape[0], self.k, 4)
        # 将 indices 转置为 [batch_size, 3, num_classes]
        params = self.indices.transpose(1, 2)
        # 将 topk_cls_indices 转置为 [batch_size, 3, k]
        ids = topk_cls_indices.transpose(1, 2)
        # 使用 gather 在最后一个维度上索引 params
        topk_cls = torch.gather(params, dim=-1, index=ids)
        # 将 topk_cls 转置为 [batch_size, k, 3]
        topk_cls = topk_cls.transpose(1, 2)
        # TCCS -> C_k  存的是class的词索引 这里是根据logits得到的C_k，

        # # SAC Top-k Class Embedding
        # # 得到Ek batch_size*k*1024

        cls_emb = self.cls_embedding(topk_cls)
        m = ftm.shape[-1]
        ftm = torch.reshape(ftm, (ftm.shape[0], ftm.shape[1], m * m))
        ftm = ftm.transpose(1, 2)  # 12*26*26*768

        # feature_shape = feature_maps.size() ## 12*768*26*26*
        # attention_shape = attention_maps.size() ## 12*num_parts*26*26
        # # print(feature_shape,attention_shape)
        # phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps)) ## 12*32*768
        # phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))
        # phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))
        # phi_I = phi_I.view(feature_shape[0],-1)
        # raw_features = torch.nn.functional.normalize(phi_I, dim=-1) ##12*(32*768)
        # pooling_features = raw_features*100
        # print(pooling_features.shape)
        bap_logits, att_w, att_w_b_sfx = self.sac_bap(ftm, cls_emb)
        # avg_pool = nn.AvgPool2d(kernel_size=m, stride=1, padding=0)
        bap_logits = bap_logits.transpose(1, 2)  # 12*676*768 -> 12*768*676
        # bap_logits = bap_logits.reshape(bap_logits.shape[0], -1, m, m)  # 12*768*676 -> 12*768*26*26
        bap_logits = bap_logits.reshape(bap_logits.shape[0], -1, 1, 1)  # 12*1024*1 -> 12*1024*1*1
        # bap_logits = avg_pool(bap_logits)  # 12*768*26*26 -> 12*768*1*1
        dropout = nn.Dropout(p=.2)
        bap_logits = dropout(bap_logits)
        bap_logits = self.fc_new(bap_logits)  # 12*768*1*1 -> 12*200*1*1
        bap_logits = bap_logits.reshape(bap_logits.shape[0], self.k)
        for i in range(bap_logits.shape[0]):
            um[i][tmp_topk_cls_indices[i]] = bap_logits[i]
        bap_logits = um
        Topk = topk_cls.detach()
        labels = labels.detach()
        result = self.indices_to_words(Topk, labels, Topk_idx)
        return bap_logits, att_w, topk_cls, {'ftm': ftm, 'result': torch.tensor(result, device=ftm.device), 'Topk': Topk, 'Topk_idx': Topk_idx, 'labels': labels, 'attention': att_w, 'attention_b_sfx': att_w_b_sfx}
