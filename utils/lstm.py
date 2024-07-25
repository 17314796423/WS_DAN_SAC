import torch
import torch.nn as nn


class ClassEmbedding(nn.Module):
    def __init__(self, word2vec, rnn_size, emb_dim, use_gpu, dropout=0.2):
        super(ClassEmbedding, self).__init__()
        self.word2vec = torch.tensor(word2vec, requires_grad=True)
        self.rnn_size = rnn_size
        self.use_gpu = use_gpu
        # 创建 LSTM 单元
        self.lstm_1 = nn.LSTMCell(input_size=emb_dim, hidden_size=rnn_size)
        self.lstm_2 = nn.LSTMCell(input_size=rnn_size, hidden_size=rnn_size)
        self.dropout = nn.Dropout(p=dropout)
        self.initialized = False

    def forward(self, sentence):
        if not self.initialized:
            self.word2vec = self.word2vec.to(sentence.device)
            self.initialized = True
        batch, num_class, max_words = sentence.size()
        sentence = sentence.reshape(batch * num_class, max_words).to(torch.long)
        # 创建 word embedding
        embed_ques_W = nn.Embedding.from_pretrained(self.word2vec, freeze=False)
        # 初始化 LSTM 隐藏状态
        h_t_1 = torch.zeros(batch * num_class, self.rnn_size, device=sentence.device)
        c_t_1 = torch.zeros(batch * num_class, self.rnn_size, device=sentence.device)
        h_t_2 = torch.zeros(batch * num_class, self.rnn_size, device=sentence.device)
        c_t_2 = torch.zeros(batch * num_class, self.rnn_size, device=sentence.device)

        for i in range(max_words):
            cls_emb_linear = embed_ques_W(sentence[:, i])
            cls_emb_drop = self.dropout(cls_emb_linear)
            cls_emb = torch.tanh(cls_emb_drop)

            # LSTM 第一个层
            h_t_1, c_t_1 = self.lstm_1(cls_emb, (h_t_1, c_t_1))
            h_t_1 = self.dropout(h_t_1)

            # LSTM 第二个层
            h_t_2, c_t_2 = self.lstm_2(h_t_1, (h_t_2, c_t_2))
            h_t_2 = self.dropout(h_t_2)

        output = h_t_2.view(batch, num_class, self.rnn_size)

        return output
