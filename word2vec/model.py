import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    self属性:
        emb_size: embedding表大小
        emb_dimention: 每个embedding的尺寸，常规是50到500
        u_embedding: center word的embedding，由输入层到隐藏层的权重矩阵决定，即input vector
        v_embedding: center word的上下文word的embedding，由隐藏层到输出层矩阵决定，即output vector
    """

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)  # 10000*100
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        """
        初始化两个权重矩阵
        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size],
        the elements of v_embedding are zeroes.
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        定义forward处理过程
        输入数据都是batch形式的，所有的输入都是word的id标识

        Args:
            pos_u: 正例的中心词
            pos_v: 正例中心词的context词
            neg_v: 一个正例pair对 对应5个负例pair对
        Returns:
            loss，相当于极大似然取反
        """
        emb_u = self.u_embeddings(pos_u)    # batch_size*100
        emb_v = self.v_embeddings(pos_v)    # batch_size*100
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)  # batch_size
        score = F.logsigmoid(score)    # 定义正样本的概率取对数
        neg_emb_v = self.v_embeddings(neg_v)  # batch_size*5*100

        # emb_u.unsqueeze(2)=batch_size*100*1,结果为：batch_size*5
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)  # 定义负样本的概率取对数
        return -1 * (torch.sum(score)+torch.sum(neg_score))

    def save_embedding(self, id2word, file_name, use_cuda):
        """
        存储embedding.原始为word_id和embedding,为了可读性需传入id->word的映射
        Args:
            id2word: id->word
            file_name: 输出文件名
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(10000, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
