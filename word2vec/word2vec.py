from input_data import InputData
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=50,
                 window_size=5,
                 iteration=1,
                 initial_lr=0.025,
                 min_count=5):
        """
        初始化
        Args:
            input_file_name: 输入文件，用空格分隔
            output_file_name: 保存embedding的文件
            emb_dimention: embedding的尺寸,一般是50-500
            batch_size: 批大小
            window_size: 中心词单侧窗口大小
            iteration: 相当于epoch的概念
            initial_lr: 初始学习率
            min_count: 词频下限，严格低于此值则过滤
        """
        self.data = InputData(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))

        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  self.window_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data.item(), self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2])
    w2v.train()
