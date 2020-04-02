import numpy
from collections import deque
numpy.random.seed(12345)


class InputData:
    """
    数据输入类的完整定义
    self属性：
        word_frequency: 文集中的词频统计，去掉低频词
        word2id: 词->id的映射，不算低频词
        id2word: id->词的映射，不算低频词
        sentence_count: 文集中共有多少个句子
        word_count: 文集中共有多少个词，不算低频词
    """
    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.word_pair_catch = deque()    # 双向队列
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    # 遍历文集进行初如化
    def get_words(self, min_count):
        self.input_file = open(self.input_file_name, encoding='utf-8')
        self.sentence_length = 0   # 整个文件中有多少个单词
        self.sentence_count = 0    # 总共有多少句子
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
        pass

    # 此函数构建sample_table是为了negative_sample,详细原理可参考：http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        # 由于往self.word_frequency中添加的时候key是按照0,1,2,...这样上涨的。
        # 因此应用enumerate，其实就是wid->c
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # 获取一个batch的数据，一个中心词和各context词构成的一系列 pair
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name, encoding='utf-8')
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    # 进行负采样
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    # 文集中共有多个pair对，结合batch_size来统计有多少batch的
    # 由于(sc-1)*ws*(ws+1)>sc*ws*ws，因此返回的count比实际要少ws(sc-ws-1)，我理解如此设计的
    # 目的是为了每次调用get_batch_pairs时，while都能正常退出
    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - \
               (self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
