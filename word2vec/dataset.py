from torch.utils.data import Dataset
from collections import Counter
import numpy as np


class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2, negative_sampling=True, neg_samples_num=5):
        # 对文本进行处理，生成中心词和背景词对
        tokenized_text = text.split()
        word_counts = Counter(tokenized_text)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.window_size = window_size
        self.data = []
        self.neg_samples_num = neg_samples_num
        self.negative_sampling = negative_sampling

        if negative_sampling:
            # Compute the frequency distribution for negative sampling:
            word_frequencies = np.array(list(word_counts.values()))
            self.neg_sampling_distribution = np.power(word_frequencies, 3 / 4)
            self.neg_sampling_distribution /= np.sum(self.neg_sampling_distribution)  # Normalize

        for idx, center_word in enumerate(tokenized_text):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_idx = idx + offset
                if context_idx < 0 or context_idx >= len(tokenized_text):
                    continue

                context_word = tokenized_text[context_idx]
                self.data.append((center_word, context_word))

    def get_negative_samples(self, context_idx):
        neg_samples = np.random.choice(len(self.idx_to_word),
                                       size=self.neg_samples_num,
                                       p=self.neg_sampling_distribution)
        return [idx for idx in neg_samples if idx != context_idx]

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        center_idx = self.word_to_idx[center_word]
        context_idx = self.word_to_idx[context_word]

        if self.negative_sampling:
            negative_samples = self.get_negative_samples(context_idx)
            return center_idx, context_idx, negative_samples
        else:
            return center_idx, context_idx

    def __len__(self):
        return len(self.data)


class CBOWDataset(Dataset):
    def __init__(self, text, window_size=2):
        # 对文本进行处理，生成中心词和上下文词对
        tokenized_text = text.split()
        word_counts = Counter(tokenized_text)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.window_size = window_size
        self.data = []

        for idx, center_word in enumerate(tokenized_text):
            context_words = []

            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_idx = idx + offset
                if context_idx < 0 or context_idx >= len(tokenized_text):
                    continue

                context_words.append(tokenized_text[context_idx])
            self.data.append((center_word, context_words))

    def __getitem__(self, idx):
        center_word, context_words = self.data[idx]
        center_idx = self.word_to_idx[center_word]
        context_idxs = [self.word_to_idx[word] for word in context_words]
        return center_idx, context_idxs

    def __len__(self):
        return len(self.data)
