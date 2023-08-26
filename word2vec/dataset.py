from torch.utils.data import Dataset
from collections import Counter


class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2):
        # 对文本进行处理，生成中心词和背景词对
        tokenized_text = text.split()
        word_counts = Counter(tokenized_text)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.window_size = window_size
        self.data = []

        for idx, center_word in enumerate(tokenized_text):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_idx = idx + offset
                if context_idx < 0 or context_idx >= len(tokenized_text):
                    continue

                context_word = tokenized_text[context_idx]
                self.data.append((center_word, context_word))

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        center_idx = self.word_to_idx[center_word]
        context_idx = self.word_to_idx[context_word]
        return center_idx, context_idx

    def __len__(self):
        return len(self.data)
