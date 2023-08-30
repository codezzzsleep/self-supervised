text = ["I", "love", "reading", "books", "She", "enjoys", "watching", "movies"]
# context window size is two
def create_cbow_dataset(text):
    data = []
    for i in range(2, len(text) - 2):
        context = [text[i - 2], text[i - 1],
                   text[i + 1], text[i + 2]]
        target = text[i]
        data.append((context, target))
    return data


def create_skipgram_dataset(text):
    import random
    data = []
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i - 2], 1))
        data.append((text[i], text[i - 1], 1))
        data.append((text[i], text[i + 1], 1))
        data.append((text[i], text[i + 2], 1))
        # negative sampling
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i - 1)
            else:
                rand_id = random.randint(i + 3, len(text) - 1)
            data.append((text[i], text[rand_id], 0))
    return data
cbow_data = create_cbow_dataset(text)
print("CBOW Data:")
for item in cbow_data:
    print(item)

skipgram_data = create_skipgram_dataset(text)
print("\nSkip-gram Data:")
for item in skipgram_data:
    print(item)
