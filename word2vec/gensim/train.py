from gensim.models import word2vec

sentences = word2vec.Text8Corpus('text8')

model = word2vec.Word2Vec(sentences,
                          sg=1,  # sg: 指示要使用 Skip-gram(设置为1) 还是 CBOW(设置为0) 模型。
                          vector_size=100,
                          window=5,
                          min_count=5,
                          negative=3,
                          sample=0.001,
                          hs=1,
                          workers=4)

model.save('text8_word2vec_model')
