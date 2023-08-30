from gensim.models import word2vec

model = word2vec.Word2Vec.load('text8_word2vec_model')
print('---计算2个词的相似度---')
word1 = 'man'
word2 = 'woman'
result1 = model.wv.similarity(word1, word2)
print(word1 + "和" + word2 + "的相似度为：", result1)
print('\n---计算1个词的关联列表---')
word = 'cat'
# 计算得出10个最相关的词
result2 = model.wv.most_similar(word, topn=10)

print("和" + word + "相关的10个词为：")
for item in result2:
    print(item[0], item[1])
