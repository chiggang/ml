from gensim.models import word2vec

model = word2vec.Word2Vec.load('ml.model')
result = model.most_similar(positive=['Python', '파이썬'])
#result = model.most_similar(positive=['의사'])
#result = model.most_similar(positive=['아빠', '여성'], negative=['남성'])
print('-----')
print(result)
