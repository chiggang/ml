from gensim.models import word2vec

data = word2vec.Text8Corpus("ml.wakati")

# size: 100차원의 벡터로 생성
# window: 주변 단어는 앞 뒤로 2개까지 적용
# min_count: 출현 빈도가 50번 미만인 단어는 분석에서 제외
# wokers: CPU 쿼드코어 사용
# iter: 학습을 100번 반복
# sg: 분석방법론(CBOW: 0, Skip-Gram: 1)
#model = word2vec.Word2Vec(data, size=100, window=2, min_count=50, workers=4, iter=100, sg=1)
model = word2vec.Word2Vec(data, size=100)
model.save("ml.model")

print("ok")
