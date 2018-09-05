import tensorflow as tf
import numpy as np

# X값을 정의함
# [0, 0]: [털, 날개]
x_data = np.array([
	[0, 0],
	[1, 0],
	[1, 1],
	[0, 0],
	[0, 0],
	[0, 1]
])

# Y값을 정의함
#	[1, 0, 0]: 기타
#	[0, 1, 0]: 포유류
#	[0, 0, 1]: 조류
y_data = np.array([
	[1, 0, 0],
	[0, 1, 0],
	[0, 0, 1],
	[1, 0, 0],
	[1, 0, 0],
	[0, 0, 1]
])

# X, Y값을 float32형식으로 정의하고 값을 받을 준비를 함
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 기본 신경망을 사용함
# 가중치(W)를 정의함
# [2, 3]: [입력층(특징수), 출력층(레이블수)]
# 랜덤값으로 -1.0 ~ 1.0 사이의 값을 생성함
# 학습용 데이터임
W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))

# 편향(b)을 정의함
# 0으로 값을 생성함
# 학습용 데이터임
b = tf.Variable(tf.zeros([3]))

# 행렬곱으로 가중치(W)를 계산하고 편향(b)을 더함
# 선형관계로 정의하고 X, Y의 관계를 분석함
# 적절한 가중치(W), 편향(b)을 구하기 위함
L = tf.add(tf.matmul(X, W), b)

# 계산한 행렬곱에 활성화 함수인 ReLU를 적용함
# 일반적으로 출력층에 활성화 함수를 사용하지는 않음
L = tf.nn.relu(L)

# 신경망을 통해 계산된 출력값을 softmax로 정리함
# 배열 내의 모든 값들의 합이 1이 되도록 수정함
model = tf.nn.softmax(L)

# 손실값을 구함
# One-hot encoding을 사용한 대부분의 모델은 교차 엔트로피(Cross-Entropy) 함수를 사용함
# 교체 엔트로피 값은 예측값~실제값 사이의 확률 분포 차이를 계산하는 방식임
# (실측값 * 모델값의 로그값)의 모든 배열값을 더하고, 배열값의 평균을 구함
# axis=1은 1번째 차원의 값을 다 더해서 값을 1개로 만들고 그 차원을 없앰
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 경사하강법 최적화 함수를 이용하여 손실값을 최소화함
# 학습률을 적절히 조절하여 계산해야 함(숫자 큼: 그래프의 경사가 급함, 숫자 작음: 그래프의 경사가 평평함)
# 학습률의 숫자가 작을 수록 학습 속도가 느림
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 세션을 초기화함
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 100회의 학습을 진행함
for step in range(100):
	sess.run(train_op, feed_dict={X: x_data, Y: y_data})

	# 학습 중, 10번마다 한번씩 로그를 출력함
	if (step + 1) % 10 == 0:
		print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 예측값을 정리함
# 모델값을 바로 출력하면 학습한 배열값 같은 확률이 출력됨
# argmax 함수를 사용하여 배열값 중 가장 큰 값의 순서(index, One-hot encoding)를 구함
# 순서에 해당하는 레이블값을 구함
prediction = tf.argmax(model, axis=1)

# 실측값을 정리함
# 실측값을 바로 출력하면 미리 정의한 배열값이 출력됨
# argmax 함수를 사용하여 배열값 중 가장 큰 값의 순서(index, One-hot encoding)를 구함
# 순서에 해당하는 레이블값을 구함
target = tf.argmax(Y, axis=1)

# 학습된 값을 이용하여 원하는 값을 출력함
print('예측:', sess.run(prediction, feed_dict={X: x_data}))
print('실측:', sess.run(target, feed_dict={Y: y_data}))

# 예측값과 실측값(레이블값)을 비교하여 학습결과의 성공여부를 판단함
# 학습결과가 맞으면 true, 틀리면 false로 계산함
# 계산한 결과를 100% 비율로 계산하여 정확도를 출력함
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
