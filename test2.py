import tensorflow as tf

# X, Y값을 정의함
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# 가중치(W), 편향(b)을 정의함
# 랜덤값으로 -1.0 ~ 1.0 사이의 값을 생성함
# 학습용 데이터임
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# X, Y값을 float32형식으로 정의하고 값을 받을 준비를 함
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 행렬곱이 아니므로 일반적인 수식으로 계산함
# 선형관계로 정의하고 X, Y의 관계를 분석함
# 적절한 가중치(W), 편향(b)을 구하기 위함
hypothesis = W * X + b

# 손실값을 구함
# 예측값에서 실측값을 뺀 후, 결과값을 제곱하여 부호를 제거함
# 제곱한 결과값의 평균을 구함(이때, 값의 차원을 줄이고 1개의 값으로 추출)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 최적화 함수를 이용하여 손실값을 최소화함
# 학습률을 적절히 조절하여 계산해야 함(숫자 큼: 그래프의 경사가 급함, 숫자 작음: 그래프의 경사가 평평함)
# 학습률의 숫자가 작을 수록 학습 속도가 느림
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# 세션 블록을 정의함
with tf.Session() as sess:
	# 세션을 초기화함
	sess.run(tf.global_variables_initializer())

	# 100회의 학습을 진행함
	for step in range(100):
		# 최적화 함수와 손실값을 학습함
		# feed_dict 매개변수를 통해 가중치(W), 편향(b)의 값을 구함
		_, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
		print(step, cost_val, sess.run(W), sess.run(b))

	# 학습된 값을 이용하여 원하는 Y값을 구함
	print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
	print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
