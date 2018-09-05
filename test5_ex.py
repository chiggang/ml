import tensorflow as tf
import numpy as np

# 기초 데이터를 불러옴
# unpack=True: 데이터의 행과 열을 뒤바꿈
data = np.loadtxt('./test5_ex.csv', delimiter=',', unpack=True, dtype='float32')

# X값을 정의함
# np.transpose를 통해 열과 행을 뒤바꿈
# [0, 0]: [털, 날개]
x_data = np.transpose(data[0:2])

# Y값을 정의함
# np.transpose를 통해 열과 행을 뒤바꿈
#	[1, 0, 0]: 기타
#	[0, 1, 0]: 포유류
#	[0, 0, 1]: 조류
y_data = np.transpose(data[2:])

# 학습 횟수를 기억함
# trainable=False: 학습에 직접 사용하지 않음
global_step = tf.Variable(0, trainable=False, name='global_step')

# X, Y값을 float32형식으로 정의하고 값을 받을 준비를 함
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 심층 신경망(다층 신경망)을 사용함
# 가중치(W)를 정의함
# W1 [2, 10]: [입력층(특징수), 출력층(은닉층 뉴런수)]
# W2 [10, 3]: [입력층(은닉층 뉴런수), 출력층(레이블수)]
# 은닉층의 뉴런수는 실험을 통해 가장 적절한 수를 사용함
# 랜덤값으로 -1.0 ~ 1.0 사이의 값을 생성함
# 학습용 데이터임
W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0))

# 편향(b)을 정의함
# b1 [10]: 은닉층 뉴런수
# b2 [3]: 레이블수(분류수)
# 0으로 값을 생성함
# 학습용 데이터임
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([3]))

# 행렬곱으로 가중치(W)를 계산하고 편향(b)을 더함
# 선형관계로 정의하고 X, Y의 관계를 분석함
# 적절한 가중치(W), 편향(b)을 구하기 위함
L1 = tf.add(tf.matmul(X, W1), b1)

# 계산한 행렬곱에 활성화 함수인 ReLU를 적용함
# 일반적으로 출력층에 활성화 함수를 사용하지는 않음
L1 = tf.nn.relu(L1)

# 행렬곱으로 가중치(W)를 계산하고 편향(b)을 더함
# 선형관계로 정의하고 X, Y의 관계를 분석함
# 적절한 가중치(W), 편향(b)을 구하기 위함
L2 = tf.add(tf.matmul(L1, W2), b2)

# 계산한 행렬곱에 활성화 함수인 ReLU를 적용함
# 일반적으로 출력층에 활성화 함수를 사용하지는 않음
L2 = tf.nn.relu(L2)

# 신경망을 통해 계산된 출력값을 softmax로 정리함
# 배열 내의 모든 값들의 합이 1이 되도록 수정함
model = tf.add(tf.matmul(L2, W3), b3)

# 손실값을 구함
# One-hot encoding을 사용한 대부분의 모델은 교차 엔트로피(Cross-Entropy) 함수를 사용함
# 교체 엔트로피 값은 예측값~실제값 사이의 확률 분포 차이를 계산하는 방식임
# Tensorflow의 손실함수 중 하나인 교차 엔트로피(Cross-Entropy) 함수를 사용함
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# AdamOptimizer 최적화 함수를 이용하여 손실값을 최소화함
# AdamOptimizer 함수는 GradientDescentOptimizer 함수보다 보편적으로 성능이 조금 더 좋음
# 학습률을 적절히 조절하여 계산해야 함(숫자 큼: 그래프의 경사가 급함, 숫자 작음: 그래프의 경사가 평평함)
# 학습률의 숫자가 작을 수록 학습 속도가 느림
# global_step=global_step: 최적화 함수가 학습용 변수들을 최적화할 때마다 +1씩 증가함
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

# 세션을 정의함
sess = tf.Session()

# 위에서 정의한 모든 변수들을 불러옴
saver = tf.train.Saver(tf.global_variables())

# 해당 경로에 학습해둔 모델이 있는지 확인함
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	# 모델 있음: 학습된 모델을 불러옴
	# 기존에 학습한 ?회의 단계를 그대로 불러옴
	saver.restore(sess, ckpt.model_checkpoint_path)
else:
	# 모델 없음: 세션을 초기화함
	sess.run(tf.global_variables_initializer())

# 2회의 학습을 진행함
# 기존에 학습한 단계가 있으면 그 다음 단계부터 시작함
for step in range(2):
	sess.run(train_op, feed_dict={X: x_data, Y: y_data})

	# 학습 중, 로그를 출력함
	print('Step: %d, ' % sess.run(global_step), 'Cost: %3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 최적화가 완료된 후, 학습된 값들을 해당 경로에 저장함
# 기존에 생성된 모델이 있으면 global_step 수치가 모델 파일명 뒤에 붙어서 생성됨
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

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









"""
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
"""
