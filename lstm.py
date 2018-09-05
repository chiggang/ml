import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

##########

# 현재 데이터와 다음 데이터를 Numpy로 생성함
def createDataset(dataset, lookBack=1):
  dataX, dataY = [], []

  # 전체 데이터를 불러와서 현재 데이터와 다음 데이터를 같은 순서의 numpy로 생성함
  # dataX: 현재 데이터
  # dataY: 다음 데이터
  for i in range(len(dataset) - lookBack - 1):
    # 현재 데이터를 기억함(0번째 컬럼을 기준으로 i번째 데이터)
    dataX.append(dataset[i:(i + lookBack), 0])

    # 다음 데이터를 기억함(0번째 컬럼을 기준으로 i + 1번째 데이터)
    dataY.append(dataset[i + lookBack, 0])

  # Pandas 형식을 Numpy 형식으로 변환하여 반환함
  return np.array(dataX), np.array(dataY)

##########

# 결과값을 위해 난수 시드를 수정함
np.random.seed(7)

# CSV 파일을 불러옴
# 마지막 3줄은 주석이므로 skipfooter를 이용하여 제외함
# usecols를 이용하여 2번째 컬럼을 불러옴
dataframe = pd.read_csv('./international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values

# 신경망을 사용하여 모델링할 때 적합한 부동 소수점 값으로 변경함
dataset = dataset.astype('float32')

# 데이터를 그래프로 확인함
#plt.plot(dataset)
#plt.show()

# 데이터의 정규화를 통해 성능을 높임
# 0~1 사이의 값으로 변경함
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 데이터 중, 67%는 학습 데이터로 사용하고 나머지는 테스트 데이터로 사용함
trainSize = int(len(dataset) * 0.67)
testSize = len(dataset) - trainSize
train, test = dataset[0:trainSize, :], dataset[trainSize:len(dataset), :]

# 학습 및 테스트 데이터를 이용하여 현재 데이터와 다음 데이터를 Numpy로 생성함
lookBack = 1
trainX, trainY = createDataset(train, lookBack)
testX, testY = createDataset(test, lookBack)

# 학습 데이터를 3차원 배열로 변환함
# 학습 데이터 행 갯수만큼의 1x1 배열
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 학습할 모델을 정의함(선형)
model = Sequential()

# 1개의 LSTM 레이어를 이용하여 순환신경망 모델를 정의함
# 4개의 뉴런 Hidden layer를 정의함
# input_shape는 입력값의 모양이며 1차원 배열로 정의함
model.add(LSTM(4, input_shape=(1, lookBack)))
#model.add(Dropout(0.3))  # 30% Drop

# 1개의 수치값을 예측하기 위하여 1개의 뉴런을 가진 Dense 레이어를 사용한 출력층을 정의함
model.add(Dense(1))

# 손실 및 최적화 함수를 정의함
# loss: 손실 함수(categorical_crossentropy, mse 등)
# optimizer: 최적화 함수(sgd, adam, rmsprop 등)
model.compile(loss='mse', optimizer='adam')

# 반복하며 모델을 학습시킴
# epochs: 반복 학습 횟수
# batch_size: 한번에 처리할 분량(숫자가 높을 수록 처리 과정이 빠름)
# verbose: 자세한 정보 표시 구분(0~2)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 학습 데이터와 테스트 데이터의 예측 모델을 생성함
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 원래의 값으로 다시 구성하여 반환함
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 평균 제곱근 오차를 계산하여 예측 값과 실제 값의 차리를 비교함
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

# 
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookBack:len(trainPredict) + lookBack, :] = trainPredict

#
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (lookBack * 2) + 1:len(dataset) - 1, :] = testPredict

# 데이터를 그래프로 확인함
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
