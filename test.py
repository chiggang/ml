"""
import tensorflow as tf

x = tf.constant([[1.0, 2.0, 3.0]])
w = tf.constant([[2.0], [2.0], [2.0]])
y = tf.matmul(x, w)

print(w.get_shape())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)

print(result)


input_data = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
x = tf.placeholder(dtype=tf.float32, shape=[None, 3])
w = tf.Variable([[2.0], [2.0], [2.0]], dtype=tf.float32)
y = tf.matmul(x, w)

sess = tf.Session()
init = tf.global_variables_initializer
"""



"""
x_data
[[1, 2, 3], [4, 5,6]]

w
[[-0.07072455, 0.84648597]
 [-1.38593161, -1.2740159]
 [-0.89993954, 3.10966372]]

b
[[-1.54107356]
 [0.46613392]]

expr
[[-7.08347988, 6.08637142]
 [-12.14605999, 16.13998032]]
"""









import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])

x_data = [[1, 2, 3], [4, 5,6]]

#W = tf.Variable(tf.random_normal([3, 2]))
#b = tf.Variable(tf.random_normal([2, 1]))

W = tf.Variable([[-0.19910678, -0.6623396], [1.2703563, 0.2521515], [0.8525245, 0.11159096]])
b = tf.Variable([[1.8095733], [-0.24627878]])

expr2 = tf.matmul(X, W)
expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("x_data:")
print(x_data)
print("")

print("W:")
print(sess.run(W))
print("")

print("b:")
print(sess.run(b))
print("")

print("expr2:")
print(sess.run(expr2, feed_dict={X: x_data}))
print("")

print("expr:")
print(sess.run(expr, feed_dict={X: x_data}))
print("")

sess.close()
