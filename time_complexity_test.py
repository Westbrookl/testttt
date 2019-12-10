import numpy as np
import time
import tensorflow as tf
import random

# start=time.clock()
# print(start)
# weight=np.random.normal(50,25,[4,16,3])
# matrix=np.random.normal(50,25,[3,100,100])
# feature=np.random.normal(50,25,[4,100])
# # weight=tf.truncated_normal([4,16,3],50,25)
# # matrix=tf.truncated_normal([3,100,100],50,25)
# # feature=tf.truncated_normal([4,100],50,25)
# for c in range(10):
#     for i in range (4):
#         for j in range(16):
#             for k in range(3):
#                 #x=tf.matmul([feature[i]],weight[i][j][k]*matrix[k])
#                 x=np.matmul(feature[i], weight[i][j][k] * matrix[k])
#                 #print(x)
#
# end=time.clock()-start
# print(end)


size = 200
order = 3
input = 5
output = 32 * 5
laplacian = tf.placeholder(tf.float32, shape=(order, size, size))
input_feature = tf.placeholder(tf.float32, shape=(input, size))
weights = tf.placeholder(tf.float32, shape=(input, output, order))
    # weights=tf.truncated_normal((input,output,order))
a=list()
#s = np.ones([output, size])
temp_a=np.zeros([size])
for j in range(output):
    temp_y=list()
    #count = np.zeros([size])
    for i in range(input):
        temp=list()
        for k in range(order):
            tensor=tf.matmul([input_feature[i]], weights[i][j][k] * laplacian[k])
            temp.append(tensor)
        feature=tf.add_n(temp)
        temp_y.append(feature)
    temp_a+=tf.add_n(temp_y)
    #a.append(temp_a)
aa=temp_a
#aa=tf.stack(a)
# a_reshape=tf.reshape(aa,[output*size])
# yy=tf.nn.relu(a_reshape)
# for i in range(output*size):
# yyy=tf.const(1,tf.float32)
# print(yyy)
# train_step=tf.train.AdamOptimizer(0.001).minimize(yyy)


with tf.Session() as sess:
    rand_array = np.random.rand(order, size, size)
    rand_feature = np.random.rand(input, size)
    rand_weights = np.random.rand(input, output, order)

    # start_time = time.time()
    # for i in range(10):
    #     for i in range(input):
    #         for j in range(output):
    #             for k in range(order):
    #                 c = np.dot(rand_feature[i], rand_weights[i][j][k] * rand_array[k])
    #     #np.dot(np.dot(rand_array,rand_array), rand_array)
    # print("--- %s seconds numpy multiply ---" % (time.time() - start_time))

    start_time = time.time()
    tf.global_variables_initializer().run()
    for i in range(10):
        bb=sess.run(aa, feed_dict={laplacian: rand_array, input_feature: rand_feature, weights: rand_weights})
        #print(bb)
    print("--- %s seconds tensorflow---" % (time.time() - start_time))
