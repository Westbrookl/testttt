import numpy as np
import tensorflow as tf
import random_graph_gen as rd
import time

# start = time.clock()
# node=100
# input_features=4
# output_features=4*32
# orders=3
# learning_rate=0.000001
# max_steps=90
# n=rd.make_random_graph(node,0.5)
# adj=rd.make_adj_matrix(n)
# lap=rd.make_laplacian_matrix(n)
# base=rd.make_laplacian_matrix(n)
# sess = tf.InteractiveSession()
#
#
# with tf.name_scope("graph_laplacian"):
#     lap_ls=[]
#     lap_ls.append(np.identity(node))
#     lap_ls.append(lap)
#     for i in range (2,orders):
#         lap=np.matmul(lap,base)
#         lap_ls.append(lap)
#
#
# with tf.name_scope("placeholders"):
#     x=tf.placeholder(tf.float32,[input_features,node],name="state")
#     y_=tf.placeholder(tf.float32,[None,node],name="actions")
#
#
# def weight_init(shape):
#     with tf.name_scope("weight_init"):
#         weight=tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(weight)
#
#
#
# def bias_init(shape):
#     with tf.name_scope("bias_init"):
#         bias=tf.constant(0.1,shape=shape)
#     return tf.Variable(bias)
#
#
#
# def gcn_filter(weights):
#     with tf.name_scope("gcn_filter"):
#         lap_filter=list()
#         for i in range (len(lap_ls)):
#             #print("weight %s:%s" % (i,weights[i]))
#             lap_filter.append(weights[i]*lap_ls[i])
#     return tf.add_n(lap_filter)
# elapsed = (time.clock() - start)
# print("Time used:",elapsed)
#
#
# def gcn_layer(snode_network_feature, vnode_feature, input_dim, output_dim, name, act=tf.nn.relu):
#     with tf.name_scope(name):
#         with tf.name_scope("gcn_weights"):
#             gcn_weights = weight_init([input_dim, output_dim, orders])
#            # snode_network_feature = weight_init([input_dim, node])
#         #with tf.name_scope("pooling_weights"):
#
#         with tf.name_scope("fc_weights"):
#             fc_weights = weight_init([len(vnode_feature),node*output_dim])
#         with tf.name_scope("biases"):
#             biases = bias_init([output_dim*node])
#         with tf.name_scope("gcn"):
#             x=list()
#             for j in range(output_dim):
#                 y = list()
#                 for i in range(input_dim):
#                     y.append(tf.matmul([snode_network_feature[i]],gcn_filter(gcn_weights[i][j])))
#                 temp_y=tf.add_n(y)
#                 x.append(temp_y)
#             temp_x=tf.stack(x, axis=0)
#             y_reshape=tf.reshape(temp_x,[-1,node*output_dim])
#             for i in range(len(vnode_feature)):
#                 y_reshape+=fc_weights[i]*vnode_feature[i]
#             y_reshape+=biases
#         with tf.name_scope("activations"):
#             activations=act(y_reshape,name="activations")
#     return activations
#
# def softmax_layer(input, input_dim, output_dim, name, act=tf.nn.softmax):
#     with tf.name_scope(name):
#         with tf.name_scope("weights"):
#             weights=weight_init([input_dim,output_dim])
#
#
#         with tf.name_scope("biases"):
#             biases=bias_init([output_dim])
#         with tf.name_scope("multiply"):
#             y=tf.matmul(input,weights)+biases
#         return act(y)
#
#
# #create=weight_init([input_features,node])
# create=np.random.rand(100,input_features,node)
# #label = tf.one_hot(node - 1, node, 1., 0., axis=0)
# label=np.random.rand(node)
#
#
#
#     #feature=weight_init([20,20]).eval()
# test=gcn_layer(x,[1,2],input_features,output_features,"test")
# out=softmax_layer(test,node*output_features,node,"out",act=tf.identity)
# print("hello")
# elapsed = (time.clock() - start)
# print("Time used:",elapsed)
# elapsed=time.clock()
#
#
# #label=list()
# #for i in range(node):
# #    for a in range(i):
#
#
#
#
#
# with tf.name_scope('loss'):
#     diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=out)
#     with tf.name_scope('total'):
#         cross_entropy = tf.reduce_mean(diff)
#
#
# with tf.name_scope('train'):
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#
#
# tf.global_variables_initializer().run()
# for b in range(max_steps):
#     elapsed = (time.clock() - elapsed)
#     print("Training Time used:", elapsed)
#     elapsed = time.clock()
#     sess.run(train_step,feed_dict={x:create[b]})

class gcn():
    def __init__(self,laplacian,act=tf.nn.relu):
        self.laplacian=laplacian
        self.act=act

    def __call__(self, input):

        return self.act(input)

    def