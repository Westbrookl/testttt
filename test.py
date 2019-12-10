import tensorflow as tf
import numpy as np
import random_graph_gen as rd
import network_topology_gen as nt
# import memory_pool as mp
# import gcn_layer_with_pooling_v2 as gcn
import environment_gen as en
import time
import random
import agent_with_batch_input as ag
import multiprocessing as mp
import os
import sys
import copy
import logging
import pickle
from scipy.stats import poisson


success_a3c=0
success_random=0
success_nr=0
generated=0

G=pickle.load(open('G.var','rb'))
node_attr=pickle.load(open('node_attr.var','rb'))
wmin=pickle.load(open('wmin.var','rb'))
wmax=pickle.load(open('wmax.var','rb'))

LOG_FILE="/home/yanzx1993/VNR_A3C_LOG/LOG"

ENTROPY_WEIGHT = 0.5
EPS = 1e-6
INPUT_FEATURES = 5
EXTRACTED_FEATURES = INPUT_FEATURES *64
SNODE_SIZE = 100
VNODE_FEATURES_SIZE = 3
#NUM_AGENT = mp.cpu_count()#4 or 8 i guess
NUM_AGENT=24
ORDERS = 3
GAMMA=0.99
ALIVE_TIME=50000
#saver=tf.train.Saver()

LAPLACIAN = rd.make_laplacian_matrix(G)
LAPLACIAN_LIST = rd.make_laplacian_list(G, SNODE_SIZE, ORDERS)
LAPLACIAN_TENSOR = np.stack(LAPLACIAN_LIST)

RANDOM_SEED=93


def random_gen(rate):
    rv=poisson(rate)
    plist=[]
    for i in range(int(rate*100+1)):
        plist.append(rv.cdf(i))
    rnd=np.random.rand()
    for i in range(len(plist)):
        if(rnd<=plist[i]):
            #print(i)
            return i


'''rv=poisson(0.04)
plist=[]
for i in range(5):
    plist.append(rv.cdf(i))
#print(plist)
for c in range(100):
    rnd=np.random.rand()
    for i in range(5):
        if rnd<=plist[i]:
            print(i)
            break

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('/home/yanzx1993/VNR_A3C_MODEL_ep_29000.ckpt.meta')
    new_saver.restore(sess, '/home/yanzx1993/VNR_A3C_MODEL_ep_29000.ckpt')
    print("restore success.")
    graph=tf.get_default_graph()
    print("graph:",graph)
    actor=graph.get_operation_by_name('actor').outputs[0]
'''



def test():
    generated=0
    success_a3c=0
    graph_a3c=copy.deepcopy(G)
    graph_random=copy.deepcopy(G)
    graph_nr=copy.deepcopy(G)
    step=0
    rate=0.04
    interval=1000
    env=en.Environment(str.format("Environment_{0}", 1), SNODE_SIZE, ALIVE_TIME,
                         imported_graph=[graph_a3c, node_attr, wmin, wmax], link_embedding_type="hybrid")
    env_random=en.Environment(str.format("Environment_{0}", 2), SNODE_SIZE, ALIVE_TIME,
                         imported_graph=[graph_random, node_attr, wmin, wmax], link_embedding_type="hybrid")
    env_nr=en.Environment(str.format("Environment_{0}", 3), SNODE_SIZE, ALIVE_TIME,
                         imported_graph=[graph_nr, node_attr, wmin, wmax], link_embedding_type="hybrid")
    with tf.Session() as sess, open(LOG_FILE + 'test', 'w') as log_file:
        actor = ag.ActorNetwork(sess, "actor_master", INPUT_FEATURES, SNODE_SIZE,
                                EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                                ORDERS, laplacian=LAPLACIAN_TENSOR)
        critic = ag.CriticNetwork(sess, "critic_master", INPUT_FEATURES, SNODE_SIZE,
                                  EXTRACTED_FEATURES,
                                  VNODE_FEATURES_SIZE, ORDERS, laplacian=LAPLACIAN_TENSOR)
        
       # new_saver = tf.train.import_meta_graph('/home/yanzx1993/VNR_A3C_MODEL_ep_45000.ckpt.meta')
        new_saver=tf.train.Saver()
        sess.graph.finalize()
        new_saver.restore(sess, '/home/yanzx1993/VNR_A3C_MODEL_ep_3000.ckpt')
        print("restore successfully")
        print(actor)
        successful=[]
        while (True):
            if(step>interval):
                interval+=1000
                print("step:",step)
                print("generated:",generated)
                print("success:",success_a3c)
                ratio=success_a3c/generated
                successful.append(ratio)
                print("accept rate:",ratio)
            events=random_gen(rate)
            if(events==0):
                step+=1
                env.time=step
                env.release_resource(env.time)
            else:
                while(events>0):
                    #events-=1
                    start = time.clock()
                    s,v = env.get_state()
                    #print("state acquired.")
                    #snode_batch.append(s)
                    #vnode_batch.append(v)
                    env.snode_state=s
                    env.vnode_state=v
                    action_prob = actor.predict(s, v)
                    print("current action prob:",action_prob)
                    #print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
                    action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[4]["attributes"],sess,phase="Testing")
                    print("action_pick:",action_pick)
                    is_terminal, failure, reward = env.perform_action(action_pick)
        
                    if(is_terminal!=0):
                        events-=1
                        generated+=1
                    now = time.clock()
                    
                    if is_terminal==1:
                        success_a3c+=0
                        #print(str.format("worker {0} failed.",worker_index))
                    elif is_terminal==2:
                        success_a3c+=1
                        #print(str.format("worker {0} success.",worker_index))
                    if(events==0):
                        step+=1
                    env.time=step
                    if is_terminal!=0:
                        env.release_resource(env.time)
                         
                        #print(str.format("worker {0} now assigned cpu list:{1}", worker_index,
                        #             env.current_assigned_cpu))
                        #print(str.format("worker {0} now with CPU in use:{1}",worker_index, env.substrate_network.attribute_list[0]["attributes"]))
                        #print(str.format("worker {0} now with bandwidth in use:{1}", worker_index,
                        #            env.substrate_network.attribute_list[1]["attributes"]))
                        print("current assigned VNR:", len(env.assigned_VNR_list))
            '''                    start = time.clock()
            s,v = env.get_state()
            print("state acquired.")
            #snode_batch.append(s)
            #vnode_batch.append(v)
            env.snode_state=s
            env.vnode_state=v
            action_prob = actor.predict(s, v)
            #print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
            action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[4]["attributes"],sess,phase="Testing")
            print("action_pick:",action_pick)
            is_terminal, failure, reward = env.perform_action(action_pick)
            #a_batch.append(action_one_hot)
            #r_batch.append(reward)

            if(is_terminal!=0):
                s,v = env.get_state()
                snode_batch.append(s)
                vnode_batch.append(v)
                exp_queue.put([snode_batch,vnode_batch,a_batch,r_batch,is_terminal])
                actor_parameters,critic_parameters=network_parameter_queue.get()
                #print("worker network parameter shape:",tf.shape(actor_parameters))
                actor.set_network_params(actor_parameters)
                critic.set_network_params(critic_parameters)

                del snode_batch[:]
                del vnode_batch[:]
                del a_batch[:]
                del r_batch[:]
            now = time.clock()
            #print(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start))
            #log_file.write(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start) + '\n')
            #log_file.flush()
            if is_terminal==1:
                #print(str.format("worker {0} failed.",worker_index))
                step +=3
            elif is_terminal==2:
                #print(str.format("worker {0} success.",worker_index))
                step+=1
            env.time=step
            if is_terminal!=0:
                env.release_resource(env.time)
                #print(str.format("worker {0} now assigned cpu list:{1}", worker_index,
                #             env.current_assigned_cpu))
                #print(str.format("worker {0} now with CPU in use:{1}",worker_index, env.substrate_network.attribute_list[0]["attributes"]))
                #print(str.format("worker {0} now with bandwidth in use:{1}", worker_index,
                #            env.substrate_network.attribute_list[1]["attributes"]))
                #print(str.format("worker {0} now assigned vnr list:{1}", worker_index,
                #             len(env.assigned_VNR_list)))

            
            start = time.clock()
            s,v = env.get_state()
            print("state acquired.")
            #snode_batch.append(s)
            #vnode_batch.append(v)
            env.snode_state=s
            env.vnode_state=v
            action_prob = actor.predict(s, v)
            #print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
            action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[4]["attributes"],sess,phase="Testing")
            print("action_pick:",action_pick)
            is_terminal, failure, reward = env.perform_action(action_pick)
            #a_batch.append(action_one_hot)
            #r_batch.append(reward)

            if(is_terminal!=0):
                s,v = env.get_state()
                snode_batch.append(s)
                vnode_batch.append(v)
                exp_queue.put([snode_batch,vnode_batch,a_batch,r_batch,is_terminal])
                actor_parameters,critic_parameters=network_parameter_queue.get()
                #print("worker network parameter shape:",tf.shape(actor_parameters))
                actor.set_network_params(actor_parameters)
                critic.set_network_params(critic_parameters)

                del snode_batch[:]
                del vnode_batch[:]
                del a_batch[:]
                del r_batch[:]
            now = time.clock()
            #print(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start))
            #log_file.write(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start) + '')
            #log_file.flush()
            if is_terminal==1:
                #print(str.format("worker {0} failed.",worker_index))
                step +=3
            elif is_terminal==2:
                #print(str.format("worker {0} success.",worker_index))
                step+=1
            env.time=step
            if is_terminal!=0:
                env.release_resource(env.time)
                #print(str.format("worker {0} now assigned cpu list:{1}", worker_index,
                #             env.current_assigned_cpu))
                #print(str.format("worker {0} now with CPU in use:{1}",worker_index, env.substrate_network.attribute_list[0]["attributes"]))
                #print(str.format("worker {0} now with bandwidth in use:{1}", worker_index,
                #            env.substrate_network.attribute_list[1]["attributes"]))
                #print(str.format("worker {0} now assigned vnr list:{1}", worker_index,
                #             len(env.assigned_VNR_list)))'''


for c in range (10):
    random_gen(0.08)
test()
