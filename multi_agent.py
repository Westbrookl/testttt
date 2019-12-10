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

os.environ['CUDA_VISIBLE_DEVICES']=''


ENTROPY_WEIGHT = 0.5
EPS = 1e-6
INPUT_FEATURES = 3
EXTRACTED_FEATURES = INPUT_FEATURES *24
SNODE_SIZE = 100
VNODE_FEATURES_SIZE = 3
#NUM_AGENT = mp.cpu_count()#4 or 8 i guess
NUM_AGENT=24
ORDERS = 3
GAMMA=0.99
ALIVE_TIME=50000


NEW_DIR="/home/yanzx1993/new_not_averaged/"
#G,node_attr,wmin,wmax= rd.make_weighted_random_graph(SNODE_SIZE, 0.5, "normal",1)
#pickle.dump(G,open(NEW_DIR+"G.var",'wb'))
#pickle.dump(node_attr,open(NEW_DIR+"node_attr.var",'wb'))
#pickle.dump(wmin,open(NEW_DIR+"wmin.var",'wb'))
#pickle.dump(wmax,open(NEW_DIR+"wmax.var",'wb'))
#G=pickle.load(open('G.var','rb'))
GOOD_DIR="/home/yanzx1993/good_models/"

G=pickle.load(open(NEW_DIR+"G.var",'rb'))
print("total edges:",G.number_of_edges())
node_attr=pickle.load(open(NEW_DIR+"node_attr.var",'rb'))
wmin=pickle.load(open(NEW_DIR+"wmin.var",'rb'))
wmax=pickle.load(open(NEW_DIR+"wmax.var",'rb'))


ENVIRONMENT_LIST=list()
LAPLACIAN = rd.make_laplacian_matrix(G)
LAPLACIAN_LIST = rd.make_laplacian_list(G, SNODE_SIZE, ORDERS)
LAPLACIAN_TENSOR = np.stack(LAPLACIAN_LIST)
RANDOM_SEED=93
MAX_ITER=1000

N_WORKERS=mp.cpu_count()
epoch=0
#NN_MODEL=None
NN_MODEL='/home/yanzx1993/new_not_averaged/_ep_41000.ckpt'
LOG_DIR = "/home/yanzx1993/VNR_A3C_SUMMARY"
MODEL_DIR="/home/yanzx1993/VNR_A3C_MODEL"
LOG_FILE="/home/yanzx1993/VNR_A3C_LOG/LOG"



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


def test_env(env,generated,actor,sess):
    success=0
    i=0
    while i < generated:
        # test_time=env.time
        # env.release_resource(test_time)
        s, v = env.get_state()
        # print("state acquired.")
        # snode_batch.append(s)
        # vnode_batch.append(v)
        env.snode_state = s
        env.vnode_state = v
        action_prob = actor.predict(s, v)
        # print("current action prob:",action_prob)
        # print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
        action_one_hot, action_pick = actor.pick_action(action_prob, env.substrate_network.attribute_list[2]["attributes"],
                                                        sess, phase="Testing")
        # print("action_pick:",action_pick)
        is_terminal, failure, reward = env.perform_action(action_pick)
        now = time.clock()

        if is_terminal == 1:
            success += 0
            # print(str.format("worker {0} failed.",worker_index))
        elif is_terminal == 2:
            success += 1
            # print(str.format("worker {0} success.",worker_index))
        # if(events==0):
        # step+=1
        # env.time=step
        if is_terminal != 0:
            i += 1
            env.VNR_counter += 1
            env.release_resource(env.time)
    return success


def master(network_parameter_queue, exp_queue):
    assert len(network_parameter_queue) == NUM_AGENT
    assert len(exp_queue) == NUM_AGENT
    logging.basicConfig(filename=LOG_FILE + '_master_meta',
                        filemode='w',
                        level=logging.INFO)
    epoch = 0


    c_graph=copy.deepcopy(G)
    c_graph_2=copy.deepcopy(G)
    c_graph_3=copy.deepcopy(G)
    c_graph_4=copy.deepcopy(G)
    c_graph_5=copy.deepcopy(G)
    c_graph_6=copy.deepcopy(G)
    c_env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
    testing_list=c_env.generate_VNR_list(2000)
    c_env_2=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_2,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
    c_env_3=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.08)
    testing_list_frequent=c_env_3.generate_VNR_list(4000)
    c_env_4=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_4,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="edge_v")
    testing_list_edge=c_env_4.generate_VNR_list(2000)
    c_env_5=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_5,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="node_v")
    testing_list_node=c_env_5.generate_VNR_list(2000)
    c_env_6=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="intense")
    testing_list_intense=c_env_6.generate_VNR_list(2000)



    with tf.Session() as sess,open(LOG_FILE + '_master_0', 'w') as log_file:
        actor = ag.ActorNetwork(sess, "actor_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS,laplacian=LAPLACIAN_TENSOR)
        critic = ag.CriticNetwork(sess, "critic_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS,laplacian=LAPLACIAN_TENSOR)
        print(str("Network created."))
        #log_file.write(str("Network created.") + '\n')
        #log_file.flush()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(str("Variables initialized."))
        #log_file.write(str("Variables initialized.") + '\n')
        #log_file.flush()
        writer=tf.summary.FileWriter(LOG_DIR+"/master",sess.graph)
        saver=tf.train.Saver(max_to_keep=10000)
        print(str("Saver created."))
        #log_file.write(str("Saver created.") + '\n')
        #log_file.flush()
        loaded_model=NN_MODEL
        if NN_MODEL is not None:
            saver.restore(sess,loaded_model)
            print("model loaded")

        sess.graph.finalize()
        while(True):
            start = time.clock()
            actor_parameters = actor.get_network_params()
            critic_parameters = critic.get_network_params()
            print(str("Network parameter get."))
            #log_file.write(str("Network parameter get.") + '\n')
            #log_file.flush()
            for i in range(NUM_AGENT):
                network_parameter_queue[i].put([actor_parameters, critic_parameters])
                #print(str.format("Network parameter put to worker {0}.",i))
                #log_file.write(str.format("Network parameter put to worker {0}.",i)+ '\n')
                #log_file.flush()
            actor_gradient_batch=[]
            critic_gradient_batch=[]
            for i in range(NUM_AGENT):
                s_batch, v_batch, a_batch, r_batch, terminal = exp_queue[i].get()
                print(str.format("Get exp from worker {0}.",i))
                # print("s_batch:",s_batch)
                # print("v_batch:", v_batch)
                # print("a_batch:", a_batch)
                # print("r_batch:", r_batch)
                #log_file.write(str.format("Get exp from worker {0}.",i)+ '\n')
                #log_file.flush()
                actor_gradient, critic_gradient, td_batch=ag.compute_gradients(s_batch=np.stack(s_batch,axis=0),vnode_batch=np.stack(v_batch,axis=0),a_batch=np.vstack(a_batch),r_batch=np.vstack(r_batch),terminal=terminal,actor=actor,critic=critic)
                #print(str.format("Gradient from worker {0}.",i))
                #log_file.write(str.format("Gradient from worker {0}.",i)+ '\n')
                #log_file.flush()
                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)


            assert  NUM_AGENT==len(actor_gradient_batch)
            assert  len(actor_gradient_batch[0])==len(critic_gradient_batch[0])
            for i in range(len(critic_gradient_batch[0])):
                #print("actor gradient batch:",actor_gradient_batch[i])
                #print("critic gradient batch:",critic_gradient_batch[i])
                #print("shape:",tf.shape(actor_gradient_batch[i]))
                # print("gradient length of master:",len(actor_gradient_batch[0][i]))
                # print("gradient of master:",actor_gradient_batch[0][i])
                # for j in range (len(actor_gradient_batch[i])):
                #     print("gradient shape of element %d:"%j, np.shape(actor_gradient_batch[i][j]))
                actor.apply_gradients(actor_gradient_batch[0][i])
                critic.apply_gradients(critic_gradient_batch[0][i])

            now = time.clock()
            print(str.format("training step {0} costs:{1}",epoch,now - start))
            #log_file.write(str.format("training step {0} costs:{1}",epoch,now - start)+'\n')
            #log_file.flush()
            epoch+=1
            if(epoch%1000==0):
                save=saver.save(sess,NEW_DIR+"_ep_"+str(epoch)+".ckpt")
                print("Model saved in file:"+save)
            if(epoch%1000==0):
                master_graph=copy.deepcopy(G)
                master_graph_2=copy.deepcopy(G)
                master_graph_3=copy.deepcopy(G)
                master_graph_5=copy.deepcopy(G)
                master_graph_4=copy.deepcopy(G)
                master_graph_6=copy.deepcopy(G)

                master_env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
                master_env.testing_VNR_list=copy.deepcopy(testing_list)
                
                master_env_2=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph_2,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
                master_env_2.testing_VNR_list=copy.deepcopy(testing_list)
                   
                master_env_3=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.08)
                master_env_3.testing_VNR_list=copy.deepcopy(testing_list_frequent)
                master_env_4=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph_4,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="edge_v")
                master_env_4.testing_VNR_list=copy.deepcopy(testing_list_edge)
                master_env_5=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph_5,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="node_v")
                master_env_5.testing_VNR_list=copy.deepcopy(testing_list_node)

                master_env_6=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[master_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="intense")
                master_env_6.testing_VNR_list=copy.deepcopy(testing_list_intense)



                print("Start Testing:")
                success=test_env(master_env,2000,actor,esss)
                success_r=test_env(master_env_2,2000,actor,sess)
                success_f=test_env(master_env_3,4000,actor,sess)
                success_e=test_env(master_env_4,2000,actor,sess)
                success_n=test_env(master_env_5,2000,actor,sess)
                success_i=test_env(master_env_6,2000,actor,sess)
                #length=len(testing_list)
                '''success=0
                success_r=0
                generated=0
                test_time=0
                env=master_env
                env_2=master_env_2
                while generated<2000:
                    #test_time=env.time
                    #env.release_resource(test_time)
                    s,v = env.get_state()
                    #print("state acquired.")
                    #snode_batch.append(s)
                    #vnode_batch.append(v)
                    env.snode_state=s
                    env.vnode_state=v
                    action_prob = actor.predict(s, v)
                    #print("current action prob:",action_prob)
                    #print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
                    action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[2]["attributes"],sess,phase="Testing")
                    #print("action_pick:",action_pick)
                    is_terminal, failure, reward = env.perform_action(action_pick)
                    now = time.clock()
                    
                    if is_terminal==1:
                        success+=0
                        #print(str.format("worker {0} failed.",worker_index))
                    elif is_terminal==2:
                        success+=1
                        #print(str.format("worker {0} success.",worker_index))
                    #if(events==0):
                        #step+=1
                    #env.time=step
                    if is_terminal!=0:
                        generated+=1
                        env.VNR_counter+=1
                        env.release_resource(env.time)
                generated=0
                while generated<2000:    
                    s_2,v_2 = env_2.get_state()
                    #print("state acquired.")
                    #snode_batch.append(s)
                    #vnode_batch.append(v)
                    env_2.snode_state=s_2
                    env_2.vnode_state=v_2
                    #action_prob = actor.predict(s, v)
                    #print("current action prob:",action_prob)
                    #print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
                    action_one_hot_2,action_pick_2=actor.random_pick_action()
                    #print("action_pick:",action_pick)
                    is_terminal_2, failure_2, reward_2 = env_2.perform_action(action_pick_2)
                    now = time.clock()

                    if is_terminal_2==1:
                        success_r+=0
                        #print(str.format("worker {0} failed.",worker_index))
                    elif is_terminal_2==2:
                        success_r+=1
                        #print(str.format("worker {0} success.",worker_index))
                    #if(events==0):
                        #step+=1
                    #env.time=step
                    if is_terminal_2!=0:
                        generated+=1
                        env_2.VNR_counter+=1
                        env_2.release_resource(env_2.time)'''
  

                    
                print("current acc ratio:",success/2000)
                fo=open("aaa","a")
                fo.write(str.format("static acc_ratio:{0}\n",success/2000))
                fo.write(str.format("static ran_ratio:{0}\n",success_r/2000))
                fo.write(str.format("freq_0.08_ratio:{0}\n",success_f/4000))
                fo.write(str.format("edge_ratio:{0}\n",success_e/2000))
                fo.write(str.format("node_ratio:{0}\n",success_n/2000))
                fo.write(str.format("intense_ratio:{0}\n",success_i/2000))
                fo.write("\n")
                fo.close()
                if(success/2000>0.67):
                    save=saver.save(sess,GOOD_DIR+"ep_"+str(epoch)+"_acc_"+str(success/2000)+".ckpt")            



                    
                
            ##tf summary need to be done







def worker(worker_index,G,wmin,wmax,network_parameter_queue, exp_queue):
    # assert len(network_parameter_queue) == NUM_AGENT
    # assert len(exp_queue) == NUM_AGENT
    np.random.seed(RANDOM_SEED+worker_index)
    step=0
    env_graph=copy.deepcopy(G)
    env = en.Environment(str.format("Environment_{0}",worker_index), SNODE_SIZE,1000000,imported_graph=[env_graph,node_attr,wmin,wmax],link_embedding_type="hybrid")
    pickle.dump(env, open(str.format('env_worker_{0}.var',worker_index), 'wb'))
    with tf.Session() as sess,open(LOG_FILE + 'worker' + str(worker_index), 'w') as log_file:
        actor = ag.ActorNetwork(sess, str.format("actor_worker_{0}",worker_index), INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS,laplacian=LAPLACIAN_TENSOR)
        critic = ag.CriticNetwork(sess, str.format("actor_worker_{0}",worker_index), INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS,laplacian=LAPLACIAN_TENSOR)
        actor_parameters, critic_parameters = network_parameter_queue.get()
        actor.set_network_params(actor_parameters)
        critic.set_network_params(critic_parameters)
        #print("worker network parameter:", actor_parameters)

        state = env.get_state()
        # action
        snode_batch = []
        vnode_batch=[]
        a_batch = []
        r_batch = []
        
        sess.graph.finalize()
        ass=0
        while (True):
            start = time.clock()
            s,v = env.get_state()
            snode_batch.append(s)
            vnode_batch.append(v)
            env.snode_state=s
            env.vnode_state=v
            action_prob = actor.predict(s, v)
            out_s,out_v,out_b=actor.out_debug(s,v)
            print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
            print("out_s:",out_s)
            print("out_v:",out_v)
            print("out_b:",out_b)
            action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[2]["attributes"],sess)
            is_terminal, failure, reward = env.perform_action(action_pick)
            '''if(action_pick!=0):
                reward=-100
            else:
                reward=100'''
            print("action:",action_pick)
            ass+=1
            a_batch.append(action_one_hot)
            r_batch.append(reward)

            if(is_terminal!=0):
                ass=0
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
            print(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start))
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

#np.random.seed(RANDOM_SEED)
network_parameter_queue=[]
exp_queue=[]
for i in range(NUM_AGENT):
    network_parameter_queue.append(mp.Queue(1))
    exp_queue.append(mp.Queue(1))

coordinator=mp.Process(target=master,args=(network_parameter_queue,exp_queue))
coordinator.start()
workers=[]
for i in range(NUM_AGENT):
    workers.append(mp.Process(target=worker,args=(i,G,wmin,wmax,network_parameter_queue[i],exp_queue[i])))

for i in range(NUM_AGENT):
    workers[i].start()

coordinator.join()



    # now = time.clock()
    # print(now - start)
    # start = time.clock()
    # for i in range(200):
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata=tf.RunMetadata()
    #     s, v = env.get_state()
    #     td=cc.predict(s,v)
    #     action=aa.predict(s,v)
    #     print("the action prob is:",action)
    #     action_one_hot,action_pick=aa.pick_action(action,env.substrate_network.attribute_list[4]["attributes"])
    #     # print("this is training feed:", action_pick, td)
    #     is_terminal,failure,reward=env.perform_action(action_pick)
    #     print(str("this is after perform action:"),is_terminal,failure,reward)
    #     # print()
    #     s_,v_=env.get_state()
    #     td_=cc.predict(s_,v_)
    #     print(str("td initialized from critic:"), td)
    #     print(str("td acted from next critic:"), td_)
    #     td_error=td_target(td_,reward)
    #
    #     new_td=cc.get_td(s,v,td_error)
    #     print(str("td created from reward:"),new_td)
    #     c_optimize,c_loss_summary,c_summary=cc.train(s, v, td_error)
    #     #a=aa.train(s,v,LAPLACIAN_TENSOR,action_one_hot,td)
    #     a_objective,a_entropy,a_entropy_summary,a_optimize,a_summary=aa.train(s,v,action_one_hot,new_td)
    #     print(str.format("actor objective:{0},{1}",a_objective,a_entropy))
    #     now = time.clock()
    #     print(now - start)
    #     start = time.clock()
    #     ENTROPY_WEIGHT=ENTROPY_WEIGHT/1.01
    #     print("EntropyWeight:%.6f" % ENTROPY_WEIGHT)
    #     print(now - start)
    #     start = time.clock()
    #     ENTROPY_WEIGHT=ENTROPY_WEIGHT/1.01
    #     print("EntropyWeight:%.6f" % ENTROPY_WEIGHT)
    #     train_writer.add_run_metadata(run_metadata,'step%03d'%i)
    #     train_writer.add_summary(c_summary,i)
