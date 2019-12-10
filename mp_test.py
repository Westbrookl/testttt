import multiprocessing as mp
import random as rd
import numpy as np

NUM_AGENT = 40

def master(param_queue,exp_queue):
    a=rd.random()
    b=rd.random()
    for i in range(NUM_AGENT):
        param_queue[i].put([a,b])
    print(a,b)
    all=0
    for i in range(NUM_AGENT):
        i,sum,zero=exp_queue[i].get()
        all+=sum
        print(str.format("get from worker{0}:{1}",i,all))



def worker(param_queue,exp_queue,i):
    a,b=param_queue.get()
    print(str.format("worker:{0} adder:{1},{2}", i, a,b))
    factor=rd.random()
    suma=(a+b)*factor
    exp_queue.put([i,suma,np.zeros(10)])
    print(str.format("worker:{0} sum:{1}",i,suma))



def main():
    param_queue = []
    exp_queue = []
    for i in range(NUM_AGENT):
        param_queue.append(mp.Queue(1))
        exp_queue.append(mp.Queue(1))
    for j in range(1000):
        coord=mp.Process(target=master,args=(param_queue,exp_queue))
        coord.start()
        agent=[]
        for i in range(NUM_AGENT):
            agent.append(mp.Process(target=worker,args=(param_queue[i],exp_queue[i],i)))
        for i in range(NUM_AGENT):
            agent[i].start()
        coord.join()

if __name__ == '__main__':
    main()
