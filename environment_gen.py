import numpy as np
import network_topology_gen as nt
import heapq as hp
import tensorflow as tf
import copy
from scipy.stats import poisson


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



class Environment(object):
    def __init__(self, name, substrate_node_size, time_to_live,rate=0.04, link_embedding_type="shortest",phase="Training",testing_size=2000,imported_network=None,imported_graph=None,VNR_type="mini_v"):
        self.name = name
        #self.active_VNR_list = hp.heapify
        self.time_to_live = time_to_live
        self.time = 0
        self.rate=rate
        self.VNR_type=VNR_type
        self.substrate_node_size=substrate_node_size
        if imported_network is None:
            self.substrate_network = nt.SubstrateNetwork(name + "_SubstrateNetwork", substrate_node_size,link_probability=0.1, weights="normal",imported_graph=imported_graph)
        else:
            self.substrate_network=imported_network
        self.sweight_max=self.substrate_network.weight_max
        self.snode_max=self.substrate_network.snode_max
        self.VNR_generator = VNRGenerator(name + "_VNRGenerator",type=self.VNR_type)
        self.link_embedding_type = link_embedding_type
        self.phase=phase
        self.testing_size=testing_size

        self.pending_VNR_list = list()
        self.testing_VNR_list = list()
        self.pending_node_list = list()
        self.current_assigned_cpu = list()
        self.current_assigned_bandwidth = list()
        self.assigned_VNR=None
        self.assigned_VNR_list=list()
        #self.generate_VNR_list(self.testing_size)

        self.reward_eligibility_trace=np.ones([self.substrate_node_size])
        self.eps=0.001
        self.discount=0.99
        self.VNR_counter = 0
        self.success_embeddings = 0
        self.LARGE_PENALTY = -100
        self.LARGE_REWARD = 100
        

        self.total_cost=0
        self.total_reward=0
        self.total_cost_backup=0
        self.total_reward_backup=0

        self.substrate_network_backup =copy.deepcopy( self.substrate_network)
        self.snode_state_backup = self.substrate_network.attribute_list

        self.current_VNR = self.fetch_new_VNR()
        self.snode_state, self.vnode_state = self.get_state()



    def get_state(self):
        snode_state = self.substrate_network.attribute_list
        snode_list=list()
        for i in range(len(snode_state)):
            snode_list.append(snode_state[i]["attributes"])
        snode_features=np.stack(snode_list)
        index = self.pending_node_list[0][0]
        vnode_feature_size = len(self.current_VNR.attribute_list)
        vnode_state = np.zeros([vnode_feature_size+1])
        for i in range(vnode_feature_size):
            vnode_state[i] = self.current_VNR.attribute_list[i]['attributes'][index]
        vnode_state[vnode_feature_size]=(len(self.pending_node_list)/self.current_VNR.node_size)
        return snode_features, vnode_state

    def generate_VNR(self,name, lifetime=1000, node_size=5, link_probability=0.5, weights="mini_v", graph_type=0,
                     imported_graph=None):
        start = self.time
        weights=self.VNR_type
        node_size=np.random.randint(low=2,high=10)
        #print("lifetime:",lifetime)
        self.assigned_VNR=AssignedVNR(self.VNR_counter,self.time,lifetime)
        if self.VNR_generator is None:
            print("VNR Generator has not been initialized")
            return 0
        else:
            VNR = self.VNR_generator.generate_VNR(name, start, lifetime,self.snode_max,self.sweight_max, node_size, link_probability,
                                                  weights, graph_type, imported_graph)
            #self.pending_VNR_list.append(VNR)
            self.VNR_counter += 1
            print(str("generated:"), self.VNR_counter)
            return VNR

    def generate_VNR_list(self, size):
        time=1
        while(len(self.testing_VNR_list)<size):
            index=random_gen(self.rate)
            while(index>0):
                index-=1
                name="aaa"
                start=time
                lifetime=start+np.random.uniform(250,750)
                node_size=np.random.randint(low=2,high=10)
                link_probability=0.5
                weights=self.VNR_type
                graph_type=0
                imported_graph=None
                VNR=self.VNR_generator.generate_VNR(name, start, lifetime,self.snode_max,self.sweight_max, node_size, link_probability,
                                                  weights, graph_type, imported_graph)
                self.testing_VNR_list.append(VNR)
            time+=1
        return self.testing_VNR_list
            
        


    def fetch_new_VNR(self):
        #if len(self.pending_VNR_list) == 0:
        if (self.phase=="Training") or len(self.testing_VNR_list)==0:
            current_VNR = self.generate_VNR(self.name + "_VNR")
        else:
            current_VNR = copy.deepcopy(self.testing_VNR_list[0])
            self.time=current_VNR.start
            self.assigned_VNR=AssignedVNR(self.VNR_counter,self.time,current_VNR.lifetime)
            del self.testing_VNR_list[0]
        self.pending_VNR_list.append(current_VNR)
            # print("no VNRs to embed")
            # return 0
            # else:
            #self.current_VNR = self.pending_VNR_list.pop(0)
        self.pending_node_list = current_VNR.node_rank("cpu_in_use")
        self.substrate_network.copy_substrate_network()
        return current_VNR

    def clock(self):
        self.time += 1

    def backup_env(self):
        self.substrate_network_backup = copy.deepcopy(self.substrate_network)
        self.total_cost_backup=self.total_cost
        self.total_reward_backup=self.total_reward

    def restore_env(self):
        self.substrate_network = self.substrate_network_backup
        self.total_cost_backup=self.total_cost
        self.total_reward_backup=self.total_reward
        self.backup_env()

    def clear_partial_embedding(self):
        self.assigned_VNR=None
        self.current_VNR = self.fetch_new_VNR()
        self.current_assigned_bandwidth = list()
        self.current_assigned_cpu = list()
        self.substrate_network.temporary_embedding_result = np.zeros([self.substrate_network.node_size],dtype=np.int)
        self.substrate_network.attribute_list[2]["attributes"]=np.zeros([self.substrate_network.node_size],dtype=np.int)
        self.substrate_network_backup.temporary_embedding_result = np.zeros([self.substrate_network.node_size],dtype=np.int)
        self.substrate_network_backup.attribute_list[2]["attributes"]=np.zeros([self.substrate_network.node_size],dtype=np.int)


    def perform_action(self, action):
        """
        
        :param action: 
        :return: 
        """
        reward = 0
        """link_request = self.current_VNR.graph_topology[pending_node][temporary_embedding_result[i]]
        for j in range(len(shortest_path)-1):
            link_capacity=self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j+1]]["weight"]
            if link_request>link_capacity:
                return False"""
        pending_node = self.pending_node_list.pop(0)[0]
        #decay_factor_cpu=(self.substrate_network.attribute_list[2]["attributes"][action]-self.substrate_network.attribute_list[0]["attributes"][action])/self.substrate_network.attribute_list[2]["attributes"][action]
        #decay_factor_bandwidth=(self.substrate_network.attribute_list[3]["attributes"][action]-self.substrate_network.attribute_list[1]["attributes"][action])/self.substrate_network.attribute_list[3]["attributes"][action]
        decay_factor_cpu=(self.substrate_network.attribute_list[0]["attributes"][action])/self.substrate_network.capacity_list[0]["attributes"][action]
        decay_factor_bandwidth=(self.substrate_network.attribute_list[1]["attributes"][action])/self.substrate_network.capacity_list[1]["attributes"][action]
        self.reward_eligibility_trace[action]+=1
        self.reward_eligibility_trace*=self.discount
        


        # print("virtual node: {0}".format(pending_node))
        #print("selected action: {0}".format(action))
        # print("current virtual partial embedding:")
        # print("attr1:")
        # print(self.substrate_network.attribute_list[0]["attributes"])
        # print("attr2:")
        # print(self.substrate_network.attribute_list[1]["attributes"])
        # print("attr3:")
        # print(self.substrate_network.attribute_list[2]["attributes"])
        # print("attr4:")
        # print(self.substrate_network.attribute_list[3]["attributes"])
        # print("attr5:")
        # print(self.substrate_network.attribute_list[4]["attributes"])
        node_availability = self.check_node_embedding_availability(action)
        if (node_availability == False):
            self.clear_partial_embedding()
            self.restore_env()
            return 1,"node", self.LARGE_PENALTY/(len(self.assigned_VNR_list)+1)
        temporary_embedding_result = self.substrate_network.temporary_embedding_result
        for i in range(len(temporary_embedding_result)):
            if (temporary_embedding_result[i] != 0 )and (i!=action) and (self.current_VNR.graph_topology.has_edge(pending_node,temporary_embedding_result[i])):
                if self.link_embedding_type == "shortest":
                    shortest_path = self.substrate_network.shortest_paths[action][1][i]
                    res = self.check_link_embedding_availability(shortest_path, pending_node,
                                                                 temporary_embedding_result[i])
                    if (res == False):
                        self.clear_partial_embedding()
                        self.restore_env()
                        return 1,"link", self.LARGE_PENALTY/(len(self.assigned_VNR_list)+1)
                    else:
                        reward+=res
                elif self.link_embedding_type == "hybrid":
                    shortest_path = self.substrate_network.shortest_paths[action][1][i]
                    res = self.check_link_embedding_availability(shortest_path, pending_node,
                                                                 temporary_embedding_result[i])
                    if (res == False):
                        shortest_path = self.substrate_network.simple_paths(action, i)
                        success = 0
                        for l in range(len(shortest_path)):
                            res = self.check_link_embedding_availability(shortest_path[l], pending_node,
                                                                         temporary_embedding_result[i])
                            if (res != False):
                                success = 1
                                reward += res
                                break
                        if (success == 0):
                            self.clear_partial_embedding()
                            self.restore_env()
                            return 1, "link", self.LARGE_PENALTY/(len(self.assigned_VNR_list)+1)
                    else:
                        reward += res
                else:
                    shortest_path = self.substrate_network.simple_paths(action, i)
                    success = 0
                    for l in range(len(shortest_path)):
                        res = self.check_link_embedding_availability(shortest_path[l], pending_node,
                                                                     temporary_embedding_result[i])
                        if (res != False):
                            success = 1
                            reward+=res
                            break
                    if (success == 0):
                        self.clear_partial_embedding()
                        self.restore_env()
                        return 1,"link", self.LARGE_PENALTY/(len(self.assigned_VNR_list)+1)
        self.substrate_network.temporary_embedding_result[action] = pending_node
        for i in range(len(self.substrate_network.attribute_list)):
            if(self.substrate_network.attribute_list[i]["name"]=="current_embedding"):
                self.substrate_network.attribute_list[i]["attributes"][action]=1
        if (len(self.pending_node_list) == 0):
            self.success_embeddings += 1
            print(str.format(("success:{0},generated:{1}"),self.success_embeddings,self.VNR_counter))
            self.assign_resource()
            self.assigned_VNR.assigned_node=self.current_assigned_cpu
            self.assigned_VNR.assigned_link=self.current_assigned_bandwidth
            index=len(self.assigned_VNR_list)
            for i in range (len(self.assigned_VNR_list)):
                if(self.assigned_VNR_list[i].lifetime>self.assigned_VNR.lifetime):
                    index=i
                    break
            self.assigned_VNR_list.insert(index,self.assigned_VNR)
            self.clear_partial_embedding()
            return 2, "success",self.LARGE_REWARD*len(self.assigned_VNR_list)*decay_factor_cpu*decay_factor_bandwidth/self.reward_eligibility_trace[action]
        else:
            return 0,"none", len(self.assigned_VNR_list)*self.LARGE_REWARD*reward/(1+len(self.pending_node_list))/self.reward_eligibility_trace[action]

    def check_node_embedding_availability(self, action):
        max =None
        in_use = None
        node_request = self.vnode_state[0]
        for i in range(len(self.substrate_network.attribute_list)):
            if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                in_use = self.substrate_network.attribute_list[i]["attributes"]
        if (in_use[action] < node_request):
            return False
        else:
            for i in range(len(self.substrate_network.attribute_list)):
                if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                    self.substrate_network.attribute_list[i]["attributes"][action] -= node_request
                    self.current_assigned_cpu.append([action,node_request])
            return True

    def check_link_embedding_availability(self, shortest_path, virtual_source_node, virtual_target_node):
        """
        reward definition to be optimized
        :param shortest_path: 
        :param virtual_source_node: 
        :param virtual_target_node: 
        :return: 
        """
        cost = 0
        reward = 0
        # print(str("source:"),virtual_source_node)
        # print(str("target:"),virtual_target_node)
        # print(str("link embedded:"),self.current_VNR.graph_topology[virtual_source_node][virtual_target_node]["weight"])
        link_request = self.current_VNR.graph_topology[virtual_source_node][virtual_target_node]["weight"]
        for i in range(len(self.substrate_network.attribute_list)):
            if(self.substrate_network.attribute_list[i]["name"]=="bandwidth_in_use"):
                bandwidth_used=self.substrate_network.attribute_list[i]["attributes"]
        for j in range(len(shortest_path) - 1):
            link_capacity = \
                self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j + 1]][
                    "weight"]
            if link_request > link_capacity:
                return False
        reward+=link_request
        for j in range(len(shortest_path) - 1):
            cost+=link_request
            link_capacity = \
                self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j + 1]][
                    "weight"]
            new_link_weight = link_capacity - link_request
            # self.current_assigned_bandwidth[shortest_path[j]] += link_request
            # self.current_assigned_bandwidth[shortest_path[j + 1]] += link_request
            normalized_bandwidth=link_request/self.substrate_network.max_bandwidth
            bandwidth_used[shortest_path[j]] -= normalized_bandwidth
            bandwidth_used[shortest_path[j+1]] -= normalized_bandwidth
            self.substrate_network.graph_topology.add_edge(shortest_path[j],
                                                           shortest_path[j + 1],
                                                           weight=new_link_weight)
            self.current_assigned_bandwidth.append([shortest_path[j],shortest_path[j+1],link_request,normalized_bandwidth])
        for i in range(len(self.substrate_network.attribute_list)):
            if(self.substrate_network.attribute_list[i]["name"]=="bandwidth_in_use"):
                self.substrate_network.attribute_list[i]["attributes"]=bandwidth_used
                #print(str("max node bandwidth:"),self.substrate_network.max_bandwidth)
        return reward/cost

    def assign_resource(self):
        # for i in range(len(self.substrate_network.temporary_embedding_result)):
        #     if (self.substrate_network.temporary_embedding_result[i] != 0):
        #         index = self.substrate_network.temporary_embedding_result[i]
        #         self.current_assigned_cpu[i] = self.current_VNR.attribute_list[0]["attributes"][index]
        # assigned_resource = {"cpu_in_use": self.current_assigned_cpu,
        #                      "bandwidth_in_use": self.current_assigned_bandwidth,
        #                      "expire_time": self.time + self.current_VNR.lifetime}
        # for i in range(len(self.substrate_network.attribute_list)):
        #     if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
        #         self.substrate_network.attribute_list[i]["attributes"] += self.current_assigned_cpu
        #     if (self.substrate_network.attribute_list[i]["name"] == "cpu_remaining"):
        #         self.substrate_network.attribute_list[i]["attributes"] -= self.current_assigned_cpu
        #     if (self.substrate_network.attribute_list[i]["name"] == "bandwidth_in_use"):
        #         self.substrate_network.attribute_list[i]["attributes"] += self.current_assigned_bandwidth
        # '''add the assigned vnr to the heap using tuples'''
        return 0

    def release_resource_doing_wrong(self,time):
        while(len(self.assigned_VNR_list)!=0 and self.assigned_VNR_list[0].lifetime<time):
            """getting resource with the earliest expiring VNR and release them on substrate network based on the heap tuples"""
            for i in range(len(self.substrate_network.attribute_list)):
                if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                    in_use = self.substrate_network.attribute_list[i]["attributes"]
            for i in range(len(self.assigned_VNR_list[0].assigned_node)):
                node_num,node_res=self.assigned_VNR_list[0].assigned_node[i]
                in_use[node_num]-=node_res
            for i in range(len(self.substrate_network.attribute_list)):
                if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                    self.substrate_network.attribute_list[i]["attributes"]=in_use
            for i in range(len(self.assigned_VNR_list[0].assigned_link)):
                start,end,res=self.assigned_VNR_list[0].assigned_link[i]
                link_capacity = self.substrate_network.graph_topology[start][end]["weight"]+res
                #print("link_capacity:",link_capacity)
                self.substrate_network.graph_topology.add_edge(start,end,weight=link_capacity)
            #print("VNR %s has been removed in %s:"%(self.assigned_VNR_list[0].VNR_num,time))
            self.assigned_VNR_list.remove(self.assigned_VNR_list[0])
        return 0



    def release_resource_doing_wrong_2(self,time):
        while(len(self.assigned_VNR_list)!=0):
            """getting resource with the earliest expiring VNR and release them on substrate network based on the heap tuples"""
            in_use = self.substrate_network.attribute_list[0]["attributes"]
            for i in range(len(self.assigned_VNR_list[0].assigned_node)):
                node_num,node_res=self.assigned_VNR_list[0].assigned_node[i]
                in_use[node_num]-=node_res
            self.substrate_network.attribute_list[0]["attributes"]=in_use
            bandwidth_in_use=self.substrate_network.attribute_list[1]["attributes"]
            for i in range(len(self.assigned_VNR_list[0].assigned_link)):
                start,end,res=self.assigned_VNR_list[0].assigned_link[i]
                link_capacity = self.substrate_network.graph_topology[start][end]["weight"]+res
                bandwidth_in_use[start]-=res/self.substrate_network.max_bandwidth
                bandwidth_in_use[end]-=res/self.substrate_network.max_bandwidth
                #print("link_capacity:",link_capacity)
                self.substrate_network.graph_topology.add_edge(start,end,weight=link_capacity)
            self.substrate_network.attribute_list[1]["attributes"]=bandwidth_in_use
            #print("VNR %s has been removed in %s:"%(self.assigned_VNR_list[0].VNR_num,time))
            self.assigned_VNR_list.remove(self.assigned_VNR_list[0])
        return 0

    def release_resource(self,time):
        while(len(self.assigned_VNR_list)!=0 and self.assigned_VNR_list[0].lifetime<time):
            """getting resource with the earliest expiring VNR and release them on substrate network based on the heap tuples"""
            for i in range(len(self.assigned_VNR_list[0].assigned_node)):
                node_num,node_res=self.assigned_VNR_list[0].assigned_node[i]
                self.substrate_network.attribute_list[0]["attributes"][node_num]+=node_res
            for i in range(len(self.assigned_VNR_list[0].assigned_link)):
                start,end,res,normres=self.assigned_VNR_list[0].assigned_link[i]
                link_capacity = self.substrate_network.graph_topology[start][end]["weight"]+res
                self.substrate_network.attribute_list[1]["attributes"][start]+=normres
                self.substrate_network.attribute_list[1]["attributes"][end]+=normres
                #print("link_capacity:",link_capacity)
                self.substrate_network.graph_topology.add_edge(start,end,weight=link_capacity)
            #print("VNR %s has been removed in %s:"%(self.assigned_VNR_list[0].VNR_num,time))
            self.assigned_VNR_list.remove(self.assigned_VNR_list[0])
        self.backup_env()
        return 0

    # def choose_action(self,action_prob,phase="Training"):
    #     a=np.arange(0,self.substrate_node_size)
    #     if(phase=="Training"):
    #         return np.random.choice(a, p=action_prob.eval(session=self.))
    #     else:
    #         return np.max(action_prob).__index__()

    def reward(self, action_result):
        """
        reward calculation in the environment
        possible reward function:
        
        one step or a whole trajectory?
        :return: 
        """

        return 0

    """
    stack trajectory(s,a,r,s')
    apply them to networks
    using td (critic) and policy gradient (actor)
    
    if all stacked are successful, update
    if failed...?
    1.update all using a huge negative reward
    2.update step-by-step, only affect the last failed move
    """


class VNRGenerator(object):
    def __init__(self, name, type="mini_v", rate=1):
        self.name = name
        self.type = type
        self.rate = rate

    def generate_VNR(self, name, start, lifetime,smax,wmax,node_size=5, link_probability=0.5, weights="mini_v", graph_type=0,
                     imported_graph=None):
        weights=self.type
        VNR = nt.VirtualNetworkRequest(name, start, lifetime,smax,wmax, node_size, link_probability, weights, graph_type,
                                       imported_graph)
        return VNR

    def auto_generate_VNR(self, time=1000, rate=1):
        return 0

class AssignedVNR(object):
    def __init__(self,num,start,lifetime):
        self.start=start
        self.VNR_num=num
        if lifetime==1000:
            #self.lifetime=start+np.random.uniform(30,50)
            #self.lifetime=start+np.random.uniform(250,750)
            self.lifetime=start+np.random.uniform(2000,3000)
        else:
            self.lifetime=lifetime
        #print("VNR num %s created from %s to %s:"%(self.VNR_num, self.start,self.lifetime))
        self.assigned_node=list()
        self.assigned_link=list()

    def add_node(self,node_num,node_request):
        self.assigned_node.append([node_num,node_request])

    def add_link(self,link_start,link_end,link_request):
        self.assigned_link.append([link_start,link_end,link_request])




# aa=Environment("env",50,10000)
#
# prob=np.random.uniform(1,2,50)
# bb=tf.nn.softmax(prob)
# ss=aa.choose_action(bb)
# gg=aa.choose_action(bb,phase="Testing")
# print(bb)
# print(ss)
# print(gg)

