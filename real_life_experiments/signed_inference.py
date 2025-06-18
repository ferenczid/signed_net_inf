import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('ticks')
import sklearn
from sklearn.metrics import roc_curve
import scipy
import os
import multiprocessing
from multiprocessing import Pool
import random

#Inference algorithms

def likelihood_fast(old_val,new_val,leng,obs,partition,index_1,index_2,rate):
    curr=0
    if partition[index_1]!=partition[index_2]:
        return 0
    else:
        return obs*(np.log(rate[new_val])-np.log(rate[old_val]))+(leng-obs)*(np.log(1-rate[new_val])-np.log(1-rate[old_val]))
    for i in range(len(X)):
        for j in range(len(X)):
            if partition[i]==partition[j]:
                curr+=np.log(scipy.special.comb(n,X[i][j]))+np.log(rates[A[i][j]])*X[i][j]+(n-X[i][j])*np.log(1-rates[A[i][j]])
    #print(curr)
    return curr


def likelihood_fast_multiple_obs_mtx_change(new_val,old_val,rates,X_s,lens,partitions,index_1,index_2):
    curr_sum=0
    #print(A,rates,X_s,lens)
    for k in range(len(X_s)):
        curr_sum+=likelihood_fast(old_val,new_val,lens[k],X_s[k][index_1][index_2],partitions[k],index_1,index_2,rates[k])
       
    return curr_sum
def likelihood_fast_prob_change(A,X,lens,rate_old,rate_new,changed_key,partitions,obs_dict,comp_obs_dict):



    #print((np.log(rate_new[changed_key])),np.log(rate_old[changed_key]),sum_of_obs,rate_old,rate_new,changed_key,sum_of_obs)
    return (np.log(rate_new[changed_key])-np.log(rate_old[changed_key]))*(obs_dict[changed_key])+(comp_obs_dict[changed_key])*(np.log(1-rate_new[changed_key])-np.log(1-rate_old[changed_key]))


def balanc(N):
    """
    Parameters
    ----------
    N : ndarray
        Negative adjacency matrix.
    Returns
    --------
    y : augmented Fiedler Balance of a graph

    Notes
    -----
    Takes a graph represented by its adjacency matrix as its input, and returns its augmented Fiedler balance.
    The augmented Fiedler balance is the graphs Fiedler value, if the graph is connected. 
    If the graph is disconnected, it will return 4/number of nodes in N*(sum of diamaters of the connected components+ number of connected component)
    """

    
    nums,parts=scipy.sparse.csgraph.connected_components(N)
    if len(N)<2:
        return 1
        #return 1
    #print('komponensek: ',Anums)
    if nums==1:
        x=np.sort(scipy.linalg.eigh(scipy.sparse.csgraph.laplacian(N),eigvals_only=True))
        return x[1]
    diams=0
    for r in set(parts):
        #print(r)
        mask=parts==r
       # print(mask)
        curr=get_masked_matrix(N,mask)
        if len(curr)<2:
            diams+=1
        else:
            temp_graph=nx.from_numpy_array(curr)
            diams+=nx.diameter(temp_graph)
        #print(len(curr))
        num_groups=r
    #print (parts)
    #print('átmérők: ',diams)
    return 4/(len(N)*diams+len(N)*num_groups)


def balance(N,P):
    #print(P,'<-P, N->',N)
    """
    Parameters
    ----------
    N : ndarray
    Negative adjacency matrix.
    P : ndarray
    Positive adjacency matrix.
    Returns
    --------
    y : float
        augmented Fiedler value of signed networks
    Notes
    ---------
    Takes a signed network represented by the adjacency matrix of its positive and negative edges as an input. 
    The returned value will be the augmented Fiedler balance of a signed network, which is defined to be the Fiedler value of the positive edges matrix/the augmented Fiedler value of the negative edges matrix.
    It the positive matrix does not have any edges (i.e. disconnected), it is zero.
    If the graphs only consist of a single node, it is 0.2 ( which was just used as a random number)
    """
    if len(P)==0:
     #   print('empty partition detected')
        return 0
    if len(P)==1 and len(N)==1:
       # print("isolated vertex detected")
        return 1
    x=np.sort(scipy.linalg.eigh(scipy.sparse.csgraph.laplacian(P),eigvals_only=True))
    
    if x[1]<0.000000000000001:
        #print(' A nem öf.')
        return 0
    
    #print(' a metrika lett visszaadva',x[1],"balanc: ",balanc(N),"all-in-all" , np.abs(x[1])/np.abs(balanc(N)))
    return 1


def fact(n):
    """
    Parameters
    ----------
    n: float or int
        number
    Returns 
    -------
    y: int
        factorial of n
    Notes
    -------
    Takes a number n as an input, and returns (n!).
    """
    p=1
    if float(n)-float(int(n))!=0:
        return 0
    else :
        n=int(n)
    if n ==1:
        return 1
    for i in range(1,n+1):
        p*=i
    return float(p)

def get_masked_matrices(N, P, node_mask):
    """Get masked adjacency matrices.
    
    Parameters
    ----------
    N : ndarray
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    node_mask : ndarray
        Boolean array of nodes to mask.
    """
    return N[node_mask,:][:,node_mask], P[node_mask,:][:,node_mask]
def get_masked_matrix(P, node_mask):
    """Get masked adjacency matrix.
    
    Parameters
    ----------
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    node_mask : ndarray
        Boolean array of nodes to mask.
    """
    return P[node_mask,:][:,node_mask]
    g = nx.generators.random_graphs.erdos_renyi_graph(20, 0.3)


plt.show()
def get_overall_balance(N, P, node_partition, alpha=1.05):
    """

    Parameters
    ----------
    N : ndarray
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    partition : ndarray
                  array encoding partitioning of network
    alpha: float
            decay paratmeter
    Returns 
    --------
    y: float
        value of Balance in network
    Notes
    -------
    Takes a signed network represented as 2 adjacency matrices, corresponding to its positive and negative edges, and a partitioning of the nodes, and number as an input.
    It returns the total balance of the network and the partitioning, by iterating through the partitions, and creating subnetworks to each partition. Then calculating the balance of said subnetworks, and multiplying them in the end.
    """
    total_balance = 1
    for r in set(node_partition):
        node_mask = node_partition == r
        #print(' balance if node_part==',r,balance(*get_masked_matrices(N, P, node_mask), alpha))
        N_small=get_masked_matrix(N, node_mask)
        P_small=get_masked_matrix(P, node_mask)
        #print(total_balance,'aktuális érték')
        #print(balance(N_small,P_small),"modifier")
        total_balance *= balance(N_small,P_small)
        #print(total_balance,"current balance ")
    #print('the total balance is ',total_balance)
    return total_balance

def partition_mcmc(N,P,k,n_iter=1000):

    """
    Parameters
    ----------
    N : ndarray
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    k : int
        max number of groups
    n_iter: int
            how many steps MCMC should make
    Returns
    --------
    part: ndarray
          partitioning of network nodes. Distribution of the returned value is proportional to the augmented Fiedler value of the network partition
    Notes
    -------
    Takes a signed network represented by its positive and negative edges network, the maximum number of groups, and a number on how many iterations should be done. Returns a partitioning of the nodes sampled from the distribution proportional to the augmented Fiedler balance of the partition. We use MCMC sampling, which goes on for n_iter steps, and return the corresponding partitions.
    """
    #add prior for choice of num of groups, 
    num_nodes=N.shape[0]
    count=0
    good_part=False
    init_count=0
    while good_part==False:
        init_count+=1
        if init_count>1000:
            return 'fail'
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        part=np.array([np.random.choice(2) for _ in range(num_nodes)])
        if get_overall_balance(N,P,part)>0.00000001:
            good_part=True
        else: 
            #print('failed to init')
            count
    #print('inti success: ', get_overall_balance(N,P,part))

            
    #part=np.array([0 for _ in range(num_nodes)])
    for j in range(n_iter):
        #print('we are on step: ',j)
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        i=np.random.choice(num_nodes)
        r=part [i]
        l=min(max(part)+2,k)
        #l=int(l+1)
        #print('------------------------------',l,'--------------')
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        s=np.random.choice(  [x for x in range(l) if x!=r  ])
        prop_part=np.copy(part)
        prop_part[i]=s
        new_num=get_overall_balance(N,P,prop_part)
       # print(new_num,get_overall_balance(N,P,part))
        if new_num>0.00000001:
            #print('proposed: ',np.log(get_overall_balance(N,P,prop_part)),'current',np.log(get_overall_balance(N,P,part)))
            L=min(np.log(new_num)-np.log(get_overall_balance(N,P,part)+prior_part(part)-prior_part(prop_part)),0)
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            u=np.log(np.random.uniform(0,1))
            if L >u:
                #print(L,"csere",u)
                part=prop_part
                count+=1
                if new_num<0.000001:
                    break
                    print("itt a baj")
        
    print(count, 'darab csere történt')
    return part



def fact(n):
    """
    Parameters
    ----------
    n: float or int
        number
    Returns 
    -------
    y: int
        factorial of n
    Notes
    -------
    Takes a number n as an input, and returns (n!).
    """
    p=1
    if float(n)-float(int(n))!=0:
        return 0
    else :
        n=int(n)
    if n ==1:
        return 1
    for i in range(1,n+1):
        p*=i
    return float(p)

def prior_part(partition):
    """
    Parameters
    ----------
    partition: ndarray
                partitioning of a network
    Returns
    -------
    y:     ndarray
           probability of partition according to a uniform prior
    """
    grp_sizes=[]
    #print(set(partition))
    for i in set(partition):
        counter=0
        #print(i)
        for j in partition:
            if i==j:
                counter+=1
                #print(i,j)
        grp_sizes.append(counter)
    #print(grp_sizes)
    n=sum(grp_sizes)
    k=len(grp_sizes)
    #print(np.log(1/(scipy.special.comb(n-1,k-1)*fact(n)*np.prod(grp_sizes))))
    return (-1*(np.log(scipy.special.comb(n-1,k-1))+np.log(fact(n))+np.log(np.prod(grp_sizes))))




def genrete_binomial_measurements(partition, A, m, rates={-1:0.01,0:0.2,1:0.9},q_rate=0.05):
    """ Generate graph measurements.
    
    Parameters
    ----------
    partition : ndarray
        Node partition.
    A : ndarray
        Adjacency matrix.
    m : int
        Number of measurements.
    rates : dict
        Dictionary of measurement rates. 3 values: -1 for negative ties, 0 for no ties, and 1 for positive ties.
    Returns
    -------
    X: ndarray
       Observation matrix. X[i][j]~Bin(m rates[A[i][j])), for all i!=j
    """
    X=np.zeros((len(A),len(A)))
    for i in range(len(X)):
        for j in range(i):
            
            #print(A[i][j])
            if partition[i]==partition[j]:
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                X[i][j]=np.random.binomial(m,rates[int(A[i][j])])
                X[j][i]=X[i][j]
            else:
                X[i][j]=np.random.binomial(m,q_rate)
                X[j][i]=X[i][j]
   # print(X)
   
    
    return X


def get_masked_matrix(P, node_mask):
    """Get masked adjacency matrix.
    
    Parameters
    ----------
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    node_mask : ndarray
        Boolean array of nodes to mask.
    """
    return P[node_mask,:][:,node_mask]

def mask_matrices(matrix, node_index, part_new,part_old):
    A=(matrix>0).astype(int)
    node_mask_new=[part_new==part_new[node_index]][0]
    node_mask_old=[part_new==part_old[node_index]][0]
    """Get masked adjacency matrices.
    
    Parameters
    ----------
    N : ndarray
        Negative adjacency matrix.
    P : ndarray
        Positive adjacency matrix.
    node_mask : ndarray
        Boolean array of nodes to mask.
    """
    print(node_mask_new)
    print("itt: ",(A[node_mask_new,:][:,node_mask_new]>0).astype(int))
    print("ott: ",(A[node_mask_old,:][:,node_mask_old]>0).astype(int))
    new_sub_mx,old_sub_mx=A[node_mask_new,:][:,node_mask_new],A[node_mask_old,:][:,node_mask_old]
    if len(new_sub_mx)>1:
        x=np.sort(scipy.linalg.eigh(scipy.sparse.csgraph.laplacian(new_sub_mx),eigvals_only=True))
    else:
        x=[1,1]
    if len(old_sub_mx)>1:
        y=np.sort(scipy.linalg.eigh(scipy.sparse.csgraph.laplacian(old_sub_mx),eigvals_only=True))
    else:
        y=[1,1]
    if x[1]>0 and y[1]>0:
        return 1
    else:
        return 0

def prop_group_move(node_index,partition,obs):
    candidates=list(set(partition))
    candidates.append(max(candidates)+1)
    weights={el:1.0 for el in candidates}
    #print(weights)
    #print(weights[2],weights[partition[2]])
    s=sum(obs)
    for i in range(len(obs)):
        weights[partition[i]]+=obs[node_index][i]
    l=list(weights.values())
    l/=sum(l)
    #print(l)
    #print(candidates)
    a=np.random.choice(candidates,p=l)
    return a
    #print(a)

def prop_probs(Part_old,Part_new,node_moved,Obs):
    group_old=len(set(Part_old))
    group_new=len(set(Part_new))
    part_matr_new=create_part_matr_label(Part_new,Part_new[node_moved])
    part_matr_old=create_part_matr_label(Part_old,Part_old[node_moved])
    neighbors_old=np.transpose(part_matr_old)@Obs[node_moved]
    neighbors_new=np.transpose(part_matr_new)@Obs[node_moved]
    #print(group_new,group_old,neighbors_old,neighbors_new)
    if group_new==group_old:
        return -(np.log(neighbors_new+1)-np.log(neighbors_old+1))
    else: 
        observations=np.transpose(np.ones(len(Obs)))@Obs[node_moved]
        return -(np.log(neighbors_new+1)-np.log(neighbors_old+1)+np.log(observations+group_old)-np.log(observations+group_new))
    
"""
Part_curr=partitions[0]

Part_new[3]=3
print(prop_probs(Part_new,Part_curr,3,X_s[0]))
B=True
steps=0
while B==True:
    steps+=1
    if steps>10:
        B=False
    Part_new=Part_curr.copy()
    index_chosen=np.random.choice(len(X_s[0]))
    Part_new[index_chosen]=np.random.choice(list(range(1+len(set(Part_curr)))))
    print(prop_probs(Part_new,Part_curr,3,X_s[0]))
    Part_curr=Part_new.copy()
    

"""
def relabel(l):
    temp=[]
    vals=set(l)
    keys=list(range(len(vals)))
    d=dict(zip(np.sort(list(vals)),keys))
    for i in l:
        temp.append(d[i])
    return temp
print(relabel([3,1,3,0]))
print(relabel([2,222,2,2342342341249]))



def check_mtx_change_list(parts,new_A,old_A,ind_1=None,ind_2=None):
    if old_A[ind_1][ind_2]!=1:
        return True
    for partitions in parts:
        if check_mtx_change(partitions,new_A,old_A,ind_1,ind_2)==False:
            return False
    return True

def check_mtx_change(partitions,new_A,old_A,ind_1=None,ind_2=None):
    if ind_1==None or ind_2==None:
        for i in range(len(new_A)):
            for j in range(i):
                if new_A[i][j]!=old_A[i][j]:
                    ind_1=i
                    ind_2=j
    if ind_1==None or ind_2==None:
        return True
    if partitions[ind_1]==partitions[ind_2]:
        part_label=partitions[ind_1]
    else:
        return True
    if old_A[ind_1][ind_2]==1 and new_A[ind_1][ind_2]!=1:
        parts_filtered=[partitions==part_label]
        #print(parts_filtered)
        

        #parts_filtered[0][ind_1]=False
        #print(parts_filtered)
        curr_mtx=get_masked_matrix(new_A,parts_filtered[0])
        curr_mtx=(curr_mtx>0).astype(int)
        #print(curr_mtx)
        if len(curr_mtx)>1:
            y=np.sort(scipy.linalg.eigh(scipy.sparse.csgraph.laplacian(curr_mtx),eigvals_only=True))
            #print("ez itt: ",y[1], y[1]>1e-30)
            return y[1]>1e-30
            
        else:
            return True
    else:
        return True

def create_part_matr_label(Part,label):
    l=[]
    for i in set(Part):
        temp=np.zeros(len(Part))
        temp=np.array([Part==i]).astype(int)[0]
        if i==label:
            return temp
    return np.zeros(len(Part))
def update_list_fast(Matr,Obs,Part_old,Part_new,leng,node_moved,d,comp_d,q,comp_q):

    if list(Part_old)==list(Part_new):
        return d,q,comp_d,comp_q
    d=d.copy()
    comp_d=comp_d.copy()
    q=q
    comp_q=comp_q
    part_matr_old_g1=create_part_matr_label(Part_old,Part_old[node_moved])
    part_matr_old_g2=create_part_matr_label(Part_old,Part_new[node_moved])
    part_matr_new_g1=create_part_matr_label(Part_new,Part_old[node_moved])
    part_matr_new_g2=create_part_matr_label(Part_new,Part_new[node_moved])
    #print("diag:",part_matr_new_g1-part_matr_old_g1,node_moved,part_matr_new_g2-part_matr_old_g2)
    #print(part_matr_old_g1,print(Obs[node_moved],"33"))
    #print(part_matr_old_g1,part_matr_old_g2,part_matr_new_g1,part_matr_new_g2,"dddd")
    Apos=(Matr[node_moved]>0).astype(int)
    Aneg=(Matr[node_moved]<0).astype(int)
    Azer=(Matr[node_moved]==0).astype(int)
    #Azer[node_moved]=1
    ones=np.ones(len(Matr))
    
    An=Aneg*Obs[node_moved]
    Ap=Apos*Obs[node_moved]
    Az=Azer*Obs[node_moved]
    Aneg_comp=(Aneg*leng-An)
    Azer_comp=(Azer*leng-Az)
    Apos_comp=(Apos*leng-Ap)
    #print(Apos_comp,"Apos_comp")
    #print(Ap)
    #print(An)
    #print(np.transpose(part_matr_old_g1)@(An*Obs[node_moved]),"it")
    #print(part_matr_new_g2)
    valng1_old=np.transpose(part_matr_old_g1)@(An)
    valpg1_old=np.transpose(part_matr_old_g1)@(Ap)
    valzg1_old=np.transpose(part_matr_old_g1)@(Az)
    valng1_old_comp=np.transpose(part_matr_old_g1)@Aneg_comp
    valpg1_old_comp=np.transpose(part_matr_old_g1)@Apos_comp
    valzg1_old_comp=np.transpose(part_matr_old_g1)@Azer_comp
    """
    valng2_old=np.transpose(part_matr_old_g2)@(An)
    valpg2_old=np.transpose(part_matr_old_g2)@(Ap)
    valzg2_old=np.transpose(part_matr_old_g2)@(Az)
    valng1_new=np.transpose(part_matr_new_g1)@(An)
    valpg1_new=np.transpose(part_matr_new_g1)@(Ap)
    valzg1_new=np.transpose(part_matr_new_g1)@(Az)
    valng2_new_comp=leng*np.transpose(part_matr_new_g2)@Aneg_comp
    valpg2_new_comp=leng*np.transpose(part_matr_new_g2)@Apos_comp
    valzg2_new_comp=leng*np.transpose(part_matr_new_g2)@Azer_comp
    """
    valng2_new=np.transpose(part_matr_new_g2)@(An)
    valpg2_new=np.transpose(part_matr_new_g2)@(Ap)
    valzg2_new=np.transpose(part_matr_new_g2)@(Az)
    valng2_new_comp=np.transpose(part_matr_new_g2)@Aneg_comp
    valpg2_new_comp=np.transpose(part_matr_new_g2)@Apos_comp
    valzg2_new_comp=np.transpose(part_matr_new_g2)@Azer_comp
    q+=valng1_old+valpg1_old+valzg1_old-valpg2_new-valzg2_new-valng2_new
    #print(valpg1_new,valpg1_old,valpg2_new,valpg2_old,valzg1_new,valzg1_old)
    d[1]+=-valpg1_old+valpg2_new
    d[0]+=-valzg1_old+valzg2_new
    d[-1]+=-valng1_old+valng2_new
    comp_d[-1]+=valng2_new_comp-valng1_old_comp
    comp_d[0]+=valzg2_new_comp-valzg1_old_comp
    comp_d[1]+=valpg2_new_comp-valpg1_old_comp
    comp_q-=valpg2_new_comp+valng2_new_comp+valzg2_new_comp-valzg1_old_comp-valpg1_old_comp-valng1_old_comp
    #print("itt,",comp_d,"halo",(np.transpose(part_matr_new_g2)@(Aneg*leng)-An))
    #print(d,valpg1_old,valng1_old,valzg1_old,valng1_new)
    return d,q,comp_d,comp_q

def likelihood_partition(A,Obs,probs,leng,q,node_index,partition_new,partition_old,new_part_label=None,old_part_label=None):
    val=0
    if new_part_label==None or old_part_label==None:
        for i in range(len(partition_new)):
            if partition_new[i]!=partition_old[i]:
                new_part_label=partition_new[i]
                old_part_label=partition_old[i]
                #print('i',i)
                break
    if new_part_label==old_part_label:
        #print("identical partitions")
        return -240
    for v in range(len(A)):
        if partition_old[v]==old_part_label:
            val+=Obs[node_index][v]*(np.log(q)-np.log(probs[A[v][node_index]]))+(leng-Obs[node_index][v])*(np.log(1-q)-np.log(1-probs[A[v][node_index]]))
        if partition_new[v]==new_part_label:
            val+=Obs[node_index][v]*(np.log(probs[A[v][node_index]])-np.log(q))+(leng-Obs[node_index][v])*(np.log(1-probs[A[v][node_index]])-np.log(1-q))
        #print(partition_new[v],partition_new[node_index])
    val+=prop_probs(partition_old,partition_new,node_index,Obs)
    return val



def edges_btwn_groups(A,X,part_old,ind_1,ind_2,lab1=None,lab2=None):
    if lab1==None or lab2==None:
        #print("p",part_old)
        lab1=part_old[ind_1]
        lab2=part_old[ind_2]
    l=[]
    c=0
    if lab1==lab2:
        return []
    for i in range(len(A)):
        if part_old[i]==lab1:
            for j in range(len(A)):
                if part_old[j]==lab2:
                    #print(A[i][j])
                    l.append([i,j])
                    c+=1
    return l

def update_agg(matr_par,obs_par,part_par,leng_par,ind_1_par,ind_2_par):
    matr=matr_par
    obs=obs_par
    part=part_par
    leng=leng_par
    ind_1=ind_1_par
    ind_2=ind_2_par
    edges=edges_btwn_groups(matr,obs,part,ind_1,ind_2)
    out={-1:0,1:0,0:0}
    out_comp={-1:0,1:0,0:0}
    #print(len(edges),"élek")
    for e in edges:
        out[matr[e[0]][e[1]]]-=obs[e[0]][e[1]]
        out_comp[matr[e[0]][e[1]]]-=leng-obs[e[0]][e[1]]
    return out,out_comp
def update_agg_lists(matr_par,Obs_par,part_par,lens_par,ind_1_par,ind_2_par):
    temp=[]
    for i in range(len(Obs_par)):
        temp.append(update_agg(matr_par,Obs_par[i],part_par[i],lens_par[i],ind_1_par,ind_2_par))
    #print(temp)
    return temp

def create_part_matr(Part):
    l=[]
    for i in set(Part):
        temp=np.zeros(len(Part))
        temp=np.array([Part==i]).astype(int)[0]
        l.append(temp)
    return np.array(l)
def create_part_matr_mult(Parts):
    l=[]
    for i in Parts:
        l.append(create_part_matr(i))
        
    return l

def fill_dict(Matr,Obs,Part_mat,leng,Ap=None,Az=None,An=None):
    d={-1:0,0:0,1:0}
    d_comp={-1:0,0:0,1:0}.copy()
    q_comp=0
    
    one=np.ones(len(Part_mat))
    one_Matr=np.ones(len(Matr))
    comp_Matr=((one_Matr-np.eye(len(Matr)))*leng)-Obs
    max_obs_mtx=(np.ones((len(Matr),len(Matr)))*leng)
    #print('ide',max_obs_mtx)
    q_comp=np.transpose(one_Matr)@comp_Matr@one_Matr*0.5
    #print(q_comp)
    q=np.transpose(one_Matr)@Obs@one_Matr*0.5
    id=np.eye(len(Part_mat))
    out_obs=0
    boolean=(type(Ap)==type(None) or type(Az)==type(None) or type(An)==type(None))
    if boolean:
        #print("sparse")
        Ap=scipy.sparse.csr_array((Matr>0).astype(int))
        An=scipy.sparse.csr_array((Matr<0).astype(int))
        Az=scipy.sparse.csr_array(np.ones((len(Matr),len(Matr)))-Ap-An)
    yp=Ap*Obs
    yn=An*Obs
    yz=Az*Obs
    yp_comp=Ap*comp_Matr
    #print("yp_comp",yp_comp)
    yn_comp=An*comp_Matr
    yz_comp=Az*comp_Matr
    sum=0


    d_comp[1]=(Part_mat@yp_comp@np.transpose(Part_mat))*id*0.5
    #print(d_comp,"--------d_comp")
    d_comp[1]=np.transpose(one)@d_comp[1]@one
    #print(d_comp,"--------d_comp")

    #print(d_comp,"??",q_comp)

    d[1]=(Part_mat@yp@np.transpose(Part_mat))*id*0.5
    d[1]=np.transpose(one)@d[1]@one
    q=q-d[1]
    q_comp=q_comp-d_comp[1]
    #print(d[1],d_comp[1],"------------")
    #q_comp=q_comp-d_comp[1]
    d[0]=(Part_mat@yz@np.transpose(Part_mat))*id*0.5
    d[0]=np.transpose(one)@d[0]@one
    d_comp[0]=(Part_mat@yz_comp@np.transpose(Part_mat))*id*0.5
    d_comp[0]=np.transpose(one)@d_comp[0]@one
    q=q-d[0]
    q_comp=q_comp-d_comp[0]
    d[-1]=(Part_mat@yn@np.transpose(Part_mat))*id*0.5
    d[-1]=np.transpose(one)@d[-1]@one
    d_comp[-1]=(Part_mat@yn_comp@np.transpose(Part_mat))*id*0.5
    d_comp[-1]=np.transpose(one)@d_comp[-1]@one
    q=q-d[-1]
    q_comp=q_comp-d_comp[-1]
    #print("--------",d_comp,d,"--------")
    return d,q,d_comp,q_comp
def fill_dict_list(Matr,Obs_s,Part_s,lens,Ap=None, An=None, Az=None):
    d_list=[]
    d_comp_list=[]
    q_comp_list=[]
    q_list=[]
    boolean=(type(Ap)==type(None) or type(Az)==type(None) or type(An)==type(None))
    if boolean:
        """
        Apl=scipy.sparse.csr_array((Matr>0).astype(int))
        Anl=scipy.sparse.csr_array((Matr<0).astype(int))
        Azl=scipy.sparse.csr_array(np.ones((len(Matr),len(Matr)))-Apl-Anl)
        """
       
        Apl=(Matr>0).astype(int)
        Anl=(Matr<0).astype(int)
        Azl=np.ones((len(Matr),len(Matr)))-Apl-Anl
       
    for i in range(len(Obs_s)):
        a,b,a_comp,b_comp=fill_dict(Matr,Obs_s[i],Part_s[i],lens[i],Apl,Azl,Anl)
        #print('---',a,a_comp)
        d_list.append(a)
        d_comp_list.append(a_comp)
        q_list.append(b)
        q_comp_list.append(b_comp)
    return d_list,q_list,d_comp_list,q_comp_list
def create_part_matr_label(Part,label):
    l=[]
    for i in set(Part):
        temp=np.zeros(len(Part))
        temp=np.array([Part==i]).astype(int)[0]
        if i==label:
            return temp
    return np.zeros(len(Part))
def update_list_fast(Matr,Obs,Part_old,Part_new,leng,node_moved,d,comp_d,q,comp_q):

    if list(Part_old)==list(Part_new):
        return d,q,comp_d,comp_q
    d=d.copy()
    comp_d=comp_d.copy()
    q=q
    comp_q=comp_q
    part_matr_old_g1=create_part_matr_label(Part_old,Part_old[node_moved])
    part_matr_old_g2=create_part_matr_label(Part_old,Part_new[node_moved])
    part_matr_new_g1=create_part_matr_label(Part_new,Part_old[node_moved])
    part_matr_new_g2=create_part_matr_label(Part_new,Part_new[node_moved])
    #print("diag:",part_matr_new_g1-part_matr_old_g1,node_moved,part_matr_new_g2-part_matr_old_g2)
    #print(part_matr_old_g1,print(Obs[node_moved],"33"))
    #print(part_matr_old_g1,part_matr_old_g2,part_matr_new_g1,part_matr_new_g2,"dddd")
    Apos=(Matr[node_moved]>0).astype(int)
    Aneg=(Matr[node_moved]<0).astype(int)
    Azer=(Matr[node_moved]==0).astype(int)
    #Azer[node_moved]=1
    ones=np.ones(len(Matr))
    
    An=Aneg*Obs[node_moved]
    Ap=Apos*Obs[node_moved]
    Az=Azer*Obs[node_moved]
    Aneg_comp=(Aneg*leng-An)
    Azer_comp=(Azer*leng-Az)
    Apos_comp=(Apos*leng-Ap)
    #print(Apos_comp,"Apos_comp")
    #print(Ap)
    #print(An)
    #print(np.transpose(part_matr_old_g1)@(An*Obs[node_moved]),"it")
    #print(part_matr_new_g2)
    valng1_old=np.transpose(part_matr_old_g1)@(An)
    valpg1_old=np.transpose(part_matr_old_g1)@(Ap)
    valzg1_old=np.transpose(part_matr_old_g1)@(Az)
    valng1_old_comp=np.transpose(part_matr_old_g1)@Aneg_comp
    valpg1_old_comp=np.transpose(part_matr_old_g1)@Apos_comp
    valzg1_old_comp=np.transpose(part_matr_old_g1)@Azer_comp
    """
    valng2_old=np.transpose(part_matr_old_g2)@(An)
    valpg2_old=np.transpose(part_matr_old_g2)@(Ap)
    valzg2_old=np.transpose(part_matr_old_g2)@(Az)
    valng1_new=np.transpose(part_matr_new_g1)@(An)
    valpg1_new=np.transpose(part_matr_new_g1)@(Ap)
    valzg1_new=np.transpose(part_matr_new_g1)@(Az)
    valng2_new_comp=leng*np.transpose(part_matr_new_g2)@Aneg_comp
    valpg2_new_comp=leng*np.transpose(part_matr_new_g2)@Apos_comp
    valzg2_new_comp=leng*np.transpose(part_matr_new_g2)@Azer_comp
    """
    valng2_new=np.transpose(part_matr_new_g2)@(An)
    valpg2_new=np.transpose(part_matr_new_g2)@(Ap)
    valzg2_new=np.transpose(part_matr_new_g2)@(Az)
    valng2_new_comp=np.transpose(part_matr_new_g2)@Aneg_comp
    valpg2_new_comp=np.transpose(part_matr_new_g2)@Apos_comp
    valzg2_new_comp=np.transpose(part_matr_new_g2)@Azer_comp
    q+=valng1_old+valpg1_old+valzg1_old-valpg2_new-valzg2_new-valng2_new
    #print(valpg1_new,valpg1_old,valpg2_new,valpg2_old,valzg1_new,valzg1_old)
    d[1]+=-valpg1_old+valpg2_new
    d[0]+=-valzg1_old+valzg2_new
    d[-1]+=-valng1_old+valng2_new
    comp_d[-1]+=valng2_new_comp-valng1_old_comp
    comp_d[0]+=valzg2_new_comp-valzg1_old_comp
    comp_d[1]+=valpg2_new_comp-valpg1_old_comp
    comp_q-=valpg2_new_comp+valng2_new_comp+valzg2_new_comp-valzg1_old_comp-valpg1_old_comp-valng1_old_comp
    #print("itt,",comp_d,"halo",(np.transpose(part_matr_new_g2)@(Aneg*leng)-An))
    #print(d,valpg1_old,valng1_old,valzg1_old,valng1_new)
    return d,q,comp_d,comp_q


def likelihood_of_parameter_set(rates,q_rates,obs,comp_obs,obs_out,comp_obs_out):
    val=0
    for i in comp_obs.keys():
        val+=obs[i]*np.log(rates[i])+comp_obs[i]*np.log(1-rates[i])
    val+= obs_out*np.log(q_rates)+comp_obs_out*np.log(1-q_rates)
    return val


def Joint_MCMC(Obs_s,lengs,part_input,n_iter=1000,m_sample_size=1,Temp=1,rate_input=None,q_rate_input=None):
    X_s=Obs_s.copy()
    lens=lengs.copy()
    #print("kezdődik",part_input)
    partitions=part_input.copy()
    rate={-1:0.3,0:0.31,1:0.32}
    rates=[]
    ret_mtx=[]
    q_rates=[]

    ret_rates=[]
    #rates={-1: 0.003978140225228608, 0: 0.20028941740668166, 1: 0.8152465133081441}
    if rate_input==None or q_rate_input==None:
        for i in range(len(X_s)):
            rates.append(rate.copy())
            q_rates.append(0.05)
    else:
        rate=rate_input
        q_rates=q_rate_input
    #rates=[{-1: 0.1992735498059376, 0: 0.4883190199211896, 1: 0.7483995329033881}, {-1: 0.19556273275727853, 0: 0.469835404799444, 1: 0.7336705418377941}, {-1: 0.07735813085188166, 0: 0.422853449371757, 1: 0.7233756346385437}, {-1: 0.14246849598250963, 0: 0.40714616511560586, 1: 0.7072911155847346}, {-1: 0.13982386443614958, 0: 0.41472723462385225, 1: 0.72449584295059}, {-1: 0.1601165034224108, 0: 0.46501507396983577, 1: 0.7415460214840506}, {-1: 0.11792081221987215, 0: 0.4672556706245484, 1: 0.7150129966340323}]

    keys=[-1,0,1]
    A=np.ones((len(X_s[0]),len(X_s[0])))-np.identity(len(X_s[0]))
    curr_param=[A.copy()]


    observation_agg=[]
    comp_obs_agg=[]
    obs_agg_out=[]
    obs_agg_out_comp=[]
    for i in rates:
        curr_param.append(i.copy())
    #print("kezdő",curr_param,"  vég")
    #print(A)
    #d_list,q_list,d_comp_list,q_comp_list
    observation_agg,obs_agg_out,comp_obs_agg,obs_agg_out_comp=fill_dict_list(curr_param[0],X_s,create_part_matr_mult(partitions),lens)
    
                #print(A[i][j])
    print("START OF Sampler, Init completed")
    for i in range(m_sample_size):
        for j in range(n_iter):
            if j%1000==0:
                #print(round((((j+i*n_iter)/(m_sample_size*n_iter)))*100),'%-nál vagyunk',end="\r") 
                print(j)
            """
            print(observation_agg)
            print(comp_obs_agg)
            print(obs_agg_out)
            print(obs_agg_out_comp)
            """
            for k in range(len(curr_param)):
                """
                print("------------------")
                print(observation_agg)
                print("******************")
                print(comp_obs_agg)
                print("------------------")
                """
                prop_param=[curr_param[0].copy(),*curr_param[1:].copy()]
                #print("----prop_param",prop_param[0],"idáig")
                if k==0:
                    part_change=False
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    a=np.random.choice(len(X_s[0]))
                    b=a
                    while a==b:
                        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                        b=np.random.choice(len(X_s[0]))
                    #print(a,b)
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    prop_param[0][a][b]=np.random.choice([-1,0,1])
                    prop_param[0][b][a]=prop_param[0][a][b]
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    u=np.log(np.random.rand())
                    #val1=likelihood(prop_param[0],curr_param[1],X_s,lens)
                    #val2=likelihood(curr_param[0],curr_param[1],X_s,lens)
                    #val=likelihood_fast(curr_param[0][a][b],prop_param[0][a][b],lens,X_s[a][b],partitions,a,b,curr_param[1])
                    if check_mtx_change_list(partitions,prop_param[0],curr_param[0],a,b)==True:
                        val=likelihood_fast_multiple_obs_mtx_change(prop_param[0][a][b],curr_param[0][a][b],curr_param[1:],X_s,lens,partitions,a,b)
                        #likelihood_fast_multiple_obs_mtx_change(new_val,old_val,reals,X_s,lens,partitions,a,b)
                    
                    #print("csere",j,val,u,val1,val2,curr_param[0]-prop_param[0])
                    else:
                        """
                        temp_part=partitions.copy()
                        part_new=new_partition_split_list(prop_param[0],temp_part,a,b)
                        #val=likelihood_fast_multiple_obs_mtx_change(prop_param[0][a][b],curr_param[0][a][b],curr_param[1:],X_s,lens,partitions,a,b)#+likelihood_of_split_list(A,X_s,a,curr_param[1:],q_rates,lens,partitions,a,b)
                        #ezt a sort alul kikommenteltem ,hogy a régi futás legyen. ha ezt elengedem, akkor az új alg megy majd.
                        part_change=True
                        """
                        val=-240
                    #print(val)
                    if val>u and part_change==False:
                        #print("csere",j,val,u,val1,val2,curr_param[0]-prop_param[0])
                        for updater in range(len(partitions)):
                            if partitions[updater][a]==partitions[updater][b]:
                                
                                #print("akció",observation_agg[updater][prop_param[0][a][b]],X_s[updater][a][b])
                                #print(X_s)
                                observation_agg[updater][prop_param[0][a][b]]=X_s[updater][a][b]+observation_agg[updater][prop_param[0][a][b]]
                                observation_agg[updater][curr_param[0][a][b]]-=X_s[updater][a][b]
                                comp_obs_agg[updater][prop_param[0][a][b]]+=lens[updater]-X_s[updater][a][b]
                                comp_obs_agg[updater][curr_param[0][a][b]]-=lens[updater]-X_s[updater][a][b]
                                #print(observation_agg)
                        curr_param[0]=prop_param[0].copy()
                    if val>u and part_change==True and True ==False:
                        print("break")
                        break
                        partitions=temp_part.copy()
                        curr_param[0]=prop_param[0].copy()
                        observation_agg,obs_agg_out,comp_obs_agg,obs_agg_out_comp=fill_dict_list(curr_param[0],X_s,create_part_matr_mult(partitions),lens)
                     
                    #("itt: ",)
                       
                if k!=0 and True==True:
                    #print("itt")
                    
                    for l in keys:
                        prop_param[k]=curr_param[k].copy()
                        #print(prop_param[k],prop_param)
                        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                        prop_param[k][l]=prop_param[k][l]+np.random.normal(0,0.1,1)[0]
                       
                        if 0<prop_param[k][-1] and prop_param[k][-1]<prop_param[k][0] and prop_param[k][0]<prop_param[k][1] and prop_param[k][1]<1:
                            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                            u=np.log(np.random.rand())
                            #print("legit")
                            """
                            val1=likelihood(curr_param[0],prop_param[1],X_s,lens,partitions)
                            val2=likelihood(curr_param[0],curr_param[1],X_s,lens,partitions)
                            val=val1-val2
                            val_new=likelihood_fast_prob_change(curr_param[0],X_s,lens,curr_param[1],prop_param[1],l,partitions)
                            if val-val_new>10e-5:
                                print("--------------------------")
                                print(val,val_new)
                                print("--------------------------")
                            #print("csere",j,val,u,val1,val2,curr_param[0]-prop_param[0])
                            """
                            #print(val)
                            #print((curr_param[0],X_s[k-1],lens[k-1],curr_param[k],prop_param[k],l,partitions[k-1]))
                            #Nem 1 hanem az összes X_stől kéne függenie a probnak
                            val_new=likelihood_fast_prob_change(curr_param[0],X_s[k-1],lens[k-1],curr_param[k],prop_param[k],l,partitions[k-1],observation_agg[k-1],comp_obs_agg[k-1])
                            if val_new>u:
                                1
                                curr_param[k]=prop_param[k].copy()
                                #print('probcsere',j,curr_param[1],prop_param[1])
                        else: 
                            #print("forbidden params", prop_param)
                            prop_param[k]=curr_param[k].copy()
                            1
                else:
                    1
            #q_adjustment
            for f in range(len(q_rates)):
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                prop_q=np.random.normal(0,0.01,1)[0]+q_rates[f]
                if prop_q>0.1 or prop_q<0:
                    val=-240
                else:
                    val=obs_agg_out[f]*(np.log(prop_q)-np.log(q_rates[f]))+obs_agg_out_comp[f]*(np.log(1-prop_q)-np.log(1-q_rates[f]))
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    u=np.log(np.random.rand())
                    if val>u:
                        q_rates[f]=prop_q
                    else:
                        1
            #parts_inference
            if j==2500:
                observation_agg=[]
                comp_obs_agg=[]
                obs_agg_out=[]
                obs_agg_out_comp=[]
                for h in range(len(partitions)):
                    partitions[h]=np.array(list(range(len(X_s[h]))))
                    observation_agg.append({-1:0,0:0,1:0})
                    comp_obs_agg.append({-1:0,0:0,1:0})
                    obs_agg_out.append(0)
                    obs_agg_out_comp.append(0)
                for i in range(len(A)):
                    for j in range(i):
                        for k in range(len(X_s)):
                            if partitions[k][i]==partitions[k][j]:
                                observation_agg[k][A[i][j]]+=X_s[k][i][j]
                                comp_obs_agg[k][A[i][j]]+=lens[k]-X_s[k][i][j]
                            else:
                                obs_agg_out[k]+=X_s[k][i][j]
                                obs_agg_out_comp[k]+=lens[k]-X_s[k][i][j]
                    
                    
            if j>2501:
                for h in range(len(partitions)):
                    prop_partition=np.copy(partitions[h])
                    #print(partitions[h])
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    ind_to_move=np.random.choice(len(X_s[0]))
                    #print("ind to move",ind_to_move)
                    group_prop=prop_group_move(ind_to_move,partitions[h],X_s[h])
                    #print("group_prp",group_prop)

                    
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    u=np.log(np.random.rand())
                    prop_partition[ind_to_move]=group_prop
                    val=likelihood_partition(curr_param[0],X_s[h],curr_param[h+1],lens[h],q_rates[h],ind_to_move,prop_partition,partitions[h],new_part_label=None,old_part_label=None)
                    if val>u:
                        last_change=j
                        observation_agg[h],obs_agg_out[h],comp_obs_agg[h],obs_agg_out_comp[h]=update_list_fast(curr_param[0],X_s[h],partitions[h],prop_partition,lens[h],ind_to_move,observation_agg[h],comp_obs_agg[h],obs_agg_out[h],obs_agg_out_comp[h])    
                        #update_list_fast(Matr,Obs,Part_old,Part_new,leng,node_moved,d[h],comp_d[h],q[h],comp_q[h]):
                        partitions[h]=prop_partition.copy()
                        #observation_agg,obs_agg_out,comp_obs_agg,obs_agg_out_comp=fill_dict_list(curr_param[0],X_s,create_part_matr_mult(partitions),lens)
                        #partitions[h]=relabel(partitions[h])
                #print(A[i][j])



                    else:
                        1
                

            
            


                
        ret_mtx.append(curr_param[0].copy())
        ret_rates.append(curr_param[1:].copy())



    #print(curr_param)
    print(last_change,"last change was at")
    #print("param_likelihood",likelihood_of_parameter_set_list(curr_param[1:],q_rates,observation_agg,comp_obs_agg,obs_agg_out,obs_agg_out_comp))
    return ret_mtx,ret_rates,partitions