import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import sklearn
from sklearn.metrics import roc_curve
import scipy
import os
import multiprocessing
import pandas as pd
from multiprocessing import Pool
import random
from signed_inference import *
import subprocess

from statsmodels.miscmodels.ordinal_model import OrderedModel

import datetime
def reg_threshold(A,X,percentage_known):
    num_obs=len(A)*(len(A)-1)*0.5
    #print(num_obs)
    stop=False
    chosen_indices=[]
    while stop==False:
        
        tempo1=np.random.choice(range(len(A)))
        tempo2=np.random.choice(range(len(A)))
        temp1=min(tempo1,tempo2)
        temp2=max(tempo1,tempo2)
        if temp1!=temp2 and [temp1,temp2] not in chosen_indices:
            chosen_indices.append([temp1,temp2])
        if len(chosen_indices)/num_obs>=percentage_known:
            stop=True
    data=[]
    network=[]
    
    for i in chosen_indices:
        data.append(X[i[0]][i[1]])
        network.append(A[i[0]][i[1]])


    #print(scipy.stats.linregress(data,network))
    return data,network
def infer_mtx(thresholds,observations):
    inf=np.zeros((len(observations),len(observations)))
    for i in range(len(inf)):
        for j in range(i):
            if observations[i][j]<thresholds[0]:
                inf[i][j]=-1
                inf[j][i]=-1
            if observations[i][j]>thresholds[1]:
                inf[i][j]=1
                inf[j][i]=1
    return inf
def prob_list(num):
    num_og=num
    t=np.zeros(num)
    for i in range(len(t)):
        t[i]=np.random.randint(0,num+1)
        num=num-t[i]
    print('sum(t)',sum(t))
    if sum(t)==0:
        t=np.ones(num_og)
    np.random.shuffle(t)
    return t
print(prob_list(20))
def part_covering(par):
    parm=create_part_matr(np.transpose(par))@np.ones((len(par),1))
    t=0
    for i in parm:
        #print(i[0])
        t+=i[0]*(i[0]-1)*0.5
    return t/(len(par)*(len(par)-1)*0.5)

def eval_inf(A,A_inf):
    pos_reals=[]
    pos_inferred=[]
    neg_inferred=[]
    neg_reals=[]
    P=(A>0). astype(int)
    N=(A<0). astype(int)
    for j in range(len(P)):
        for i in range(len(P)): 
            pos_reals.append(P[i][j])
    print(len(pos_reals),'lens')
    for j in range(len(N)):
        for i in range(len(N)): 
            neg_reals.append(N[i][j])
                        
    """
    for yy in pos_mean:
        for j in yy:
            pos_inferred.append(j)
    """
    print(len(pos_reals),'lens')
    pos_mean=(A_inf>0). astype(int)
    for j in range(len(pos_mean)):
        for i in range(len(pos_mean)):
            pos_inferred.append(pos_mean[i][j])

    neg_mean=(A_inf<0). astype(int)
    for j in range(len(neg_mean)):
        for i in range(len(neg_mean)):
            neg_inferred.append(neg_mean[i][j])


    zeros_inferred=[]
    zeros_real=[]
    zeros_mean=(A_inf==0). astype(int)

    Z=(A==0).astype(int)
    for j in range(len(zeros_mean)):
        for i in range(len(zeros_mean)):
            zeros_inferred.append(zeros_mean[i][j])
    for j in range(len(Z)):
        for i in range(len(Z)):
            zeros_real.append(Z[i][j])
        
    """
    fpr_pos,tpr_pos,thresholds=sklearn.metrics.roc_curve(pos_reals,pos_inferred)


    fpr_neg,tpr_neg,thresholds=sklearn.metrics.roc_curve(neg_reals,neg_inferred)

    plt.subplot(2,2,1)
    plt.title("ROC curve positive edges")

    plt.plot(fpr_pos,tpr_pos)
    plt.plot([0,1],[0,1],"--r")
    plt.text(0.5,0,"AUC: "+str(np.round(sklearn.metrics.roc_auc_score(pos_reals, pos_inferred),4)))

    plt.subplot(2,2,2)
    plt.title("ROC curve negative edges")

    plt.plot(fpr_neg,tpr_neg)
    plt.plot([0,1],[0,1],"--r")
    plt.text(0.5,0,"AUC: "+str(np.round(sklearn.metrics.roc_auc_score(neg_reals, neg_inferred),4)))
    """
    print(sklearn.metrics.roc_auc_score(pos_reals, pos_inferred))

    print(sklearn.metrics.roc_auc_score(neg_reals, neg_inferred))
    #pos,neg return
    return [sklearn.metrics.roc_auc_score(pos_reals, pos_inferred),sklearn.metrics.roc_auc_score(neg_reals, neg_inferred)]
def deg_based_inference(X):
    ones=np.ones(len(X))
    deg_vec=np.transpose(ones)@X
    edges=2*deg_vec@ones
    #print(edges)
    #print(X-np.outer(deg_vec,deg_vec)/edges)
    a=X-np.outer(deg_vec,deg_vec)/edges
    for i in range(len(a)):
        a[i][i]=0
    return a
reg_params=[0.5]

d_mcmc={}
thr={}
d_cm={}
d_reg={}
for x in range(64):
    g = nx.generators.random_graphs.erdos_renyi_graph(64, 0.5)

    for e in g.edges():
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        g.edges[e]['weight'] = np.random.choice([-1, 1])

    A = nx.adjacency_matrix(g).todense()
    P = (A > 0).astype(int)
    N = (A < 0).astype(int)
   

    X_s=[]
    #print(A)
    partitions=[]
    #rr={-1:0.25,0:0.655,1:0.9751}
    reals=[]
    lens=[]
    for i in range(1):
        #partition= partition_mcmc(N, P, round(len(P)/2),1000)
        #partition=np.zeros(len(A))
        #partition=np.random.randint(0,2,len(A))
        leng=64
        probs=np.ones(leng)
        probs=prob_list(leng)
        probs/=sum(probs)   
        partition=np.random.choice(range(leng),64,p=probs)
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        curr_len=np.random.randint(20,110)
        curr_len=250
        if np.random.rand()>10:
            rr1=np.random.uniform(0.1,0.2)
            rr2=np.random.uniform(0.4,0.5)
            rr3=np.random.uniform(0.7,0.75)
            print("+++")
        else:
            rr1=np.random.uniform(0,0.01)
            rr2=np.random.uniform(0.2,0.3)
            rr3=np.random.uniform(0.8,0.9)
        rr={-1:rr1,0:rr2,1:rr3}
        reals.append(rr)
        X_s.append(genrete_binomial_measurements(partition, A, curr_len, rates=rr,q_rate=0.003))
        lens.append(curr_len)
        #Parallelized inference
    parta=parta=np.zeros((len(X_s),len(X_s[0])))

    def tryer_mcmc2(l):
        a,b,c,d=Joint_MCMC(X_s,lens,parta,l,1)
        print("rates:",b)
        print("partitions:",c)
        return a,b


    inputs=[]
    for i in range(8):
        inputs.append(22500)

    if __name__ == '__main__':
        with Pool(12) as p:
            #print(p.map(f, [1, 2, 3]))
            ll=p.map(tryer_mcmc2,inputs)


    inferences_mtx=[]
    rates_inferred=[]
    for i in ll:
        for j in i[0]:
            inferences_mtx.append(j)
        for k in i[1]:
            rates_inferred.append(k)
    y=inferences_mtx
    print(sum(y)/len(y))
    y=sum(y)/len(y)

    positives=[]
    negatives=[]
    zeros=[]
    #A=np.load('A.npy')
    B=(A  <0).astype(int)
    B=-B
    #y=a
    y=inferences_mtx
    for i in y:
        negatives.append((i  <0).astype(int))
        positives.append((i  >0).astype(int))
        zeros.append((i==0).astype(int))
    pos_mean=sum(positives)/len(positives)
    zeros_mean=sum(zeros)/len(zeros)

    neg_mean=sum(negatives)/len(negatives)


    pos_reals=[]
    pos_inferred=[]
    P=(A>0). astype(int)
    N=(A<0). astype(int)
    for j in range(len(P)):
        for i in range(len(P)): 
            pos_reals.append(P[i][j])
                        
    """
    for yy in pos_mean:
        for j in yy:
            pos_inferred.append(j)
    """
    for j in range(len(pos_mean)):
        for i in range(len(pos_mean)):
            pos_inferred.append(pos_mean[i][j])
    #print(len(pos_inferred),'pozitivak',pos_reals,pos_inferred)
    fpr_pos,tpr_pos,thresholds=sklearn.metrics.roc_curve(pos_reals,pos_inferred)

    #print(pos_inferred)
    neg_reals=[]
    neg_inferred=[]
    """
    for xx in N:
        for i in xx:
            neg_reals.append(i)
    for yy in neg_mean:
        for j in yy:
            neg_inferred.append(j)
    """
    for j in range(len(N)):
        for i in range(len(N)):
            neg_reals.append(N[i][j])

    for j in range(len(neg_mean)):
        for i in range(len(neg_mean)):
            neg_inferred.append(neg_mean[i][j])
        
    fpr_neg,tpr_neg,thresholds=sklearn.metrics.roc_curve(neg_reals,neg_inferred)

    zeros_inferred=[]
    zeros_real=[]
    Z=(A==0).astype(int)
    for j in range(len(zeros_mean)):
        for i in range(len(zeros_mean)):
            zeros_inferred.append(zeros_mean[i][j])
    for j in range(len(Z)):
        for i in range(len(Z)):
            zeros_real.append(Z[i][j])


    print(len(pos_inferred),"alappontok száma")



    print("AUC score for positive edges: ",sklearn.metrics.roc_auc_score(pos_reals, pos_inferred) )
    print("AUC score for negative edges: ",sklearn.metrics.roc_auc_score(neg_reals, neg_inferred) )
    d_mcmc[part_covering(partition)-np.abs(np.random.normal(0,0.000001,1)[0])]=[[sklearn.metrics.roc_auc_score(pos_reals, pos_inferred),sklearn.metrics.roc_auc_score(zeros_real, zeros_inferred),sklearn.metrics.roc_auc_score(neg_reals, neg_inferred)]]
    #print('negatív',neg_reals,neg_inferred)
    

    A_inf=deg_based_inference(X_s[0])
    #print('eval eval',eval_inf(A,A_inf),r,part_covering(partition))
    d_cm[part_covering(partition)-np.abs(np.random.normal(0,0.000001,1)[0])]=eval_inf(A,A_inf)
    
    temp={}
    for r in reg_params:
        
        a=reg_threshold(A,X_s[0],r)
        print('lensssss',len(a[0]))
        data=[a[0],a[1]]
        data=np.transpose(data)
        df = pd.DataFrame(data)
        df=df.rename(columns={0: "observation", 1: "network"})
        from pandas.api.types import CategoricalDtype
        s = df['network']
        cat_type = CategoricalDtype(categories=[-1,0,1],ordered=True)
        s_cat = s.astype(cat_type)
        s_cat

        df["network"]=df["network"].astype(cat_type)
        from pandas.api.types import CategoricalDtype
        s = df['network']
        cat_type = CategoricalDtype(categories=[-1,0,1],ordered=True)
        s_cat = s.astype(cat_type)
        s_cat

        df["network"]=df["network"].astype(cat_type)

        from pandas.api.types import CategoricalDtype
        s = df['network']
        cat_type = CategoricalDtype(categories=[-1,0,1],ordered=True)
        s_cat = s.astype(cat_type)
        s_cat
        print(df.head())
        mod_prob = OrderedModel(df['network'],
                                df[['observation']],
                                distr='probit')

        res_prob = mod_prob.fit(method='bfgs')
        res_prob.summary()
        num_of_thresholds = 2
        print(mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:]))
        thr=[mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:])[1],mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:])[2]]
        A_inf=infer_mtx(thr,X_s[0])
        print("thr------",thr)
        #A_inf=np.random.randint(-1,2,size=(64,64))
        #print('eval eval',eval_inf(A,A_inf),r,part_covering(partition))
        try:
            temp[r].append(eval_inf(A,A_inf))
        except: 

            temp[r]=eval_inf(A,A_inf)
    d_reg[part_covering(partition)-np.abs(np.random.normal(0,0.000001,1)[0])]=temp
     
cmpos=[]
cmneg=[]
cmagg=[]
for i in d_cm.keys():
    cmpos.append(d_cm[i][1])
    cmneg.append(d_cm[i][0])
    cmagg.append(0.5*(d_cm[i][0]+d_cm[i][1]))
posmc,negmc,mcagg=[],[],[]
for i in d_mcmc.keys():
    for j in d_mcmc[i]:
        negmc.append(j[2])
        posmc.append(j[0])
        mcagg.append((j[1]+j[0])*0.5)
reg_pos,reg_neg,reg_agg=[],[],[]
for i in d_reg.keys():
    reg_pos.append(d_reg[i][0.5][0])
    reg_neg.append(d_reg[i][0.5][1])
    reg_agg.append(0.5*(d_reg[i][0.5][0]+d_reg[i][0.5][1]))

f, axs = plt.subplots(1, 2, figsize=(12, 6))


plt.subplot(1,2,1)
plt.scatter(d_cm.keys(),cmpos,label='Configuration Model',marker='P',c='crimson')
plt.scatter(d_reg.keys(),reg_pos,label='Regression Model',marker='*',c='orchid')

plt.scatter(d_mcmc.keys(),posmc,label='MCMC',marker='^',c='blue')
plt.xticks(np.arange(0, 1.2, step=0.2),size=16)
plt.yticks(np.arange(0.5, 1.05, step=0.1),size=16)
plt.legend(fontsize=14)
plt.title("Inference of positive edges",size=18)
plt.xlabel('Internal edge fraction',size=18)
plt.ylabel('AUC',size=18)
plt.subplot(1,2,2)
plt.scatter(d_cm.keys(),cmneg,label='Configuration Model',marker='P',c='crimson')
plt.scatter(d_reg.keys(),reg_neg,label='Regression Model',marker='*',c='orchid')

plt.scatter(d_mcmc.keys(),negmc,label='MCMC',marker='^',c='blue')
#plt.legend(fontsize=14)
plt.xticks(np.arange(0, 1.2, step=0.2),size=16)
plt.yticks(np.arange(0.5, 1.05, step=0.1),size=16)

plt.title("Inference of negative edges",size=18)
plt.xlabel('Internal edge fraction',size=18)
plt.ylabel('AUC',size=18)
plt.savefig('Inference_baselines.svg')
plt.savefig('Inference_baselines.pdf')
