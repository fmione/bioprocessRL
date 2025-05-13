# %% Import
import numpy as np
# import json
import time
from scipy.integrate import solve_ivp
# from joblib import Parallel, delayed

from copy import deepcopy

from scipy.optimize import shgo,dual_annealing,minimize,differential_evolution

import matplotlib.pyplot as plt

# %%
def calculate_DIV(tt,XX0,uu,TH_param0,DD,Cov_y=[]):
    # print('calculating reward...')
    XX_th0=simulate_parallel(tt,XX0,uu,TH_param0,DD)
    # t_X=np.arange(tt[0]+1,tt[-1])
    DIV_min=XX_th0['sample'][0][0][-1]
    
        
    # div_X={} 
    DIV=[]  
    # for i1 in range(uu[0][0]): 
    #     for i2 in range(i1+1,uu[0][0]):
    #         ts_X1=DD[i1]['time_sample']
    #         ts_X2=DD[i2]['time_sample']
    #         profileX_1=np.interp(t_X,ts_X1,np.array(XX_th0['sample'][i1][0]))
    #         profileX_2=np.interp(t_X,ts_X2,np.array(XX_th0['sample'][i2][0]))
    #         div_X[i1]={i2:np.sum(abs(profileX_1-profileX_2))}
            
    #         # div_X[i1]={i2: abs(np.sum((profileX_1-profileX_2)))}

    #         # plt.plot(t_X,profileX_1,t_X,profileX_2)
    #         DIV.append(1/(1/(1e-9+div_X[i1][i2])))
            
    # DIV_min=min(DIV)
            

    return XX_th0,DIV,DIV_min
# %%

def simulate_parallel(ts,XX0,uu,TH_param,DD):  
    XX=deepcopy(XX0)
    brxtor_list=np.arange(uu[0][0]).tolist()
    
    ty={}
    
    # results = Parallel(n_jobs=-1)(
    #     delayed(simulate_interval)(i1, ts,XX,uu,TH_param,DD)
    #     for i1 in brxtor_list
    # )
    

    # for i1, result in zip(brxtor_list, results):
    #     ty[i1] = result
        
    for i1 in brxtor_list: 
        ty[i1]=simulate_interval(i1, ts,XX,uu,TH_param,DD)

        
    for i1 in brxtor_list:
        XX['state'][i1]=ty[i1][-1,1:]
        
        
        for i2 in [0,1,2,4]:#range(4):
            ts_sample_all=DD[i1]['time_sample'] #CHECK
            ts_sample=ts_sample_all[(ts_sample_all>ts[0]) & (ts_sample_all<=ts[1])]
            
            sample_interp=np.interp(ts_sample,ty[i1][:,0],ty[i1][:,i2+1])
            # print(sample_interp.tolist())
            try:
                # print(XX['sample'][i1][i2])#,sample_interp.tolist())
                XX['sample'][i1][i2]=XX['sample'][i1][i2]+sample_interp.tolist() #CORRECT, append to existing
                
            except:
                XX['sample'][i1][i2]=sample_interp.tolist()
                # print(XX['sample'][i1][i2],sample_interp.tolist())
                
                
        ts_sensor_all=DD[i1]['time_sensor'] #CHECK
        ts_sensor=ts_sensor_all[(ts_sensor_all>ts[0]) & (ts_sensor_all<=ts[1])]
        sensor_interp=np.interp(ts_sensor,ty[i1][:,0],ty[i1][:,4])

        try:
            XX['sample'][i1][3]=XX['sample'][i1][3]+sensor_interp.tolist() #CORRECT, append to existing
            # print(len(XX['sample'][i1][3])   )
        except:
            XX['sample'][i1][3]=sensor_interp.tolist()
            # plt.plot(ts_sensor,sensor_interp)
            # plt.show()
            # time.sleep(1)
            

    return XX

# %%
# def func_parallel(index_mbr,number_mbr,time_initial,time_final,EMULATOR_state,EMULATOR_design,EMULATOR_config):
def simulate_interval(index_mbr,ts,XX,uu,TH_param,DD):

            
            u=[uu[index_mbr][1]]+[index_mbr]+[uu[index_mbr][0]]+[uu[index_mbr][2]]+[1]
            X = np.array(XX['state'][index_mbr])
            # print(u)
            D = DD[index_mbr]
            # print(u)
            t, y = function_simulation(ts, X, u, TH_param, D)
    

            # return t,y
            return np.hstack((t[:,None],y))


            raise
            
# %%
def function_simulation(ts0,Xo0,u0,THs,D0={}):
    TH1=THs[0:16]


    TH1=np.append(TH1,THs[16+int(u0[1])])
    TH1=np.append(TH1,THs[16+int(u0[1])+int(u0[2])]) 
    
    
    ts_start=ts0[0]
    ts_end=ts0[-1]
    
    time_pulse_all=np.array(D0['time_pulse'])
    Feed_pulse_all=np.array(D0['Feed_pulse'])
    
    t_u=time_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
    uu_base_design=Feed_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
 
    uu=uu_base_design

    if len(t_u)==0:
        t_u=np.array([ts_start,ts_end])
        uu=np.array([0,0])
    else:
        if ts_start<t_u[0]:
            t_u=np.append(ts_start,t_u)
            uu=np.append(0,uu)
        if ts_end>t_u[-1]:
            t_u=np.append(t_u,ts_end)
            uu=np.append(uu,0)

    Xo1=Xo0.copy()
    
    tt=np.array(ts_start)
    yy=np.array([Xo1])
    yy=yy.transpose()
    
    ni=0
    
    for i in uu[:-1]:
        ts1=np.linspace(t_u[ni],t_u[ni+1],25+1)
        Xo1[1]=Xo1[1]+uu[ni]*1e-6*u0[0]/0.01
        t,y=intM(ts1,Xo1,u0,TH1)
        Xo1=y[:,-1].copy()

        
        tt=np.append(tt,t[1:])
        yy=np.append(yy,y[:,1:],axis=1)
        ni=ni+1

       
    # y_pd=pd.DataFrame(yy)
    return tt,yy.transpose()

# %%    
def odeFB(t,Xo,THo,u):

    X=Xo.copy()
    TH=THo.copy()
    # print(X)
    X = np.maximum(X, 1e-9)
    
    Xv=X[0]
    S=X[1]
    A=X[2]
    DOT=X[3] 
    P=X[4]
    mu_m=X[5]

    DOT = np.minimum(DOT, 100)
            
    qs_max=TH[0]
    fracc_q_ox_max=TH[1]
    qa_max=TH[2]
    # b_prod=TH[3]
    
    
    Ys_ox=TH[4]
    Ya_p=TH[5]
    Ya_c=TH[6]
    # Yp=TH[7]
    Yo_ox=TH[8]
    Yo_a=TH[9]
    Yxs_of=TH[10]
    
    Ks=TH[11]
    n_ox=4
    
    Ka=TH[12]
    Ksi=TH[3]#7.0767#TH[13]# 
    Kai=TH[7]#.4242#TH[13]#
    Ko=0.1057#TH[15]#
    
    kla=TH[16]
    k_sensor=TH[17]
    
    ky_1=TH[13] 
    ky_2=TH[14]
    ky_3=TH[15]
    
    DO_star=100
    H=13000#
    
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)#*(1-P/70)
    q_ox_max=fracc_q_ox_max*qs_max
    
    q_ox_ss=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)
    qac_ss=qa_max*A/(A+Ka)*Kai/(Kai+S)
    b_ss=Ko+(q_ox_ss*Yo_ox+qac_ss*Yo_a)*Xv*H/kla-DO_star
    c_ss=-DO_star*Ko
    DOT_ss=(-b_ss+(b_ss*b_ss-4*c_ss)**.5)/2


    # qm=qm_max*qs_max*S/(S+1e-6)*DOT_ss/(DOT_ss+Ko)
    
    q_ox=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)*DOT_ss/(DOT_ss+Ko)
    q_of=qs-q_ox
    
    qac=qa_max*A/(A+Ka)*Kai/(Kai+S)*DOT_ss/(DOT_ss+Ko)
    
    qap=q_of*Ya_p
    
    mu=q_ox*Ys_ox+qac*Ya_c+Yxs_of*q_of

    if t>=u[3]:
        s_prod=u[4]
    else:
        s_prod=0
        
    
    f_qp=ky_1*mu_m/(mu_m+ky_2+(ky_3*mu_m)**2)

    q_prod=s_prod*f_qp


    dXv=(mu)*Xv
    dS=-(qs)*Xv
    dA=qap*Xv-qac*Xv
    dDOT=k_sensor*(DOT_ss-DOT)
    dP=q_prod*Xv
    dmu_m=(mu-mu_m)/(.167)
    
    dX=np.array([dXv,dS,dA,dDOT,dP,dmu_m])
    return dX
# %%     
def intM(ts0,Xo0,u0,TH0):    

    tspan=np.array([ts0[0],ts0[-1]])
    Xo1=Xo0.tolist().copy()



    sol=solve_ivp(lambda t,y: odeFB(t,y,TH0,u0) ,tspan,Xo1,method="BDF", rtol=1e-5, atol=1e-5,t_eval=ts0)
    y_interm=sol.y
    y_interm[y_interm<0]=0
    y_return=y_interm.copy()

    return sol.t,y_return