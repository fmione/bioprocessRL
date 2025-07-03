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
def optimizer_reference(LB,UB,optim_options,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0=[]):
   
    ux_lb=np.array(LB)
    ux_ub=np.array(UB)
    ux_mean=np.linspace(LB[0],UB[0],len(LB))
    
    bounds_ux=list(range(len(ux_lb)))
    for i1 in range(len(ux_lb)):
        bounds_ux[i1]=(ux_lb[i1],ux_ub[i1])
    
    t_test0=time.time()    
    e1=obj_fun(ux_lb,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0)
    e2=obj_fun(ux_ub,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0)
    e3=obj_fun(ux_mean,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0) 
    t_test=(time.time()-t_test0)/3

    n_fun_eval0=round(60*optim_options[0]/(t_test*1.2))
    n_fun_eval1=round(60*optim_options[1]/(t_test*1.2))
    
    print(n_fun_eval0,n_fun_eval1)  
    
    ux_opt0 = dual_annealing(lambda ux: obj_fun(ux,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0), bounds=bounds_ux, maxfun=n_fun_eval0, no_local_search=True)
    print('global opt ',ux_opt0.x)
    ux_opt1 = minimize(lambda ux: obj_fun(ux,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0), ux_opt0.x, bounds=bounds_ux, method='Nelder-Mead', options={'maxfev': n_fun_eval1})

    print('local opt ',ux_opt1.x)
    return ux_opt1        

# %%

def obj_fun(ux,tt,XX0,uu,TH_param0,DD,Cov_y,Cov_TH0=[]):
    DDx=deepcopy(DD)
        
    for i2 in range(uu[0][0]):
        t_pulse=np.array(DDx[i2]['time_pulse'])
        Feed_pulse=(32.406)*ux[i2]*np.exp(ux[i2]*(t_pulse-t_pulse[0]))
        Feed_pulse=np.round(Feed_pulse*2)/2
        Feed_pulse[Feed_pulse<5]=5
        DDx[i2]['Feed_pulse']=Feed_pulse.tolist()
    
        
    XX_th0,DIV=calculate_DIV(tt,XX0,uu,TH_param0,DDx,Cov_y,Cov_TH0)
    
    DOT_min=[]
    DIV_constrain=[]
    for i2 in range(uu[0][0]):
        dot_min=min(XX_th0['sample'][i2][3])
        DOT_min.append(dot_min)
        if dot_min<20:
            # FIM_crit=1e-11
            DIV_constrain.append((1+(20-dot_min)*10)*1e0)
        else:
            DIV_constrain.append(1)
    
    DIV_constr=np.array(DIV_constrain)
    print(ux,'',DIV/np.sum(DIV_constr),'constraint ',min(DOT_min))
    return DIV*(-1)/np.sum(DIV_constr)

# %%
def calculate_DIV(tt,XX0,uu,TH_param0,DD,Cov_y=[],Cov_TH0=[]):
    XX_th0=simulate_parallel(tt,XX0,uu,TH_param0,DD)    

    DIV=XX_th0['sample'][0][0][-1]  

            

    return XX_th0,DIV
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
            ts_sample_all=DD[i1]['time_sample'] 
            ts_sample=ts_sample_all[(ts_sample_all>ts[0]) & (ts_sample_all<=ts[1])]
            
            sample_interp=np.interp(ts_sample,ty[i1][:,0],ty[i1][:,i2+1])
            try:
                XX['sample'][i1][i2]=XX['sample'][i1][i2]+sample_interp.tolist() 
                
            except:
                XX['sample'][i1][i2]=sample_interp.tolist()
                
                
        ts_sensor_all=DD[i1]['time_sensor'] 
        ts_sensor=ts_sensor_all[(ts_sensor_all>ts[0]) & (ts_sensor_all<=ts[1])]
        sensor_interp=np.interp(ts_sensor,ty[i1][:,0],ty[i1][:,4])

        try:
            XX['sample'][i1][3]=XX['sample'][i1][3]+sensor_interp.tolist() 
        except:
            XX['sample'][i1][3]=sensor_interp.tolist()

    return XX

# %%
def simulate_interval(index_mbr,ts,XX,uu,TH_param,DD):

            
            u=[uu[index_mbr][1]]+[index_mbr]+[uu[index_mbr][0]]+[uu[index_mbr][2]]+[1]
            X = np.array(XX['state'][index_mbr])
            D = DD[index_mbr]
            t, y = function_simulation(ts, X, u, TH_param, D)

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

    return tt,yy.transpose()
# %%    
def odeFB(t,Xo,THo,u):

    X=Xo.copy()
    TH=THo.copy()
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

    
    Ys_ox=TH[4]
    Ya_p=TH[5]
    Ya_c=TH[6]
    Yo_ox=TH[8]
    Yo_a=TH[9]
    Yxs_of=TH[10]
    
    Ks=TH[11]
    n_ox=4
    
    Ka=TH[12]
    Ksi=TH[3]
    Kai=TH[7]
    Ko=0.1057
    
    kla=TH[16]
    k_sensor=TH[17]
    
    ky_1=TH[13] 
    ky_2=TH[14]
    ky_3=TH[15]
    
    DO_star=100
    H=13000#
    
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)
    q_ox_max=fracc_q_ox_max*qs_max
    
    q_ox_ss=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)
    qac_ss=qa_max*A/(A+Ka)*Kai/(Kai+S)
    b_ss=Ko+(q_ox_ss*Yo_ox+qac_ss*Yo_a)*Xv*H/kla-DO_star
    c_ss=-DO_star*Ko
    DOT_ss=(-b_ss+(b_ss*b_ss-4*c_ss)**.5)/2

    
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