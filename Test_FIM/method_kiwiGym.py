# %% Import
import numpy as np
# import json
import time
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

from copy import deepcopy

from scipy.optimize import shgo,dual_annealing,minimize,differential_evolution

import matplotlib.pyplot as plt
# %% 
# def optimizer_reference2(LB,UB,ux_ref,optim_options,tt,XX0,uu,TH_param0,DD,Cov_y):
   
#     ux_lb=np.array(LB)
#     ux_ub=np.array(UB)
#     ux_mean=np.linspace(LB[0],UB[0],len(LB))
    
#     bounds_ux=list(range(len(ux_lb)))
#     for i1 in range(len(ux_lb)):
#         bounds_ux[i1]=(ux_lb[i1],ux_ub[i1])
    
#     t_test0=time.time()    
#     e1=obj_fun(ux_lb,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y)
#     e2=obj_fun(ux_ub,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y)
#     e3=obj_fun(ux_mean,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y) 
#     t_test=(time.time()-t_test0)/3

#     n_fun_eval0=round(60*optim_options[0]/(t_test*1.2))
#     n_fun_eval1=round(60*optim_options[1]/(t_test*1.2))
    
#     print(n_fun_eval0,n_fun_eval1)

#     # TH_opt00_collect=np.vstack(((THmin+THmax)*.75/2,(THmin+THmax)/2,(THmin+THmax)*1.25/2))
#     # e_00_collect=np.vstack((e1,e2,e3))
    
    
#     ux_opt0 = dual_annealing(lambda ux: obj_fun(ux,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y), bounds=bounds_ux, maxfun=n_fun_eval0, no_local_search=True)
#     print('global opt ',ux_opt0.x)
#     ux_opt1 = minimize(lambda ux: obj_fun(ux,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y), ux_opt0.x, bounds=bounds_ux, method='Nelder-Mead', options={'maxfev': n_fun_eval1})

#     # ux_opt1 = minimize(lambda ux: obj_fun(ux,tt,XX0,uu,TH_param0,DD,Cov_y), ux_mean, bounds=bounds_ux, method='Nelder-Mead', options={'maxfev': n_fun_eval1})
#     print('local opt ',ux_opt1.x)
#     return ux_opt1        

# # %%

# def obj_fun2(ux,ux_ref,tt,XX0,uu,TH_param0,DD,Cov_y):
#     DDx=deepcopy(DD)
    
#     # DD[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':(5+i*1+np.zeros(len(time_pulses.tolist()))).tolist(),
#     lenght_ux=int(len(ux)/uu[0][0])
#     for i2 in range(uu[0][0]):
#         t_pulse=np.array(DDx[i2]['time_pulse'])
#         DD_ref=(36.33)*ux_ref[i2]*np.exp(ux_ref[i2]*(t_pulse-t_pulse[0]))+ux[np.arange(lenght_ux)+i2*lenght_ux]
#         DD_ref[DD_ref<0]=0
#         DDx[i2]['Feed_pulse']=DD_ref.tolist()
#         print(DDx[i2]['Feed_pulse'])
        
#     # print('Test DD  ',np.array(DDx[0]['Feed_pulse'])-np.array(DDx[1]['Feed_pulse']))
#     Si,Q,FIM,XX_th0,traceFIM,FIM_crit=calculate_FIM(tt,XX0,uu,TH_param0,DDx,Cov_y)
    
    
    
#     print(ux,'',FIM_crit/1e6)
#     return FIM_crit*(-1)

 

# # %%

# def obj_fun(ux,tt,XX0,uu,TH_param0,DD,Cov_y):
#     DDx=deepcopy(DD)
    
#     # DD[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':(5+i*1+np.zeros(len(time_pulses.tolist()))).tolist(),
    
#     for i2 in range(uu[0][0]):
#         t_pulse=np.array(DDx[i2]['time_pulse'])
#         Feed_pulse=(36.33)*ux[i2]*np.exp(ux[i2]*(t_pulse-t_pulse[0]))
#         Feed_pulse[t_pulse>=uu[i2][2]]=(36.33)*ux[i2]*np.exp(ux[i2]*(uu[i2][2]-t_pulse[0]))
#         Feed_pulse=np.round(Feed_pulse*2)/2
#         Feed_pulse[Feed_pulse<5]=5
#         DDx[i2]['Feed_pulse']=Feed_pulse.tolist()
    
        
#     # print('Test DD  ',np.array(DDx[0]['Feed_pulse'])-np.array(DDx[1]['Feed_pulse']))
#     Si,Q,FIM,XX_th0,traceFIM,FIM_crit=calculate_FIM(tt,XX0,uu,TH_param0,DDx,Cov_y)
    
#     DOT_min=[]
#     FIM_constrain=[]
#     for i2 in range(uu[0][0]):
#         dot_min=min(XX_th0['sample'][i2][3])
#         DOT_min.append(dot_min)
#         if dot_min<20:
#             FIM_constrain.append(1+(20-dot_min)*10)
#         else:
#             FIM_constrain.append(1)

            
#     FIM_constr=np.array(FIM_constrain)
#     print(ux,'',FIM_crit*3/np.sum(FIM_constr),'constraint ',min(DOT_min))
#     return FIM_crit*(-1)*3/np.sum(FIM_constr)


# %%
def calculate_FIM(tt,XX0,uu,TH_param0,DD,Cov_y=[]):
    print('calculating reward...')
    XX_th0=simulate_parallel(tt,XX0,uu,TH_param0,DD)

    Si={}
    for i1 in range(len(TH_param0)):# range(16):#range(len(TH_param0)):

        TH_param=TH_param0.copy()
        TH_param[i1]=TH_param0[i1]*(1+1e-3)

        t_check1=time.time()
        XX_th=simulate_parallel(tt,XX0,uu,TH_param,DD)
           
        t_check2=time.time()
        Si[i1]={}
        # print(i1)
        for i2 in range(uu[0][0]):
            Si[i1][i2]={}
            for i3 in range(5):
                Si[i1][i2][i3]=(np.array(XX_th['sample'][i2][i3])-np.array(XX_th0['sample'][i2][i3]))/(TH_param[i1]-TH_param0[i1]+1e-9)*TH_param0[i1]
                # Si[i1][i2][i3]=(np.array(XX_th['sample'][i2][i3])-np.array(XX_th0['sample'][i2][i3]))/(TH_param[i1]-TH_param0[i1]+1e-9)*TH_param0[i1]/np.array(XX_th0['sample'][i2][i3])

        
    Q={}       
    for i1 in range(uu[0][0]): 
        for i2 in range(len(TH_param)):# range(16):#
                for i3 in range(5):
                    q1=Si[i2][i1][i3]
                    q2=q1[:,None]
                    if i3==0:
                        q3=q2
                    else:
                        q3=np.vstack((q3,q2))
                
                if i2==0:
                    q4=q3
                else:
                    q4=np.hstack((q4,q3))
                    
        Qi=q4
        FIM_left_i=Qi.transpose()@ np.linalg.inv(Cov_y)
        FIM_i=FIM_left_i@Qi

        Q[i1]=Qi            
        if i1==0:
            FIM=FIM_i
        else:
            FIM=FIM+FIM_i            
    
    print(FIM.shape,Cov_y.shape)   
    

    
    # traceFIM_dim=np.linalg.det(FIM)
    
    traceFIM=np.trace(FIM)
    ei,ev=np.linalg.eig(FIM)
    FIM_crit=min(ei)
    # FIM_crit=traceFIM
    # FIM_crit=np.linalg.det(FIM)
    
    return Si,Q,FIM,XX_th0,traceFIM,FIM_crit
# %%

def calculate_sensitivity(tt,XX0,uu,TH_param0,DD):
    
    tspan=[tt[0],tt[-1]]
    XX_th0=simulate_parallel(tt,XX0,uu,TH_param0,DD)

    
    s_concat=np.array([])
    for i in range(uu[0][0]*0+1):
        So=np.zeros([len(TH_param0),len(XX_th0['sample'][0])+1])
        So=So.flatten()
        
        
        u=[uu[i][1]]+[i]+[uu[i][0]]+[uu[i][2]]+[1]

        # print(odeSen(0,So,TH_param0,u,DD,XX_th0,i))
        # print("H")
        
        
        sol=solve_ivp(lambda t,s: odeSen(t,s,TH_param0,u,DD,XX_th0,i) ,tspan,So,method="RK23")#, rtol=1e-3)
        # print(sol.y)
        # print(len(sol.y))
        # for i in range(len(XX_th0['sample'][0])):
        #     if i==3:
        #         s_concat=np.vstack(s_concat,np.interp(DD[i]['time_sensor'],sol.t,sol.y))
        #     else:
        #         s_concat=np.vstack(s_concat,np.interp(DD[i]['time_sample'],sol.t,sol.y))
                
        # s_concat=np.concat(s_concat,sol.y.flatten())
        

    return sol.t,sol.y
    
# %%    
def odeSen(t,So,THo,u,DD,XX,index_mbr):
    Xo=np.zeros(len(XX['sample'][0])+1)
    print(t)
    for i in range(len(XX['sample'][0])):
        if i==3:
            Xo[i]=np.interp(t,DD[index_mbr]['time_sensor'],XX['sample'][index_mbr][i])
        else:
            Xo[i]=np.interp(t,DD[index_mbr]['time_sample'],XX['sample'][index_mbr][i])
    Xo[-1]=0.01
    
    f_ode=odeFB(t,Xo,THo,u)

    dS=np.zeros([len(THo),len(Xo)])
    So=So.reshape([len(THo),len(Xo)])
    for i in range(len(THo)):
        TH_plus=THo.copy()
        TH_plus[i]=TH_plus[i]*1.01
        f_ode_th=odeFB(t,Xo,TH_plus,u)
        df_th=(f_ode_th-f_ode)/(TH_plus[i]*0.01)

        for j in range(len(Xo)):
            X_plus=Xo.copy()
            X_plus[j]=X_plus[j]*1.01
            f_ode_y=odeFB(t,X_plus,THo,u)          
            df_y=(f_ode_y-f_ode)/(X_plus[j]*0.01)
            
            dS[i,j]=df_th[j]+df_y[j]*So[i,j]
    

    return dS.flatten()
    
# %%

def simulate_parallel(ts,XX0,uu,TH_param,DD):  
    XX=deepcopy(XX0)
    brxtor_list=np.arange(uu[0][0]).tolist()
    
    ty={}
    
    results = Parallel(n_jobs=-1)(
        delayed(simulate_interval)(i1, ts,XX,uu,TH_param,DD)
        for i1 in brxtor_list
    )
    

    for i1, result in zip(brxtor_list, results):
        ty[i1] = result
        
    # for i1 in brxtor_list: 
    #     ty[i1]=simulate_interval(i1, ts,XX,uu,TH_param,DD)

        
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
    
    
    f_ode=rates(t,X,TH,u)
    
    dXv=f_ode[0]
    dS=f_ode[1]
    dA=f_ode[2]
    dDOT=f_ode[3]
    dP=f_ode[4]
    dV=0
    
    dX=np.array([dXv,dS,dA,dDOT,dP,dV])
    return dX
# %%  
def rates(t,X,TH,u):
    X[X<0]=0
    
    Xv=X[0]
    S=X[1]
    A=X[2]
    DOT=X[3] #check order A and DOT
    P=X[4]
    V=X[5]
    if DOT>100:
        DOT=DOT*0+100
    
        
    qs_max=TH[0]
    fracc_q_ox_max=TH[1]
    qa_max=TH[2]
    b_prod=TH[3]
    
    
    Ys_ox=TH[4]
    Ya_p=TH[5]
    Ya_c=TH[6]
    Yp=TH[7]
    Yo_ox=TH[8]
    Yo_a=TH[9]
    Yxs_of=TH[10]
    
    Ks=TH[11]
    n_ox=4
    
    Ka=TH[12]
    Ksi=TH[13]
    Kai=TH[14]
    Ko=TH[15]
    
    kla=TH[16]
    k_sensor=TH[17]
    
    DO_star=100
    H=13000#
    
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)#*(1-P/70)
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
        
    q_prod=(Yp*s_prod+b_prod)*mu


    f_ode=[(mu)*Xv,-(qs)*Xv,qap*Xv-qac*Xv,k_sensor*(DOT_ss-DOT),q_prod*Xv]
    return f_ode
# %%     
def intM(ts0,Xo0,u0,TH0):    

    tspan=np.array([ts0[0],ts0[-1]])
    Xo1=Xo0.tolist().copy()



    sol=solve_ivp(lambda t,y: odeFB(t,y,TH0,u0) ,tspan,Xo1,method="BDF", rtol=1e-5, atol=1e-5,t_eval=ts0)
    y_interm=sol.y
    y_interm[y_interm<0]=0
    y_return=y_interm.copy()

    return sol.t,y_return