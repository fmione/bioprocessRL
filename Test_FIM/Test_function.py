import numpy as np

import math

import matplotlib.pyplot as plt
import time

from copy import deepcopy
import method_kiwiGym as method_kiwiGym

# import method_kiwiGymJAX as method_kiwiGym

# import method_kiwiGymDOE as method_kiwiGym

# %% Create design
n_exp=3

t_final=16

ts=np.array([0,1])
time_pulses=np.arange(5+5/60,t_final,10/60)


XX0={'state':{},'sample':{}}
uu={}
DD={}

# sample_schedule=[0.33,0.33,0.66,0.66,0.99,0.99]
# sample_schedule=[.99]*n_exp
sample_schedule=[0.33,0.66,0.99]

ux=[0.3]*n_exp#
# ux=[0.3, 0.3]*1#
ux=np.linspace(0.15,0.3,n_exp)#
ux=np.array([0.1       , 0.13029815,  0.15251801   ])

# ux=np.array([0.10456608, 0.11103395, 0.1142404 ])

# ux=np.array([0.1171261,  0.11423985, 0.11366176])#
for i in range(n_exp):
    XX0['t']=ts[0]
    XX0['state'][i]=[0.18,4,0,100,0,.01]
    XX0['sample'][i]={}
    uu[i]=[n_exp,200,10]
    # feed_profile_i=(5+i*0+np.zeros(len(time_pulses.tolist()))).tolist()
    # feed_profile_i=((36.33)*ux[i]*np.exp(ux[i]*(time_pulses-time_pulses[0]))).tolist()
    

    feed_profile_i=(36.33)*ux[i]*np.exp(ux[i]*(time_pulses-time_pulses[0]))
    feed_profile_i[time_pulses>=uu[i][2]]=(36.33)*ux[i]*np.exp(ux[i]*(uu[i][2]-time_pulses[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2
    feed_profile_i[feed_profile_i<5]=5

    
    DD[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_schedule[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,25*round(t_final))}
    

TH_param=np.array([1.2578, 0.43041, 0.6439,  2.2048,  0.4063,  0.1143,  0.1848,    287.74,    1.586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[850]*n_exp+[90]*n_exp)
# t_check1=time.time()



######### ESTAN TH CON kl y kp
# %% 
t_check1=time.time()
n_sample=len(DD[0]['time_sample'])
n_sensor=len(DD[0]['time_sensor'])
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[20]*n_sample)*1)  #*n_exp
C2=np.diag(sd_meas**2)

# XX=method_kiwiGym.simulate_parallel(np.array([0,t_final]),XX0,uu,TH_param,DD)
Si,Q,FIM,XX,traceFIM,FIM_crit=method_kiwiGym.calculate_FIM(np.array([0,t_final]),XX0,uu,TH_param,DD,C2)

# TH_sd=(np.diag(np.linalg.inv(FIM)))**.5
# CV=TH_sd/TH_param[0:16]
# CV=TH_sd#[0:16]


# print(ux)
print('FIM crit ',FIM_crit)
# print(CV*100)
t_check2=time.time()
print(t_check2-t_check1)
# %% 
# LB=[0.1]*n_exp
# UB=[0.2]*n_exp
# optim_options=[10, 10]
# u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),XX0,uu,TH_param,DD,C2)
# # %%
# ux_ref=np.linspace(0.15,0.3,n_exp)#
# LB=[-5]*n_exp*len(time_pulses.tolist())
# UB=[5]*n_exp*len(time_pulses.tolist())
# optim_options=[5, 5]
# u_opt=method_kiwiGym.optimizer_reference2(LB,UB,ux_ref,optim_options,np.array([0,t_final]),XX0,uu,TH_param,DD,C2)

# %% Run function
# t_check1=time.time()
# TH_param_cop=TH_param.copy()    
# # TH_param_cop[4:7]=TH_param_cop[4:7]*.01
# while ts[-1]<=  t_final:
#     XX0=method_kiwiGym.simulate_parallel(ts,XX0,uu,TH_param,DD)

#     print(ts,XX0['state'][0][0],)
#     ts[0]=ts[-1]
#     ts[-1]+=1
# t_check2=time.time()
# print(t_check2-t_check1)
# t=DD[0]['time_sensor']
# y=XX0['sample'][5][3]