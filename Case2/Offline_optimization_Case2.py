import numpy as np
import time
import method_kiwiGym

# %% Create design
n_exp=3

t_final=14

ts=np.array([0,1])
time_pulses=np.arange(5+5/60,t_final,10/60)


XX0={'state':{},'sample':{}}
uu={}
DD={}


sample_schedule=[0.33,0.66,0.99]

ux=np.array([0.12846724, 0.14790724, 0.07986599 ])


for i in range(n_exp):
    XX0['t']=ts[0]
    XX0['state'][i]=[0.18,4,0,100,0,.0]
    XX0['sample'][i]={}
    uu[i]=[n_exp,200,10]


    feed_profile_i=(32.406)*ux[i]*np.exp(ux[i]*(time_pulses-time_pulses[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2
    feed_profile_i[feed_profile_i<5]=5

    
    DD[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_schedule[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,25*round(t_final))}
    

TH_param=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*n_exp+[90]*n_exp)

# %% 
t_check1=time.time()
n_sample=len(DD[0]['time_sample'])
n_sensor=len(DD[0]['time_sensor'])
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[50]*n_sample)*1)  
C2=np.diag(sd_meas**2)

XX,DIV,DIV_min=method_kiwiGym.calculate_DIV(np.array([0,t_final]),XX0,uu,TH_param,DD,C2)


print(ux)
print('DIV ',DIV,DIV_min)

t_check2=time.time()
print(t_check2-t_check1)

# %% 
LB=[0.075]*n_exp
UB=[0.15]*n_exp
optim_options=[25, 5]
u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),XX0,uu,TH_param,DD,C2)

print('Optimal mu_set: ',u_opt)
