import numpy as np
import time
import method_kiwiGym as method_kiwiGym

# %% Create design
n_exp=3
t_final=14

ts=np.array([0,1])
time_pulses=np.arange(5+5/60,t_final,10/60)

XX0={'state':{},'sample':{}}
uu={}
DD={}
DDj={}

sample_schedule=[0.33,0.66,0.99]


ux=np.array([0.14529732, 0.075    ,  0.11614164])
 
for i in range(n_exp):
    XX0['t']=ts[0]
    XX0['state'][i]=[0.18,4,0,100,0,.01*0]
    XX0['sample'][i]={}
    uu[i]=[n_exp,200,10]

    feed_profile_i=(36.33*0+32.406)*ux[i]*np.exp(ux[i]*(time_pulses-time_pulses[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2
    feed_profile_i[feed_profile_i<5]=5

    
    DD[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_schedule[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,2*25*round(t_final))}
    DDj[i]={'time_pulse':time_pulses.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_schedule[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,2*25*round(t_final))}    

TH_param=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*n_exp+[90]*n_exp)
# %% 

n_sample=len(DD[0]['time_sample'])
n_sensor=len(DD[0]['time_sensor'])
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.2]*n_sample+[5]*n_sensor+[50]*n_sample)*1)  #*n_exp
C2=np.diag(sd_meas**2)


sd_measj=np.array(([.2]*n_sample+[.2]*n_sample+[.2]*n_sample+[5]*len(DDj[0]['time_sensor'])+[50]*n_sample)*1)  #*n_exp
C2j=np.diag(sd_measj**2)

THsd0=TH_param[0:18]*0+.1
THsd0[13]=.5
THsd0[14]=.5
THsd0[15]=.5
THsd0[16]=.5
THsd0[17]=.5
Cov_TH0=np.diag(THsd0**2)


t_check1=time.time()
Si,Q,FIM,XX,traceFIM,FIM_crit,ei=method_kiwiGym.calculate_FIM(np.array([0,t_final]),XX0,uu,TH_param,DD,C2,Cov_TH0)
t_check2=time.time()
print(t_check2-t_check1)
print('FIM crit ',FIM_crit)
TH_sd=(np.diag(np.linalg.inv(FIM)))**.5


# %% 
LB=[0.075]*n_exp
UB=[0.15]*n_exp

optim_options=[50, 10]
u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),XX0,uu,TH_param,DD,C2,Cov_TH0)

print('Optimal mu_set: ',u_opt)