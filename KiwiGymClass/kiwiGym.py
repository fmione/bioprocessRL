# %% Import
import numpy as np
from copy import deepcopy

import method_kiwiGym
# %%
class kiwiGym:
    def __init__(self,time_current=0,number_mbr=3,time_final=16,time_step=1,sample_schedule=[0.33,0.66,0.99],time_batch=5,mu_reference=[0.2464, 0.2666, 0.2958],TH_param0=[]):
        self.number_mbr=number_mbr
    
        self.time_final=16
        self.time_current=time_current
        self.time_step=time_step
        self.time_interval=np.array([time_current,time_current+self.time_step])
        self.time_pulses=np.arange(time_batch+5/60,time_final,10/60)
    
    
        XX0={'state':{},'sample':{}}
        uu={}
        DD={}
    
        self.sample_schedule=sample_schedule
    
        self.mu_reference=np.array(mu_reference)
    
        
        for i in range(self.number_mbr):
            XX0['t']=self.time_interval[0]
            XX0['state'][i]=[0.18,4,0,100,0,.01]
            XX0['sample'][i]={0:[],1:[],2:[],3:[],4:[],}
            uu[i]=[self.number_mbr,200,10]
            
            feed_profile_i=((36.33)*self.mu_reference[i]*np.exp(self.mu_reference[i]*(self.time_pulses-self.time_pulses[0]))).tolist()
            
            DD[i]={'time_pulse':self.time_pulses.tolist(),'Feed_pulse':feed_profile_i,'time_sample':np.arange(self.time_final)+self.sample_schedule[i],#np.arange(8,16.1,8),#
                   'time_sensor':np.linspace(0,self.time_final,25*round(self.time_final))}
            
        if len(TH_param0)==0:
            self.TH_param=np.array([1.2578, 0.43041, 0.6439,  2.2048,  0.5063,  0.1143,  0.1848,    287.74,    1.2586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[750]*self.number_mbr+[90]*self.number_mbr)
        else:
            self.TH_param=np.array(TH_param0)
        self.XX0=deepcopy(XX0)
        self.XX=deepcopy(XX0)
        self.uu=uu
        self.DD=DD
        self.done=0
        return
    
    def reset(self):
        self.time_current=0
        self.time_interval=np.array([self.time_current,self.time_current+self.time_step])
        
        XX0={'state':{},'sample':{}}
        for i in range(self.number_mbr):
            XX0['t']=self.time_interval[0]
            XX0['state'][i]=[0.18,4,0,100,0,.01]
            XX0['sample'][i]={0:[],1:[],2:[],3:[],4:[],}
        self.XX=deepcopy(XX0)
        self.done=0
        return 
    
    def step(self,action=[]):

        XX_plus1=method_kiwiGym.simulate_parallel(self.time_interval,self.XX,self.uu,self.TH_param,self.DD)
        self.XX=XX_plus1
        self.time_current=self.time_interval[1]
        self.time_interval=np.array([self.time_current,self.time_current+self.time_step])
        
        if self.time_current>=self.time_final:
            self.done=1
            n_sample=len(self.DD[0]['time_sample'])
            n_sensor=len(self.DD[0]['time_sensor'])
            
            sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[20]*n_sample)*1)  #*n_exp
            C2=np.diag(sd_meas**2)
            
            Si,Q,FIM,XX,traceFIM,FIM_crit=method_kiwiGym.calculate_FIM(np.array([0,self.time_final]),self.XX0,self.uu,self.TH_param,self.DD,C2)
            
            self.reward=FIM_crit
        else:
            self.done=0
            self.reward=0
            
        return
    #     return obs,reward,terminate