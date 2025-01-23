from kiwiGym import kiwiGym
import time
# %%
KW=kiwiGym()
print('time: ',KW.time_current,' biomass (mbr 0): ',KW.XX['state'][0][0],' done: ',KW.terminated)
print()
time1=time.time()
for i in range(16):
    # KW.perform_action([0]*198)
    KW.perform_action([0,0,0])
    print('time: ',KW.time_current,' biomass (mbr 0): ',KW.XX['state'][0][0],' done: ',KW.terminated)
    print()
    
print('reward: ',KW.reward)

time2=time.time()
print(time2-time1)
