from kiwiGym import kiwiGym
import time
# %%
KW=kiwiGym()
print('time: ',KW.time_current,' biomass (mbr 0): ',KW.XX['state'][0][0],' done: ',KW.done)
print()
# time1=time.time()
for i in range(16):
    KW.step()
    print('time: ',KW.time_current,' biomass (mbr 0): ',KW.XX['state'][0][0],' done: ',KW.done)
    print()
    
print('reward: ',KW.reward)

# time2=time.time()
# print(time2-time1)
