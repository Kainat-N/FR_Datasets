import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("Output/WebFace12M_epoch5_r50/loss.csv")
c = a["lr"].values
y = a["loss"].values
n = y.shape[0]
x = np.arange(n)

# plt.figure()
# plt.plot(c)
# plt.figure()
# plt.plot(y)






# alpha = 0.999
# beta = 0.999
# tolerance = len(y)*0.015
# ema = [y[0]]
# ema.append(alpha*ema[0]+(1-alpha)*y[1])
# diff = [ema[0] - ema[1]]
# hma = [diff[0]]
# for i in range(2, n):
#     ema.append(alpha*ema[i-1]+(1-alpha)*y[i])
#     diff.append(ema[i-1] - ema[i])
#     hma.append(beta*hma[i-2]+(1-beta)*diff[i-1])

# plt.figure()
# plt.plot(y[:])
# plt.figure()
# plt.plot(ema[:])
# plt.figure()
# plt.plot(diff[:])
# plt.figure()
# plt.plot(hma[:])
# plt.hlines([0.00005], [0], [len(hma)])
# plt.hlines([0], [0], [tolerance])





alpha = 0.001
beta = 0.001
toler = len(y)*0.05
thres = 0.00005
count = 0
counts = []
d = [alpha*y[0] - alpha*y[1]]
dema = [d[0]]
for i in range(1, a.shape[0]):
    if i%1000==0:
        print(i)
        
    d.append((1-alpha)*d[i-1]+alpha*(y[i-1]-y[i]))
    dema.append(beta*d[i]+(1-beta)*dema[i-1])
    
    counts.append(0)
    if dema[i] < thres:
        if count < toler:
            count += 1
            counts[-1] = 0
        else:
            count = 0
            counts[-1] = 0.001

plt.figure()
plt.plot(y[:])
plt.figure()
plt.plot(d[:])
plt.figure()
plt.plot(dema[:])
plt.plot(counts)
plt.hlines([thres], [0], [len(dema)])
plt.hlines([0], [0], [toler])




# m = 4000
# k = np.ones(m)
# avg = np.convolve(y, k, mode="same") / m
# # k[:int(m/2)] = -1
# k1 = np.array([-1,1])
# diff = np.convolve(avg, k1, mode="same")
# y = diff[m+1:]
# ema = [y[0]]
# alpha = 0.999
# for i in range(1, y.shape[0]):
#     ema.append(alpha*ema[i-1]+(1-alpha)*y[i])

# plt.figure()
# plt.plot(avg[m+1:])

# plt.figure()
# plt.plot(diff[m+1:])

# plt.figure()
# plt.plot(ema)



# m = 4000
# k1 = np.ones(m)
# k2 = np.ones(m)
# k2[int(m/2):] = -1
# ss = np.zeros(m)
# avg = []
# diff = []
# sig = []
# w_count = []
# count = 0
# for i in range(n):
#     print(i)
#     if i < m:
#         ss[i] = y[i]
#         continue
    
#     ss[:-1] = ss[1:]
#     ss[-1] = y[i]
#     avg.append(np.sum(ss*k1)/m)
    
#     if i < 2*m:
#         continue
    
#     s_avg = avg[(i-2*m):(i-m)]
#     diff.append(np.sum(s_avg*k2)/m)
    
#     if diff[-1] < 0.06:
#         count += 1
        
#         if count > 2*m:
            
#             sig.append(1)
#             count = 0
#             w_count.append(w_count[-1]+1)
#         else:
#             sig.append(0)
#             w_count.append(w_count[-1])
            
#         if w_count[-1] > 2:
#             sig[-1] = 2
#             w_count[-1] = 0
#     else:
#         count = 0
#         sig.append(0)
#         w_count.append(0)
        
# plt.figure()
# plt.plot(avg)

# plt.figure()
# plt.plot(diff)
# plt.plot(sig)
# plt.plot(w_count)
# plt.hlines([0.06], [0], [len(avg)])
# plt.hlines([0], [0], [4000])




