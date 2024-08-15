# from torchvision.datasets import ImageFolder

# dataset = ImageFolder("../Data/WebFace260M")





# import numpy as np
# a = np.arange(1000)
# b = np.zeros(1000)
# c = np.zeros(1000)
# d = np.zeros(1000)

# alpha =0.001
# for i in range(1, a.shape[0]):
#     b[i] = alpha*a[i]+(1-alpha)*b[i-1]
#     c[i] = b[i-1]-b[i]
#     d[i] = alpha*b[i-1]-alpha*a[i]
    
# e = np.zeros(1000)
# e[1] = alpha*b[0] - alpha*a[1]
# for i in range(1, a.shape[0]):
#     e[i] = (1-alpha)*e[i-1]+alpha*(a[i-1]-a[i])
    
# f = np.zeros(1000)
# f[1] = e[1]
# f[2] = e[2]
# for i in range(2, a.shape[0]):
#     f[i] = alpha*(a[i-2]-a[i])+(1-alpha)*f[i-2]-alpha*f[i-1]





a = "ms1mv3"
if a == "webface12M":
    from configs.webface12m import cfg
elif a == "ms1mv3":
    from configs.ms1mv3 import cfg
    
print(cfg.output)





