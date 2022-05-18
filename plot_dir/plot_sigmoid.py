import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def sig(x):
    ret = 1 / (1 + math.exp(-x))
    return ret

def der_sigmoid(x):
    a = []
    for item in x:
        a.append(sig(item)*(1-sig(item)))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10., 10., 0.2)
sig = der_sigmoid(x)
plt.plot(x,sig)

plt.title(f"Derivace logistické funkce")
plt.ylabel("Y")
plt.xlabel("X")
plt.savefig(f"kap_5_sigmoid_der.pdf")

plt.show()
