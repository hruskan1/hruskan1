import template
import numpy as np
# lin = template.HiddenLinearLayer()
# x = np.array([[0.1,0.2],[0.5,0.6],[0.7,0.8]])
# w = np.arange(1,11).reshape(2,5)

# b = np.arange(-1,1.5,0.5)
# print(x)
# print(w)
# print(b)
# y = lin.forward(x,w,b)
# print(y)
# dy = np.zeros_like(y)
# dy[0:2,0] = 1
# print("+++++")
# print(dy)
# dx,dw,db = lin.backward(dy)
# print(dx)
# print(dw)
# print(db)

sigmoloss = template.SigmoidAndNLLossLayer()
x = np.arange(-3,3).reshape(-1,1)
print(x)
y = np.zeros_like(x)
y[int(y.size/2):] = 1.
print(y)
z = sigmoloss.forward(x,y)
print(z)
dx = sigmoloss.backward()
print(dx)