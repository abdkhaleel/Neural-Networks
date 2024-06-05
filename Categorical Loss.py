from math import log

#Method 1

Softmax_outputs = [0.7, 0.3, 0.2]
Target_outputs = [1, 0, 0]

loss = -(log(Softmax_outputs[0])*Target_outputs[0] + 
         log(Softmax_outputs[1])*Target_outputs[1] + 
         log(Softmax_outputs[2])*Target_outputs[2])

print(loss)

#Method 2

loss = 0
for i,j in zip(Softmax_outputs, Target_outputs):
    loss += log(i)*j
    
loss = -1*loss

print(loss)