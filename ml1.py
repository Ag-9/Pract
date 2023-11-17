import numpy as np
def GDA(start,gradient,l_rate,iteration,tol=0.01):
    steps=[start]
    X=start
    
    for i in range(iteration):
        difference=l_rate*gradient(X)
        if np.abs(difference)<tol:
            break
        X=X-difference
        steps.append(X)
    return steps,l_rate,X,len(steps)

def gradient_fun(X):
    return 2*(X+3)

history,l_rate,result,steps=GDA(2, gradient_fun, 0.1, 100)
print("Steps in GDA", history)
print("Learning rate is ", l_rate)
print("Number of steps required to reach local minima", steps)
print("Local Minima", result)
