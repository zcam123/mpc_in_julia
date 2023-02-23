import juliacall
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rand

##test Kalman filter
jl.include('md_kf.jl')

# x_est = jl.zeros(4)

# z = np.array(1)
# P = np.eye(4)*15
# P_ = jl.Array(P)

#x_est, P = jl.KF_est(jl.Array(z), P_, x_est)


##test mpc
jl.include("mpc_called.jl")


# pred = 25
# sample = 200
# zref = np.array([0.1 + 0.08*np.sin(i/(20*sample)) for i in range(25*sample)])

# zref_ = jl.Array(zref)

# results = jl.mpc(jl.zeros(4), zref_, nu=2, u_clamp=jl.nothing, sample=sample)

# print(results)


A = np.array([
    [1, -6.66e-13, -2.03e-9, -4.14e-6],
    [9.83e-4, 1, -4.09e-8, -8.32e-5],
    [4.83e-7, 9.83e-4, 1, -5.34e-4],
    [1.58e-10, 4.83e-7, 9.83e-4, .9994]
])

B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T

C = np.array([[-.0096, .0135, .005, -.0095]])

def y_to_z(y):
    return np.exp(61.4*y - 5.468)

##sample control loop
steps = 100
sample = 80
zref = [0.1 + 0.08*np.sin(i/(20*sample)) for i in range(25*sample)]
for _ in range(steps*sample - len(zref)):
    zref.append(zref[-1])
plot_zDs = zref

x = np.zeros(4)
x_est = np.zeros(4)
zs = []
us = []
P = np.eye(4)*15
R = 0.0001
for i in range(steps):
    #kf part
    #z = np.array(C@x + 1**(rand.randint(2)) * rand.randint(5)/100 * C@x) #lfp value with noise +/- 0-4% error
    z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise added to measurement of lfp
    x_est, P = jl.only_update_KF(jl.Array(z), jl.Array(P), jl.Array(x_est))

    #mpc part - now using KF estimate's x instead of the actual state
    optimal_u = jl.mpc(jl.Array(x_est), jl.Array( np.array(zref) ), nu=1, u_clamp=jl.nothing, sample=sample)
    us.append(optimal_u)
    for i in range(sample):
        x_est, P = jl.KF_est(jl.Array(z), jl.Array(P), jl.Array(x_est), optimal_u) #kf
        x = A@x + B@optimal_u
        zs.append(y_to_z(C@x))
        #KF
        z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise
        

    #shift reference
    zref = zref[sample:] #AHHHH this was the problem most likely

#plot results
time_ax = [i for i in range(steps*sample)]

# plt_us = []
# for i in range(steps):
#     for j in range(sample):
#         plt_us.append(us[i])

plt.plot(time_ax, zs, plot_zDs)
#plt.ylim(0, 3)
plt.show()





# ######
# #trial to see if KF is working on its own
# sample = 200
# P = np.eye(4)*15
# R=10**(-4)
# x = np.zeros(4)
# x_est = np.zeros(4)

# #apply a constant input for one control period and see how off Kalman filter gets
# u_const = [0.5]
# z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise added to measurement of lfp

# for i in range(sample):
#     x = A@x + B@u_const
#     x_est, P = jl.KF_est(jl.Array(z), jl.Array(P), jl.Array(x_est), u_const[0])

# print("\n", x, "\n", x_est)

# error = [(x[i] - x_est[0][i])/x[i] * 100 for i in range(4)]
# print("\n", error)

