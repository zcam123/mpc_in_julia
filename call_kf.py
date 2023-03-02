import juliacall
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rand

##test Kalman filter
jl.include('md_kf.jl')

##test mpc
jl.include("mpc_called.jl")

#first parameters
# A = np.array([
#     [1, -6.66e-13, -2.03e-9, -4.14e-6],
#     [9.83e-4, 1, -4.09e-8, -8.32e-5],
#     [4.83e-7, 9.83e-4, 1, -5.34e-4],
#     [1.58e-10, 4.83e-7, 9.83e-4, .9994]
# ])

# B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T

# C = np.array([[-.0096, .0135, .005, -.0095]])

# #new parameters for hippocampus model
A = np.array([[ 0.99742761, -0.04690783,  0.11240742,  0.10926326],
        [ 0.03127654,  0.96165808, -0.1067485 , -0.03791996],
        [-0.01651622,  0.04731647,  0.96798153,  0.45721781],
        [ 0.00719529, -0.01325746, -0.2831643 ,  0.83708282]])
B = np.array([[ 0.00076718],
        [-0.00194198],
        [ 0.00194759],
        [-0.00080658]])
C = np.array([[-0.25342973, -0.20129639, -0.0655937 ,  0.13362195]])

def y_to_z(y):
    return np.exp(61.4*y - 5.468)
def z_to_y(z):
    return (np.log(z) + 5.468)/61.4

##sample control loop
steps = 50
sample = 10

zref = [0.1 + 0.08*np.sin(i/(20*sample)) for i in range(25*sample)]
for _ in range(steps*sample - len(zref)):
    zref.append(zref[-1])
plot_zDs = zref #store initial reference now before it gets initial values chopped off during below loop

#yref = z_to_y(zref)
from_file = np.load("tklfp.npy")
yref = [elem for elem in from_file]
for _ in range(steps*sample - len(yref)):
    yref.append(yref[-1])
plot_yDs = yref #store initial reference now before it gets initial values chopped off during below loop


x = np.zeros(4)
x_est = np.zeros(4)
zs = []
us = []
P = np.eye(4)*15
R = 10**(-7)

# *Here is a loop for firing rate
# for i in range(steps):
#     #kf part
#     #z = np.array(C@x + 1**(rand.randint(2)) * rand.randint(5)/100 * C@x) #lfp value with noise +/- 0-4% error
#     z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise added to measurement of lfp
#     x_est, P = jl.only_update_KF(jl.Array(z), jl.Array(P), jl.Array(x_est))

#     #mpc part - now using KF estimate's x instead of the actual state
#     optimal_u = jl.flex_mpc(jl.Array(x_est), jl.Array( np.array(zref) ), nu=1, u_clamp=jl.nothing, sample=sample, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C), ref_type=1)
#     us.append(optimal_u)
#     for i in range(sample):
#         x_est, P = jl.KF_est(jl.Array(z), jl.Array(P), jl.Array(x_est), optimal_u, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C)) #kf
#         x = A@x + B@optimal_u
#         zs.append(y_to_z(C@x))
#         #KF
#         z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise
        

#     #shift reference
#     zref = zref[sample:] 

#* Here is a loop for lfp values
ys = []
for i in range(steps):
    #kf part
    #z = np.array(C@x + 1**(rand.randint(2)) * rand.randint(5)/100 * C@x) #lfp value with noise +/- 0-4% error
    z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise added to measurement of lfp
    x_est, P = jl.only_update_KF(jl.Array(z), jl.Array(P), jl.Array(x_est))

    #mpc part - now using KF estimate's x instead of the actual state
    optimal_u = jl.flex_mpc(jl.Array(x_est), jl.Array( np.array(yref) ), nu=1, u_clamp=jl.nothing, sample=sample, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C), ref_type=2)
    us.append(optimal_u)
    for i in range(sample):
        x_est, P = jl.KF_est(jl.Array(z), jl.Array(P), jl.Array(x_est), optimal_u, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C)) #kf
        x = A@x + B@optimal_u
        ys.append(C@x)
        #KF
        z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise
        

    #shift reference
    yref = yref[sample:] 

#plot results
time_ax = [i for i in range(steps*sample)]

# ^for plotting u vals
# plt_us = []
# for i in range(steps):
#     for j in range(sample):
#         plt_us.append(us[i])
# plt.plot(time_ax, plt_us)
# plt.show()

#? For firing rates plotting
# plt.plot(time_ax, zs, plot_zDs)
# #plt.ylim(0, 3)
# plt.show()

#~ For plotting LFPs
plt.plot(time_ax, ys, plot_yDs)
title = "Sample: {sample}".format(sample=sample)
plt.title(title)
plt.show()



#! ignore below for now
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

