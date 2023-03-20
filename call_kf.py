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

#sample control loop
sample = 3
sim_length = 1000
steps = int(sim_length/sample) #convert to integer or else python complains

#!cc3 reference
#yref = z_to_y(zref)
from_file = np.load("tklfp.npy")
yref = [elem for elem in from_file]
for _ in range(steps*sample - len(yref)):
    yref.append(yref[-1])
plot_yDs = yref #store initial reference now before it gets initial values chopped off during below loop

#! simple yref for testing
# yref = (-1)*np.exp((-1/2000)*(np.arange(-600, 600))**2) - 0.1
# plot_yDs = yref

#!constant yref
# yref = np.zeros(1000)
# plot_yDs = yref

x = np.zeros(4)
x_est = np.zeros(4)
zs = []
us = []
P = np.eye(4)*15
R = 10**(-7)
ys = []
# *Here is a loop for firing rate
for i in range(steps):
    #kf part
    z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise added to measurement of lfp
    x_est, P = jl.only_update_KF(jl.Array(z), jl.Array(P), R, jl.Array(x_est), jl.Array(C))

    #mpc part - now using KF estimate's x instead of the actual state
    optimal_u = jl.flex_mpc(jl.Array(x_est), jl.Array( np.array(yref) ), nu=1, sample=sample, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C), ref_type=2)
    us.append(optimal_u)
    for i in range(sample):
        x_est, P = jl.KF_est(jl.Array(z), jl.Array(P), R, jl.Array(x_est), optimal_u, A=jl.Array(A), B=jl.Array(B), C=jl.Array(C)) #kf
        x = A@x + B@optimal_u
        zs.append(y_to_z(C@x))
        ys.append(C@x)
        #KF
        z = np.array(rand.normal(C@x, np.sqrt(R))) #Gaussian noise
        
    #shift reference
    yref = yref[sample:]  

#plot results
time_ax = [i for i in range(steps*sample)]

# ^for plotting u vals
plt_us = []
for i in range(steps):
    for j in range(sample):
        plt_us.append(us[i])

#~ For plotting inputs with LFPs
title = "Sample: {sample}".format(sample=sample)
figure, axis = plt.subplots(1, 2)
axis[0].plot(time_ax, ys, plot_yDs)
axis[0].set_title(title)
axis[1].plot(time_ax, plt_us)
axis[1].set_title("input")
plt.show()
