from filterpy.kalman import KalmanFilter
import numpy as np

f = KalmanFilter(dim_x=4, dim_z=1)

f.x = np.array([0., 0., 0., 0.])

#Package calls A as F to be consistent with standard KF notation
A = np.array([
    [1, -6.66e-13, -2.03e-9, -4.14e-6],
    [9.83e-4, 1, -4.09e-8, -8.32e-5],
    [4.83e-7, 9.83e-4, 1, -5.34e-4],
    [1.58e-10, 4.83e-7, 9.83e-4, .9994]
])
f.F = A

#They call input matrix B as opposed to G
B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T
f.B = B

#observability matrix (which is C in our model's notation)
C = np.array([[-.0096, .0135, .005, -.0095]])
f.H = C

#define uncertainty matrix using their prebuilt diagonal matrix however this assumes 
#that the uncertainty in each x are independent of each other
initial_p = 10^-5
f.P *= initial_p

#since measurement is one dimensional can use a scalar
meas_noise = 0.
f.R = meas_noise

#define process noise matrix - will need to tweak variance based on white noise of our model??
from filterpy.common import Q_discrete_white_noise
f.Q = Q_discrete_white_noise(dim=4, dt=0.001, var=0.)

#trial loop
iters = 500
xs = []
x_ = [0., 0., 0., 0.]
# xs.append(x_)
u_const = [0.5]
for i in range(iters):
    x_ = A@x_ + B@u_const
    xs.append(x_)

zs = [C@x for x in xs]

#z = zs[0]
# #update Kalman filter with initial guess and predict next state
# f.update(z)
# f.predict(u = u_const)
# print(f.x)


estimates = []
#filter loop
for i in range(iters):
    z = zs[i] #lfp measurement
    f.update(z)
    curr_est = f.x
    estimates.append(curr_est)
    f.predict(u = u_const)

#print(xs, "\n \n", estimates)

errors = [xs[i] - estimates[i] for i in range(iters)]
for i in range(iters):
    for j in range(4):
        errors[i][j] *= 1/(xs[i][j])
#now we have percent error
#print(errors)


import matplotlib.pyplot as plt
x_ax = [i for i in range(iters)]

# est_zs = [C@estimates[i] for i in range(iters)]
# plt.plot(x_ax, zs)
# plt.plot(x_ax, est_zs)


est1 = [errors[i][0] for i in range(iters)]
actual1 = [xs[i][0] for i in range(iters)]

est2 = [errors[i][1] for i in range(iters)]
actual2 = [xs[i][1] for i in range(iters)]

est3 = [errors[i][2] for i in range(iters)]
actual3 = [xs[i][2] for i in range(iters)]

est4 = [errors[i][3] for i in range(iters)]
actual4 = [xs[i][3] for i in range(iters)]

fig, axs = plt.subplots(4, 1)
axs[0].plot(x_ax[5:], est1[5:])
axs[0].plot(x_ax[5:], actual1[5:])
axs[1].plot(x_ax[50:], est2[50:])
axs[1].plot(x_ax[50:], actual2[50:])
axs[2].plot(x_ax[50:], est3[50:])
axs[2].plot(x_ax[50:], actual3[50:])
axs[3].plot(x_ax[50:], est4[50:])
axs[3].plot(x_ax[50:], actual4[50:])

plt.show()


