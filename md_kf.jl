using LinearAlgebra
using Plots

#parameters of the model
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
        9.83e-4 1 -4.09e-8 -8.32e-5;
        4.83e-7 9.83e-4 1 -5.34e-4;
        1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'

C = [-.0096 .0135 .005 -.0095]

#function to get kalman gain 
function kalman_gain(P, H, R)
    K = P*H'*inv(H*P*H' + R)
    return K
end

#state update equation to account for new measurement z
function state_update(curr_est, H, K, z)
    innovation = K*(z - H*curr_est)
    return curr_est + innovation
end

#state extrapolation function
function extrapolate_state(x_curr, A, B; u=0)
    return A*x_curr + B*u
end

#covariance update equation
function covariance_update(K, H, P, R)
    return (I - K*H)*P*(I - K*H)' + K*R*K'
end

#covariance extrapolation function - F is state update matrix and Q is process noise
function extrapolate_covariance(P, F, Q)
    return F*P*F' + Q
end

#state update matrix
F = A
#def initial estimate uncertainty
#assume that covariance of latent state vars is zero
P = I*0.3 #initial estimate uncertainty depends on confidence in our initial guess of the state x

#observability matrix H (which is C in our model)
H = [-.0096 .0135 .005 -.0095];

#measurement uncertainty R depends on accuracy of our LFP measurements
R = [0.000]

#process noise Q
Q = I*0

# #trial
# x_guess = [0, 0, 0, 0]
# z = [1] #treat z as a 1 by 1 to make operations go smoothly above
# K = kalman_gain(P, H, R)
# new_x = state_update(x_guess, H, K, z)
# new_p = covariance_update(K, H, P, R)
# predicted_x = extrapolate_state(new_x, A, B, u=0)
# predicted_p = extrapolate_covariance(new_p, F, Q)

# print(predicted_x, "\n", predicted_p)

#ok now try to make a full loop

function quick_trial(P, H, F, A, B, C, R, Q)
    x = [0, 0, 0, 0] #initial guess
    u = 0.1 #constant input
    iters = 2000
    lfps = []
    for i in 1:iters
        x = A*x + B*u
        append!(lfps, [C*x]) #appending as a one by one for sake of matrix operations later on
    end

    lfp_est = []
    #kf loop
    x = [0, 0, 0, 0] #one time step behind exact starting value so very good initial guess
    for i in 1:iters
        #get measurement
        z = lfps[i]
        K = kalman_gain(P, H, R) #kalman gain calculation
        x = state_update(x, H, K, z) #update the state
        P = covariance_update(K, H, P, R) #update estimate uncertainty
        #store estimates
        append!(lfp_est, C*x)
        #extrapolate estimate uncertainty and state for next iteration
        x = extrapolate_state(x, A, B, u=u)
        P = extrapolate_covariance(P, F, Q)
    end

    #print("\n", lfps[end][1], "\n", lfp_est[end])
    return lfps, lfp_est, iters
end
     
lfps, lfp_est, iters = quick_trial(P, H, F, A, B, C, R, Q)

plot_lfps = []
for elem in lfps
    append!(plot_lfps, elem[1])
end

time = [i for i in 1:(iters)]
kf_plot = plot(time, [plot_lfps, lfp_est], label=["lfp" "estimate"])

#like above testing function but this one will give us latent states
function second_trial(P, H, F, A, B, C, R, Q)
    x = [0.01, 0.01, 0.01, 0.01] #initial guess
    u = 0.1 #constant input
    iters = 2000
    xs = []
    lfps = []
    #loop to get sample values
    for i in 1:iters
        x = A*x + B*u
        append!(lfps, [C*x]) #appending as a one by one for sake of matrix operations later on
        append!(xs, [x])
    end

    x_est = []
    #kf loop
    x = [0, 0, 0, 0] #initial guess will be almost exactly right, just one time step behind
    for i in 1:iters
        #get measurement
        z = lfps[i]
        K = kalman_gain(P, H, R) #kalman gain calculation
        x = state_update(x, H, K, z) #update the state
        P = covariance_update(K, H, P, R) #update estimate uncertainty
        #store estimates
        append!(x_est, [x])
        #extrapolate estimate uncertainty and state for next iteration
        x = extrapolate_state(x, A, B, u=u)
        P = extrapolate_covariance(P, F, Q)
    end

    #print("\n", lfps[end][1], "\n", lfp_est[end])
    return xs, x_est, iters
end

xs, x_est, iters = second_trial(P, H, F, A, B, C, R, Q)

#make a plot for each state then show all at once
x1 = [x[1] for x in xs]
est1 = [est[1] for est in x_est]

time = [i for i in 1:(iters)]
kf_plot1 = plot(time, [x1, est1], label=["x1" "x1_est"])

#second
x2 = [x[2] for x in xs]
est2 = [est[2] for est in x_est]

kf_plot2 = plot(time, [x2, est2], label=["x2" "x2_est"])

#third
x3 = [x[3] for x in xs]
est3 = [est[3] for est in x_est]

kf_plot3 = plot(time, [x3, est3], label=["x3" "x3_est"])
#fourth
x4 = [x[4] for x in xs]
est4 = [est[4] for est in x_est]

kf_plot4 = plot(time, [x4, est4], label=["x4" "x4_est"])

#all together
plot(kf_plot1, kf_plot2, kf_plot3, kf_plot4; layout=(2,2))