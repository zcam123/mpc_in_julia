using LinearAlgebra
using Plots

#define system
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
     9.83e-4 1 -4.09e-8 -8.32e-5;
     4.83e-7 9.83e-4 1 -5.34e-4;
     1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]

C = [-.0096 .0135 .005 -.0095]

D = [0.0]

#for getting firing rate Z - y[1] used as julia gives a vector for y during following processes by default
function y_to_z(y)
    return ℯ^(61.4*y[1] - 5.468);
end

#For generating list of future system values as a function of a given control policy U
function sim(A, B, C, D, x_current, U, prediction_horizon, control_horizon, y_to_z)
    x = x_current
    Z = zeros(0);
    for u in U
        x = A*x + B .* u;
        y = C*x;
        z1 = y_to_z(y);
        append!(Z, z1);
    end
    #second loop will run from end of control horizon to end of prediction horizon with constant u equal to last value
    if prediction_horizon == control_horizon
        return Z
    else 
        u_const = U[control_horizon];
        for i in 1:(prediction_horizon - control_horizon)
            x = A*x + B .* u_const;
            y = C*x;
            z1 = y_to_z(y);
            append!(Z, z1);
        end
    end
    return Z
end

#creates referene of desired values for firing rate. Weight argument controls how smooth of a process this should be 
function reference(z_current, prediction_horizon; weight, set_point)
    reference_traj = zeros(0);
    append!(reference_traj, z_current)
    for i in 2:prediction_horizon
        next_ref = weight*reference_traj[i-1] + (1-weight)*set_point
        append!(reference_traj, next_ref)
    end
    return reference_traj
end

#function to create the error values in the firing rate against a reference of desired values
function errors(Z, error_weights, reference)
    errors = zeros(0);
    index = 1;
    for z in Z
        append!(errors, error_weights[index] * (reference[index] - z)^2)
        index += 1;
    end
    return errors
end

#function to get control penalty term based on U
function u_penalty(U, u_current, du_weights)
    penalties = zeros(0);
    first_pen = du_weights[1] * (U[1] - u_current)^2;
    append!(penalties, first_pen);
    #now to get remaining penalties 
    u_length = length(U);
    for i in 2:u_length
        next = du_weights[i] * (U[i] - U[i-1])^2;
        append!(penalties, next)
    end
    return penalties
end

#function which will call the sim, errors, and penalties functions 
#to calculate our expression to be minimized as a function of a control sequence U
function J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty; control_horizon, prediction_horizon, U, x_current, u_current, error_weights, du_weights)
    #get future firing rates by calling simulation function
    Z = sim(A, B, C, D, x_current, U, prediction_horizon, control_horizon, y_to_z);
    #get errors by calling errors function and giving it the results of the reference function
    z_current = y_to_z(C*(x_current));
    reference_traj = reference(z_current, prediction_horizon, weight=0.5, set_point=y_to_z(C*x_current))
    errs = errors(Z, error_weights, reference_traj);
    #get control penalty via u_penalty function
    penalties = u_penalty(U, u_current, du_weights);

    #get total value of J(x) by summing errors and control penalties
    total_cost = 0;
    for err in errs
        total_cost += err
    end
    for pen in penalties
        total_cost += pen
    end
    return total_cost
end

#now we can test the J(U) function
ctr_hor = 7;
u_test = [i for i in 1:ctr_hor];
pred = 10;
x_curr = [-1.548; 2.18; .806; -1.53];
err_w = [1 for i in 1:pred];
du_wts = [0 for i in 1:ctr_hor];
# J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon=ctr_hor, prediction_horizon=pred, U=u_test, x_current=x_curr, u_current=0, error_weights=err_w, du_weights=du_wts)

# #Now lets try to make a control loop with placeholder where solver would be
# u_current = 0;
# x = [-1.548; 2.18; .806; -1.53];
# for i in 1:1000
#     #call j and have it minimized
#     J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon=ctr_hor, prediction_horizon=pred, U=u_test, x_current=x_curr, u_current=0, error_weights=err_w, du_weights=du_wts)
#     #now that we have U impliment first step
#     x = A*x + B .* U[1];
#     u_current = U[1]
# end

# Import the package and define the problem to optimize
using Optimization
# rosenbrock(u,p) =  p[1] - p[1] + J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon=ctr_hor, prediction_horizon=pred, U=u, x_current=x_curr, u_current=0, error_weights=err_w, du_weights=du_wts)
# u0 = zeros(7)
# p  = [0.0000001, 100]

# prob = OptimizationProblem(rosenbrock,u0,p)

# # Import a different solver package and solve the optimization problem a different way
# using OptimizationBBO
# prob = OptimizationProblem(rosenbrock, u0, p, lb = zeros(7), ub = [50, 50, 50, 50, 50, 50, 50])
#sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())

#Now lets try to make a control loop with placeholder where solver would be
states = zeros(0);
u_current = 0;
x = [-1.548; 2.18; .806; -1.53];
using OptimizationBBO
function control_loop(x, u_current, states)
    for i in 1:10
        #call j and have it minimized
        rosenbrock(u,p) =  p[1] - p[1] + J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon=ctr_hor, prediction_horizon=pred, U=u, x_current=x, u_current=u_current, error_weights=err_w, du_weights=du_wts)
        u0 = [u_current];
        u0 = append!(u0, zeros(6))
        p  = [0.0000001, 100]

        prob = OptimizationProblem(rosenbrock,u0,p)

        # Import a different solver package and solve the optimization problem a different way
        prob = OptimizationProblem(rosenbrock, u0, p, lb = zeros(7), ub = [50, 50, 50, 50, 50, 50, 50])


        U = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited());
        #now that we have U impliment first step
        x = A*x + B .* U[1];
        #print(x)
        u_current = U[1];
        append!(states, x)
    end
    return states
end

xs = control_loop(x, u_current, states);
z_list = zeros(0);
for i in 1:4:37
    xadd = [xs[i]; xs[i+1]; xs[i+2]; xs[i+3]];
    z_add = y_to_z(C*xadd);
    append!(z_list, z_add);
end

time = [i for i in 1:10]

plot1 = plot(time, z_list)
# #SHouldn't be 40 members in z_list - should be 10
#we're clearly not getting our z's right