using LinearAlgebra
using Plots
using Optimization
using OptimizationBBO

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

#creates referene trajectory of desired values for firing rate. Weight argument controls how smooth of a process this should be 
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
function J(C, y_to_z, reference, errors, u_penalty, control_horizon, prediction_horizon, optimization_states, u_current, error_weights, du_weights)
    #calculate firing rates based on given set of future states
    Z = zeros(0);
    for i in 1:(length(optimization_states) - control_horizon)
        z = y_to_z(C*optimization_states[i][1]);
        append!(Z, z);
    end
 
    #get errors by calling errors function and giving it the results of the reference function
    z_current = Z[1];
    
    reference_traj = reference(z_current, prediction_horizon, weight=0.5, set_point=0.2)
    errs = errors(Z, error_weights, reference_traj);

    #get control penalty via u_penalty function
    U = zeros(0);
    for i in (prediction_horizon+1):length(optimization_states)
        u = optimization_states[i];
        append!(U, u);
    end
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

#Here are some trial values
control_horizon = 2;
prediction_horizon = 3;
error_weights = [1 for i in 1:prediction_horizon];
du_weights = [0 for i in 1:control_horizon];
u_current = 0;
x_current = [-1.548; 2.18; .806; -1.53];

#make vectors for use in constraint function
optimization_states = [[] for _ in 1:prediction_horizon+control_horizon] 
#initial guess will be states of all zeros and u's all equal to zero
for i in 1:prediction_horizon
    push!(optimization_states[i], [0; 0; 0; 0])
end
for i in (prediction_horizon+1):(prediction_horizon+control_horizon)
    push!(optimization_states[i], 0)
end

#function to get state values from t to t + prediction_horizon - 1 and then apply x = Ax + Bu to them
function X_behind(optimization_states)
    X_behind = [[] for _ in 1:(prediction_horizon-1)]
    for i in 1:(prediction_horizon-1)     #combine state with corresponding input u
        push!(X_behind[i], A*optimization_states[i][1] + B .* optimization_states[i+prediction_horizon])
    end
    return X_behind
end

#function to get state values from t + 1 to prediction_horizon to be compared with above values of the states one time
#step behind but transformed to the current time's value
function X_atm(optimization_states)
    X_atm = [[] for _ in 2:(prediction_horizon)]
    for i in 2:(prediction_horizon)     
        push!(X_atm[i-1], optimization_states[i][1])
    end
    return X_atm
end


#now try to use with optimization package
#simpler version of J to use with specific values already passed
f(optimization_states) = J(C, y_to_z, reference, errors, u_penalty, control_horizon, prediction_horizon, optimization_states, u_current, error_weights, du_weights);

#constraint - takes above functions and takes difference which must be zero
cons(optimization_states) = X_atm(optimization_states) - X_behind(optimization_states)

optimization_function = OptimizationFunction(f, cons=cons)

#define lcons to match vector difference returned by constraint
lcons = [Vector{Float64}[] for _ in 1:(prediction_horizon-1)] #specifying Float type for matching with state differences later
for i in 1:(prediction_horizon-1)     
    push!(lcons[i], [0; 0; 0; 0])
end

ucons = lcons
initial_state = optimization_states

lb = [[] for _ in 1:prediction_horizon+control_horizon] 
for i in 1:prediction_horizon
    push!(lb[i], [-100; -100; -100; -100])
end
for i in (prediction_horizon+1):(prediction_horizon+control_horizon)
    push!(lb[i], 0)
end

ub = [[] for _ in 1:prediction_horizon+control_horizon] 
for i in 1:prediction_horizon
    push!(ub[i], [100; 100; 100; 100])
end
for i in (prediction_horizon+1):(prediction_horizon+control_horizon)
    push!(ub[i], 70)
end

prob = OptimizationProblem(optimization_function, initial_state, lb=lb, ub=ub, lcons=lcons, ucons=ucons)

#isssue seems to be lack of lower bounds and upper bounds
sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited());




#ignore below for now

#function to run the control loop
#optimizes J(x) at each iteration then impliments one step of optimal control policy
#before recomputing at next step
# function control_loop(A, B, C, y_to_z, sim, reference, errors, u_penalty, control_horizon, prediction_horizon, x_current, u_current, error_weights, du_weights, states; steps=10)
#     for i in 1:steps
#         #call J and have it minimized - u is a list of x vectors followed by a list of control values - "optimization states" from above definition
#         rosenbrock(u) =  J(C, y_to_z, reference, errors, u_penalty, control_horizon, prediction_horizon, u, u_current, error_weights, du_weights)

#         u0 = [u_current];
#         append!(u0, zeros(control_horizon-1));

#         prob = OptimizationProblem(rosenbrock, u0, p, );
#         U = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited());
        
#         uAccess = zeros(0);
#         for elem in U.u
#             append!(uAccess, elem)
#         end
#     end
#     return uAccess
# end


# #now we can try to run a control loop 
# #Here are some trial values
# ctr_hor = 2;
# pred = 3;
# error_weights = [1 for i in 1:pred];
# du_weights = [0 for i in 1:ctr_hor];
# states = zeros(0);
# u_current = 0;
# x_current = [-1.548; 2.18; .806; -1.53];

# steps = 600;

# xs = control_loop(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, ctr_hor, pred, x_current, u_current, error_weights, du_weights, states; steps=steps);

# #loop over results in order to get them in plotable form 
# z_list = zeros(0);
# upper_index = steps*4 - 3;
# for i in 1:4:upper_index
#     xadd = [xs[i]; xs[i+1]; xs[i+2]; xs[i+3]];
#     z_add = y_to_z(C*xadd);
#     append!(z_list, z_add);
# end


# time = [i for i in 1:steps]

# goal = [0.2 for i in 1:steps]

# plot1 = plot(time, z_list);
# plot!(time, goal)



# #test to see if state evolves correctly
# x = x_curr;
# Z = zeros(0);
# function test(Z, x, steps, u, A, B, C, y_to_z)
#     for i in 1:steps
#         x = A*x + B .* u;
#         y = C*x;
#         z1 = y_to_z(y);
#         append!(Z, z1);
#     end
# end

# test(Z, x, 50000, 0, A, B, C, y_to_z);

# time = [i for i in 1:50000]
# plot(time, Z)
