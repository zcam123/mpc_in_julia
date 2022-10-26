using LinearAlgebra
using Plots
using Optimization
using OptimizationBBO
using Symbolics

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
#to calculate our expression to be minimized as a function of a control sequence U                    ***
function J(U, A, B, C, D, y_to_z, sim, reference, errors, u_penalty; control_horizon, prediction_horizon, x_current, u_current, error_weights, du_weights)
    #get future firing rates by calling simulation function
    Z = sim(A, B, C, D, x_current, U, prediction_horizon, control_horizon, y_to_z);
    #get errors by calling errors function and giving it the results of the reference function
    z_current = y_to_z(C*(x_current));
    reference_traj = reference(z_current, prediction_horizon, weight=0.5, set_point=0.2)
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

#try using symbolics.jl to get explicit formula
#give it all hyperparameters to make function to be optimized before running control loop
# @variables u1, u2;
# U = [u1, u2];          #actually we'll try to register
#Here are some trial values
ctr_hor = 2;
pred = 3;
error_weights = [1 for i in 1:pred];
du_weights = [0 for i in 1:ctr_hor];
#states = zeros(0)
u_current = 0;
x_current = [-1.548; 2.18; .806; -1.53];

Jay(U) = J(U, A, B, C, D, y_to_z, sim, reference, errors, u_penalty, prediction_horizon=pred, control_horizon=ctr_hor, x_current=x_current, u_current=u_current, error_weights=error_weights, du_weights=du_weights)

@register_symbolic Jay(U)

Jay([u1, u2])

#=

#function to run the control loop
#optimizes J(x) at each iteration then impliments one step of optimal control policy
#before recomputing at next step
function control_loop(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon, prediction_horizon, x_current, u_current, error_weights, du_weights, states; steps=10)
    for i in 1:steps
        #call J and have it minimized
        rosenbrock(u,p) =  p[1] - p[1] + J(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, control_horizon=control_horizon, prediction_horizon=prediction_horizon, U=u, x_current=x_current, u_current=u_current, error_weights=error_weights, du_weights=du_weights)
        u0 = [u_current];
        append!(u0, zeros(control_horizon-1));
        p  = [0.0000001, 100];

        ub = [50 for i in 1:control_horizon]
        prob = OptimizationProblem(rosenbrock, u0, p, lb = zeros(control_horizon), ub = ub);
        U = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited());
        
        uAccess = zeros(0);
        for elem in U.u
            append!(uAccess, elem)
        end
        
        #print values to get an idea of relative magnitudes 
        if i == 1
            print(uAccess, "\n", A*x_current, "\n", (B .* uAccess[1]), "\n")
        end

        #now that we have U impliment first step
        x_current = A*x_current + B .* (uAccess[1])*10^13; # to see if ctr is capable of doing anything 
        u_current = uAccess[1];
        append!(states, x_current);
    end
    return states
end


#now we can try to run a control loop 
#Here are some trial values
ctr_hor = 2;
pred = 3;
error_weights = [1 for i in 1:pred];
du_weights = [0 for i in 1:ctr_hor];
states = zeros(0);
u_current = 0;
x_current = [-1.548; 2.18; .806; -1.53];

steps = 10;

xs = control_loop(A, B, C, D, y_to_z, sim, reference, errors, u_penalty, ctr_hor, pred, x_current, u_current, error_weights, du_weights, states; steps=steps);

#loop over results in order to get them in plotable form 
z_list = zeros(0);
upper_index = steps*4 - 3;
for i in 1:4:upper_index
    xadd = [xs[i]; xs[i+1]; xs[i+2]; xs[i+3]];
    z_add = y_to_z(C*xadd);
    append!(z_list, z_add);
end


time = [i for i in 1:steps]

goal = [0.2 for i in 1:steps]

plot1 = plot(time, z_list);
plot!(time, goal)



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

=#