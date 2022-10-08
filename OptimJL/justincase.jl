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
    return exp(61.4*y[1] - 5.468);
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
    for i in 1:4:(prediction_horizon*4 - 3)
        curr_x = [optimization_states[i], optimization_states[i+1], optimization_states[i+2], optimization_states[i+3]]
        z = y_to_z(C*curr_x);
        append!(Z, z);
    end
 
    #get errors by calling errors function and giving it the results of the reference function
    z_current = Z[1];
    
    reference_traj = reference(z_current, prediction_horizon, weight=0.5, set_point=0.2)
    errs = errors(Z, error_weights, reference_traj);

    #get control penalty via u_penalty function
    U = zeros(0);
    for i in (prediction_horizon*4 + 1):length(optimization_states)
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

#make vectors for use in constraint function - optimization_states includes all x's followed by inputs
#setting all values to zero as this is a convenient initial guess (for the solver to use) that satisfies the constraints
optimization_states = [0 for _ in 1:(prediction_horizon*4 + control_horizon)] 
lngth = length(optimization_states)

#function to get state values from t to t + prediction_horizon - 1 and then apply x = Ax + Bu to them
function X_behind(optimization_states)
    X_behind = [[] for _ in 1:(prediction_horizon-1)]
    for i in 1:(prediction_horizon-1)     
        states = zeros(0);
        for j in 4*(i-1)+1:4*(i-1)+4
            append!(states, optimization_states[j])
        end                                 
        push!(X_behind[i], A*states + B .* optimization_states[prediction_horizon*4 + i])
    end
    return X_behind
end


#function to get state values from t+1 to t + prediction_horizon so we can compare them with the output of the above function
function X_atm(optimization_states)
    X_atm = [[] for _ in 2:(prediction_horizon)]
    for i in 2:(prediction_horizon)     
        states = zeros(0);
        #gather all parts of this state vector
        for j in 4*(i-1)+1:4*(i-1)+4
            append!(states, optimization_states[j])
        end                                 
        push!(X_atm[i-1], states)
    end
    return X_atm
end


#now try to use with optimization package
#simpler version of J to use with specific values already passed
f(optimization_states, p) = p[1]-p[1] + J(C, y_to_z, reference, errors, u_penalty, control_horizon, prediction_horizon, optimization_states, u_current, error_weights, du_weights);
p = [0, 0]

# #constraint - takes above functions and takes their difference which must be zero
# cons(optimization_states) = X_atm(optimization_states) - X_behind(optimization_states)
# opt_fun = OptimizationFunction(f, cons=cons)

#define lcons to match vector difference returned by constraint
lcons = [Vector{Float64}[] for _ in 1:(prediction_horizon-1)] #specifying Float type for matching with state differences later
for i in 1:(prediction_horizon-1)     
    push!(lcons[i], [0, 0, 0, 0])
end
ucons = lcons

#make lower and upper bounds that correspond with the optimization_states vector
lb = [-1000 for _ in 1:(prediction_horizon*4)] 
for i in 1:control_horizon
    push!(lb, 0)
end
ub = [1000 for _ in 1:(prediction_horizon*4)] 
for i in 1:control_horizon
    push!(ub, 70)
end

#for use in constraint later
function make_match(mismatch, prediction_horizon)
    match = [[] for _ in 1:(prediction_horizon-1)] #because Xatm and Xbehind have prediction_horizon-1 elements
    push!(match[1], mismatch)
    if prediction_horizon-1 == 1
        return match
    else
        for i in 2:(prediction_horizon-1)
            push!(match[i], [0, 0, 0, 0])
        end
        return match
    end
end

#make_match([1, 2, 3, 4], prediction_horizon)

#initial guess for optimizer
initial_state = optimization_states

#Now to run a control loop
steps = 20;

function control_loop(steps, Zs, initial_state, f, p, lcons, ucons, lb, ub, prediction_horizon, control_horizon) #should add other funcs that appear here as args
    start_state = initial_state
    for _ in 1:steps
        #same optimization set up as before but with modified constraint
        fixed_start = start_state[1:4]  #used to force start to stay the same
        cons(optimization_states) = X_atm(optimization_states) - X_behind(optimization_states) + (make_match(optimization_states[1:4], prediction_horizon) - make_match(fixed_start, prediction_horizon))
        opt_fun = OptimizationFunction(f, cons=cons)
        prob = OptimizationProblem(opt_fun, start_state, p, lb=lb, ub=ub, lcons=lcons, ucons=ucons)
        solution = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited());
        #update initial state with result of optimization step vector
        #start_state = solution.u

        #next need to add something that will make new start_state shifted forward one step
        #just take the above minus the first step then add another x based on last u - finally, append a u=0 as a guess 
        #this will effectively amount to the first input being accepted and the second element returned by optimizer becoming our current state
        access = solution.u
        start_state = zeros(0);
        for i in 5:8
            append!(start_state, access[i])
        end
        #get rest of states
        for i in 9:(prediction_horizon*4)
            append!(start_state, access[i])
        end
        #get "new" last state by transforming last state by "new" last u which we make zero for convenience 
        new_last = A*access[(4*prediction_horizon - 3):(4*prediction_horizon)]
        append!(start_state, new_last)
        #finally, add inputs, but not the first one (hence +2 below) which was used (hence replacing x1 w x2) and also must add new last one equal to 0 (given that above has no Bu term)
        for i in (prediction_horizon*4 + 2):length(access)
            append!(start_state, access[i])
        end
        append!(start_state, 0)
        #before moving to next step and solving again store results for plotting
        x_add = start_state[1:4];
        append!(Zs, y_to_z(C*x_add))
    end
    return Zs
end

Zs = zeros(0)
Zs = control_loop(steps, Zs, initial_state, f, p, lcons, ucons, lb, ub, prediction_horizon, control_horizon)

time = [i for i in 1:steps]

goal = [0.2 for i in 1:steps]

plot1 = plot(time, Zs);
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
