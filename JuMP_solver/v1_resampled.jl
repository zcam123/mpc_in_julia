using JuMP
using Gurobi
using Plots

#initialize the model
neuron = Model(Gurobi.Optimizer)
set_silent(neuron)

#defining all parameters for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
    9.83e-4 1 -4.09e-8 -8.32e-5;
    4.83e-7 9.83e-4 1 -5.34e-4;
    1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]

C = [-.0096 .0135 .005 -.0095]

#function based on C*x to get y  
x_to_y(x) = C*x

t_step = 0.001 #1 milisecond
pred = 6250 #prediction horizon - corresponds to 25 sampling intervals (each 250 ms apart)
ctr = 1250 #control horizon 
sample = 250 

input_size = ctr/sample + 1

#Define state variables
@variables(neuron, begin
    t_step ≥ 0, (start = t_step) # Time step
    # State vector
    x[1:4, 1:pred]
    #control value u         
    0 ≤ u[1:6] ≤ 70     
end);  

#fix initial conditions 
x_current = [-1, 1, -0.5, 0.5]

fix(x[1, 1], x_current[1]; force = true)
fix(x[2, 1], x_current[2]; force = true)
fix(x[3, 1], x_current[3]; force = true)
fix(x[4, 1], x_current[4]; force = true)

#Expressions for use in determining error values
@expressions(
    neuron,
    begin
        #expression for y = C*x - y_val used as y already being used by something else
        y_val[j = 1:pred], C*[x[1,j]; x[2,j]; x[3,j]; x[4,j]] 
                            #yD
        error[j = 1:pred], 0.06284303 - y_val[j][1]  #second index is necessary to get expression for some reason

        #only use errors at sampling points?
        sampled_error[j = 1:25], error[j*250] #seems that number must be manually put in for index or error is thrown
    end
);

#expression for sum of squares of error 
objective_expression = sampled_error'*sampled_error

@objective(neuron, Min, objective_expression)

for j in 2:sample
    #x vector dynamics 
    @constraint(neuron, x[1:4, j] .== A*x[1:4, j-1] + B*u[1])
end
for i in 2:convert(Int64, (ctr/sample))
    for j in (1 + sample*(i-1)):sample*i
        #x vector dynamics 
        @constraint(neuron, x[1:4, j] .== A*x[1:4, j-1] + B*u[i])
    end
end
for j in (ctr+1):pred
    #x vector dynamics 
    @constraint(neuron, x[1:4, j] .== A*x[1:4, j-1] + B*u[convert(Int64, ctr/sample + 1)])
end

steps = 10 
ys = zeros(0) #to store results during control loop
us = zeros(0)

@time begin
    for i in 1:steps
        #solve the optimization problem
        optimize!(neuron)
        solution_summary(neuron)
        #store the new y_value for plotting
        append!(ys, value.(y_val[2, 1][1])) #this gets us the second value from the optimized states returned by the solver which will become the new first y in the next loop
            #replace append with replace() usage later
        append!(us, value.(u[1]))
        #get state after application of first input to be used as new first state
        new_initial = [value.(x[i, 2]) for i in 1:4] #need to store these results from the solve before fix statements reset the model
        
        #now take sample-1 more steps with this constant input
        const_u = value.(u[1]);
        for j in 1:(sample-1)
            new_initial = A*new_initial + B*const_u
            append!(ys, C*new_initial)
        end
        #set new initial conditions before next loop based on second state of this run - after first input from solution has been applied
        fix(x[1, 1], new_initial[1]; force = true)
        fix(x[2, 1], new_initial[2]; force = true)
        fix(x[3, 1], new_initial[3]; force = true)
        fix(x[4, 1], new_initial[4]; force = true)

    end
end

time = [i for i in 1:steps*sample]

function y_to_z(y)
    return exp(61.4*y[1] - 5.468);
end
zs = [y_to_z(y) for y in ys]

plot1 = plot(time, zs, label="firing rate");

zDs = [0.2 for _ in 1:steps*sample]

plot3 = plot(time, zDs, label="zD");

#plot2 = plot(time, plotable_us, label="u");

plot(plot1, plot3, layout=[1,1])
