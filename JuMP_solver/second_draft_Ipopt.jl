using JuMP
import Ipopt
using Plots

#initialize the model
neuron = Model(Ipopt.Optimizer)
set_silent(neuron)

#defining all parameters for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
     9.83e-4 1 -4.09e-8 -8.32e-5;
     4.83e-7 9.83e-4 1 -5.34e-4;
     1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]

C = [-.0096 .0135 .005 -.0095]

#functions based on Ax + Bu to give each part of the x vector - used as storing components of x vector seperately is more convenient with their setup
# x1next(x1, x2, x3, x4, u) = x1 - 6.66*10^(-13)*x2 - 2.03*10^(-9)*x3 - 4.14*10^(-6)*x4 + 94.3*u
# x2next(x1, x2, x3, x4, u) = 0.000983*x1 + x2 - 4.09*10^(-8)*x3 - 0.0000832*x4 + 41.3*u
# x3next(x1, x2, x3, x4, u) = 4.83*10^(-7)*x1 + 0.000983*x2 + x3 - 0.000534*x4 + u*1.58*10^-10
# x4next(x1, x2, x3, x4, u) = 1.58*10^(-10)*x1 + 4.83*10^(-7)*x2 + 0.000983*x3 + 0.9994*x4 + u*3.89*10^-14

#function based on C*x to get y  
x_to_y(x) = C*x

t_step = 0.001 #1 milisecond
pred = 5 #prediction horizon - number of time steps that the optimizer will go through in determining the value of J and minimizing it
ctr = 3 #control horizon

#Define state variables
@variables(neuron, begin
    t_step ≥ 0, (start = t_step) # Time step
    # State vector
    x[1:4, 1:pred]
    #control value u         
    0 ≤ u[1:pred] ≤ 70     
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
    end
);

#expression for sum of squares of error 
objective_expression = transpose(error)*error

@NLobjective(neuron, Min, objective_expression)

for j in 2:pred
    #x vector dynamics - from Ax + Bu
    @constraint(neuron, x[1:4, j] .== A*x[1:4, j-1] + B*u[j-1])
end

# optimize!(neuron)
# solution_summary(neuron)

# y_val[4][1]

# stats = value.(x)
# stats[3, 2]

steps = 50000 
ys = zeros(0) #to store results during control loop


for i in 1:steps
    #solve the optimization problem
    optimize!(neuron)
    solution_summary(neuron)
    #store the new y_value for plotting
    append!(ys, value.(y_val[2, 1][1])) #this gets us the second value from the optimized states returned by the solver which will become the new first y in the next loop
        #replace append with replace() usage later
    
    #get state after application of first input to be used as new first state
    new_initial = [value.(x[i, 2]) for i in 1:4] #need to store these results from the solve before fix statements reset the model
    
    #set new initial conditions before next loop based on second state of this run - after first input from solution has been applied
    fix(x[1, 1], new_initial[1]; force = true)
    fix(x[2, 1], new_initial[2]; force = true)
    fix(x[3, 1], new_initial[3]; force = true)
    fix(x[4, 1], new_initial[4]; force = true)

end

time = [i for i in 1:steps]

function y_to_z(y)
    return exp(61.4*y[1] - 5.468);
end
zs = [y_to_z(y) for y in ys]

plot1 = plot(time, zs)

zDs = [0.2 for _ in 1:steps]

plot!(time, zDs)
