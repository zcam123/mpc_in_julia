using JuMP
import Ipopt
using Plots

#initialize the model
neuron = Model(Ipopt.Optimizer)
set_silent(neuron)

#defining all parameters of for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
     9.83e-4 1 -4.09e-8 -8.32e-5;
     4.83e-7 9.83e-4 1 -5.34e-4;
     1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]

C = [-.0096 .0135 .005 -.0095]

#functions based on Ax + Bu to give each part of the x vector
x1next(x1, x2, x3, x4, u) = x1 - 6.66*10^(-13)*x2 - 2.03*10^(-9)*x3 - 4.14*10^(-6)*x4 + 94.3*u
x2next(x1, x2, x3, x4, u) = 0.000983*x1 + x2 - 4.09*10^(-8)*x3 - 0.0000832*x4 + 41.3*u
x3next(x1, x2, x3, x4, u) = 4.83*10^(-7)*x1 + 0.000983*x2 + x3 - 0.000534*x4 + u*1.58*10^-10
x4next(x1, x2, x3, x4, u) = 1.58*10^(-10)*x1 + 4.83*10^(-7)*x2 + 0.000983*x3 + 0.9994*x4 + u*3.89*10^-14

#function based on C*x to get y  
x_to_y(x1, x2, x3, x4) = -0.0096*x1 + 0.0135*x2 + 0.005*x3 -0.0095*x4

t_step = 0.001 #1 milisecond
pred = 4 #prediction horizon - number of time steps that the optimizer will go through in determining the value of J and minimizing it
ctr = 3 #control horizon

#for later use in error
yD=0.06284303
y0=x_to_y(-1,1,-0.5,0.5)
J0=(yD-y0)^2

#Define state variables
@variables(neuron, begin
    t_step ≥ 0, (start = t_step) # Time step
    # State variables - state split into four parts
    x1[1:pred]   
    x2[1:pred]
    x3[1:pred]
    x4[1:pred]
    #control value u         
    u[1:pred] ≥ 0 
    #Cost value to minimize
    J[1:pred]
end);

#set objective for the optimization problem
@objective(neuron, Min, J[pred])

#fix initial conditions as well as u to be constant over 
fix(x1[1], -1; force = true)
fix(x2[1], 1; force = true)
fix(x3[1], -0.5; force = true)
fix(x4[1], 0.5; force = true)
#give starting error value based on first state
J0 = (yD-y0)^2
fix(J[1], J0; force = true)


#Expressions for use in determining state values
@NLexpressions(
    neuron,
    begin
        #expression for y = C*x - y_val used as y already being used by something else
        y_val[j = 1:pred], -0.0096*x1[j] + 0.0135*x2[j] + 0.005*x3[j] -0.0095*x4[j]
    end
);

for j in 2:pred
    #x vector dynamics
    @NLconstraint(neuron, x1[j] == x1[j-1] - 6.66*10^(-13)*x2[j-1] - 2.03*10^(-9)*x3[j-1] - 4.14*10^(-6)*x4[j-1] + 94.3*u[j-1])
    @NLconstraint(neuron, x2[j] == 0.000983*x1[j-1] + x2[j-1] - 4.09*10^(-8)*x3[j-1] - 0.0000832*x4[j-1] + 41.3*u[j-1])
    @NLconstraint(neuron, x3[j] == 4.83*10^(-7)*x1[j-1] + 0.000983*x2[j-1] + x3[j-1] - 0.000534*x4[j-1] + u[j-1]*1.58*10^-10)
    @NLconstraint(neuron, x4[j] == 1.58*10^(-10)*x1[j-1] + 4.83*10^(-7)*x2[j-1] + 0.000983*x3[j-1] + 0.9994*x4[j-1] + u[j-1]*3.89*10^-14)

    #These constraints force J to properly accrue error over the prediction horizon
    @NLconstraint(neuron, 
    J[j] == J[j-1] + (y_val[j]-yD)^2)
end

optimize!(neuron)
solution_summary(neuron)

