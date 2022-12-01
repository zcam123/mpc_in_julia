using JuMP
using Plots
using LinearAlgebra
using Gurobi

#defining all parameters for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
        9.83e-4 1 -4.09e-8 -8.32e-5;
        4.83e-7 9.83e-4 1 -5.34e-4;
        1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'

C = [-.0096 .0135 .005 -.0095]

y2z(y) = exp(61.4*y - 5.468)
z2y(z) = (log(z) + 5.468)/61.4

t_step = 0.001 #1 milisecond
pred = 25 #prediction horizon
ctr = 5 #control horizon
sample = 250 #number of steps between sampling points for control

zD = 0.2
yD = z2y(zD)
uD = inv(C*inv(I - A)*B) * yD
#xD = inv(I - A) * B * uD

Q = C'*C
R = 1e-5*I



neuron = Model(Gurobi.Optimizer)
set_silent(neuron)

#Define state variables
@variables(neuron, begin
    x[i=1:4, t=1:(pred*sample)], (start = 0)
    0 ≤ u[1:1, 1:(ctr+1)], (start = 0)
    xD[i=1:4, t=1:(pred*sample)], (start = 0)
end)

@expressions(
    neuron,
    begin
        y, C*x
        x_error[t=1:(pred*sample)], x[:, t] - xD[:, t]
        x_cost[t=1:(pred*sample)], x_error[t]'*Q*x_error[t]
        sampled_x_cost[t=1:pred], x_cost[t*sample]
        u_cost[t=1:ctr+1], u[t]'*R*u[t]
    end
)

#fix first sample steps
@constraint(neuron, x[:, 2:(sample)] .== A*x[:, 1:sample-1] + B*u[:, 1])

#fix each sample period
for i in 1:(ctr-1)
    @constraint(neuron, x[:, (sample*i+1):(sample*i+sample)] .== A*x[:, (sample*i):(sample*i+sample-1)] + B*u[:, (i+1)])
end

#fix rest of inputs 
@constraint(neuron, x[:, (ctr*sample+1):(pred*sample)] .== A*x[:, (ctr*sample):(pred*sample-1)] + B*u[:, (ctr+1)])

#add in our variable reference
#structured to take a list of firing rate values, convert them to x vectors, then pad with zeros if necessary
firing_rate_ref = [0.2 for _ in 1:5000] #trial firing rate reference


function get_states(firing_rates, A, B)
    desired_states = [[] for _ in 1:(length(firing_rates))];
    for i in 1:(length(firing_rates))
        yD = z2y(firing_rates[i])
        uD = inv(C*inv(I - A)*B) * yD
        xD = inv(I - A) * B * uD
        push!(desired_states[i], xD)
    end
    return desired_states
end

desired_states = get_states(firing_rate_ref, A, B)
#print(desired_states)
#print(length(desired_states))######
for i in 1:(length(desired_states))
       fix.(xD[:, i], desired_states[i][1]; force=true)
end
for i in (length(desired_states)+1):(pred*sample)
       fix.(xD[:, i], [0;0;0;0]; force=true)
end

print(sampled_x_cost)
# x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))

function sim(steps, x0; u_clamp=nothing, sample=250)
    zs = zeros(0)
    us = zeros(0)
    x_current = x0
    for t in 1:steps
        fix.(x[:, 1], x_current; force=true)
        if u_clamp != nothing
            fix.(u[:, 1], u_clamp; force=true)
        end
        optimize!(neuron)
        append!(zs , y2z.(value(y[1])))
        append!(us, value(u[1, 1]))
        x_current = value.(x[:, 2])
        #now effectively apply optimal first u for 250 more steps
        const_u = value.(u[1]);
        for i in 1:(sample-1)
            x_current = A*x_current + B*const_u
            append!(zs, y2z.((C*x_current)[1]))
        end
    end
    #println(solution_summary(neuron))
    return zs, us
end

@time begin
    zs, us = sim(22, zeros(4), u_clamp=nothing, sample=sample);
end

print(zs[16*sample])
pz = plot(zs, label="z", color="red")
pu = plot(us, label="u")
plot(pz, pu, layout=[1,1])