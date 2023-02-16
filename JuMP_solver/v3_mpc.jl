using JuMP
using Plots
using LinearAlgebra

#defining all parameters for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
        9.83e-4 1 -4.09e-8 -8.32e-5;
        4.83e-7 9.83e-4 1 -5.34e-4;
        1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'

B1 = B

# fake opsin
Binh = -B .* (1 .+ randn(4, 1)./5)
B2 = [B Binh]

C = [-.0096 .0135 .005 -.0095]

y2z(y) = exp(61.4*y - 5.468)
z2y(z) = (log(z) + 5.468)/61.4

t_step = 0.001 #1 milisecond
pred = 25 #prediction horizon
ctr = 5 #control horizon
sample = 200 #number of steps between sampling points for control

zD = 0.2
yD = z2y(zD)
uD = inv(C*inv(I - A)*B) * yD
#xD = inv(I - A) * B * uD
Q = C'*C
R = 1e-5*I
T = 13000
Tpred = pred*sample

# add in our variable reference
#structured to take a list of firing rate values, convert them to x vectors, then pad with zeros if necessary
##zref = [i < 3000 ? 0.1 : 0.15 for i in 1:T] #trial firing rate reference
# trial firing rate reference
zref = .1 .+ .08*sin.(range(start=0, stop=2*pi, length=Int(1.5*Tpred)));

#using OSQP
using Gurobi

function mpc(steps, x0, zref; nu=1, u_clamp=nothing, sample=250)
    if nu == 1
        B = B1
    elseif nu == 2
        B = B2
    end

    Tpred = pred*sample
    Tall = steps*sample

    if length(zref) < Tall + Tpred
        zrefpad = cat(zref, fill(zref[end], Tall + Tpred - length(zref)), dims=1)
    end

    #neuron = Model(OSQP.Optimizer)
    neuron = Model(Gurobi.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr+1)] .<= 40
        yD[i=1:1, t=1:Tpred]
        # xD[i=1:4, t=1:Tpred], (start = 0)
    end)

    @expressions(
        neuron,
        begin
            y, C*x
            # x_error[t=1:Tpred], x[:, t] - xD[:, t]
            # x_cost[t=1:Tpred], x_error[t]'*Q*x_error[t]
            y_error[t=1:Tpred], y[:, t] - yD[:, t]
            y_cost[t=1:Tpred], y_error'[t]*y_error[t]
            # sampled_x_cost[t=1:pred], x_cost[t*sample]
            sampled_y_cost[t=1:pred], y_cost[t*sample]
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
    @constraint(neuron, x[:, (ctr*sample+1):(Tpred)] .== A*x[:, (ctr*sample):(Tpred-1)] + B*u[:, (ctr+1)])
    yDall = z2y.(zrefpad)
    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    # J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    J = @objective(neuron, Min, sum(sampled_y_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    # if nu == 2 
    #     B = [B -B]
    # end

    zs = zeros(1, steps*sample)
    us = zeros(nu, steps*sample)
    x_current = x0
    tfine = 1

    for t in 1:steps
        fix.(x[:, 1], x_current; force=true)
        if u_clamp != nothing
            fix.(u[:, 1], u_clamp; force=true)
        end

        #now need to update the reference by "shifting it" one sample size forward and padding with end value
        fix.(yD[:], yDall[tfine:tfine+Tpred-1], force=true)
        optimize!(neuron)
        zs[1, tfine] = y2z.(value(y[1]))
        us[:, tfine] = value.(u[:, 1])
        # append!(zs , y2z.(value(y[1])))
        # append!(us, value(u[:, 1]))
        x_current = value.(x[:, 2])
        # println(x_current)
        # now effectively apply optimal first u for 250 more steps
        const_u = value.(u[:, 1]);
        tfine += 1

        for i in 1:(sample-1)
            x_current = A*x_current + B*const_u
            # append!(zs, y2z.((C*x_current)[1]))
            zs[1, tfine] = y2z.((C*x_current)[1])
            us[:, tfine] = const_u
            tfine += 1
        end
    end
    #println(solution_summary(neuron))
    return zs, us
end

steps = T ÷ sample
mpc2res = mpc(steps, zeros(4), zref, nu=2, u_clamp=nothing, sample=sample);
zs, us = mpc2res

uu1 = us[1,:]
uu2 = us[2,:]
plot([i for i in 1:601], uu2[2000:2600])

# zeross = [0 for _ in length(uu1)]
# print("\n", uu1 >= zeross, "   ", uu2 >= zeross)
u1_problem = false
u2_problem = false
for u in uu1
    if u < 0
        global u1_problem = true
    end
end
for u in uu2
    if u < 0
        global u2_problem = true
    end
end
print("\n", u1_problem, "  ", u2_problem)

function plotctrl(zs, us; title=nothing, plotargs...)
    nu = size(us, 1)
    last = zref[end][1]
    if (length(zref)) < T
        for i in (length(zref)):(T - 1)
            append!(zref, last)
        end
    end
    # print(length(zref), "\n", length(us), "\n", length(zs))
    # print("zref: $(length(zref))")
    time = (1:T) ./ 1000
    zs_plot = plot(time, [zref zs'], label=["r" "z"],
        color=["green" "black"], lw=2)
    plot!(legend=false)
    if title != nothing
        plot!(title=title)
    end
    # if nu == 1
    #     ucolor = :lightskyblue
    # elseif nu == 2
    #     ucolor = [:lightskyblue :orangered]
    # end
    # ucolor = "lightskyblue" if nu == 1 else [:lightskyblue :redorange] end
    u_plot = plot(time, us[1, :], color="#72b5f2", lw=2, xlabel="time (s)", legend=false, label="u_{exc}")
    if nu == 2
        plot!(time, us[2, :], color=:orangered, lw=2, label="u_{inh}")
    end
    return plot(zs_plot, u_plot; layout=(2, 1), link=:x, plotargs...)
end

mpc2plot = plotctrl(mpc2res...; title="2-input MPC")