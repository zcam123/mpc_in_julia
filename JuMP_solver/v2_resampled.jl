@time begin
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
    pred = 5 #prediction horizon
    ctr = 3 #control horizon

    zD = 0.2
    yD = z2y(zD)
    uD = inv(C*inv(I - A)*B) * yD
    xD = inv(I - A) * B * uD

    Q = C'*C
    R = 1e-5*I



    neuron = Model(Gurobi.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:pred], (start = 0)
        0 ≤ u[1:1, 1:pred], (start = 0)
    end)

    @expressions(
        neuron,
        begin
            y, C*x
            x_error[t=1:pred], x[:, t] - xD
            x_cost[t=1:pred], x_error[t]'*Q*x_error[t]
            u_cost[t=1:pred], u[t]'*R*u[t]
        end
    )
    
    #fix first sample steps
    @constraint(neuron, dyncon, x[:, 2:pred] .== A*x[:, 1:pred-1] + B*u[:, 1:pred-1])

    # for t in 2:pred
    #     #x vector dynamics - from Ax + Bu
    #     @constraint(neuron, x[:, t] .== A*x[:, t-1] + B*u[t-1])
    # end

    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    J = @objective(neuron, Min, sum(x_cost[t][1] for t in 2:pred) + sum(u_cost[t] for t in 1:pred-1))


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


    zs, us = sim(10, zeros(4), u_clamp=nothing, sample=250);

    zs


    pz = plot(zs, label="z", color="red")
    # pu = plot(us, label="u")
    # plot(pz, pu, layout=[1,1])
end