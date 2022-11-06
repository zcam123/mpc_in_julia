
using LinearAlgebra
using ModelingToolkit, Optimization, OptimizationOptimJL, ForwardDiff

#define system
@parameters A[1:4,1:4] B[1:4,1:1] C[1:1,1:4] D T=3 Q[1:4,1:4] R[1:1]
@variables X[1:4,1:3] U[1:1,1:3]
p = [
    A => [1 -6.66e-13 -2.03e-9 -4.14e-6;
        9.83e-4 1 -4.09e-8 -8.32e-5;
        4.83e-7 9.83e-4 1 -5.34e-4;
        1.58e-10 4.83e-7 9.83e-4 .9994;
    ]
    B => [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]
    C => [-.0096 .0135 .005 -.0095]
    D => [0.0]
    T => 3
    Q => C*C'
    R => [1e-4]
]

#for getting firing rate Z - y[1] used as julia gives a vector for y during following processes by default
function y_to_z(y)
    return exp(61.4*y[1] - 5.468);
end

u0 = [
    X => zeros(4, 3)
    U => zeros(1, 3)
]

L = tr(sum(X'*Q*X + U'*R*U))

my_cons = X[:, 2:end] ~ A*X[:, 1:end-1] + B*U[:, 1:end-1]

@named sys = OptimizationSystem(L, [X, U], [A, B, C, D, T, Q, R], constraints=[my_cons])

prob = OptimizationProblem(sys, u0, p, grad=true, hess=true)
sol = solve(prob, Newton())


using Plots