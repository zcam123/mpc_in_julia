using Convex, SCS

#x_desired
x_desired = [75.03983654, 481.45636789, 540.76706284, 886.34429676]


initial_x = [0,0,0,0];
initial_u = [5]

T = 100 # The number of timesteps
h = 0.001 # The time between time intervals
#matrices for dynamics        
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
     9.83e-4 1 -4.09e-8 -8.32e-5;
     4.83e-7 9.83e-4 1 -5.34e-4;
     1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]

C = [-.0096 .0135 .005 -.0095]

# Declare state variables
x_vec = Variable(4, T)
u_policy = Variable(1, T)
J = Variable(1, T)

# Create a problem instance
mu = 1

# Add constraints on our variables
constraints = Constraint[
    x_vec[:, i+1] == A*x_vec[:, i] + B*u_policy[:, i] for i in 1:T-1
]

yD=0.06284303;

for i in 1:T
    push!(constraints, J[i] == yD - C*x_vec[i]) 
end

# Add initial constraints
push!(constraints, x_vec[:, 1] == initial_x)
push!(constraints, u_policy[1] == initial_u)

# Solve the problem
problem = minimize(sumsquares(J), constraints)
solve!(problem, SCS.Optimizer; silent_solver = true)

x_vec