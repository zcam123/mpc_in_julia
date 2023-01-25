#function to compute Kalman gain based on measurement uncertainty (r) and previous estimate uncertainty (p)
function kalman_gain(r, old_p)
    K = old_p / (old_p + r)
    return K
end

#function to take previous state estimate and new measurement and compute value based on K
function state_update(old_state_est, meas)
    return old_state_est + K*(meas - old_state_est)
end

#function to update estimate uncertainty based on K 
function covariance_update(old_p, K)
    return (1-K)*old_p
end

#function to extrapolate state based on dynamic model
function state_extrapolation(curr_est, A, B, u, sample)
    x = inv(C)*curr_est
    for _ in 1:sample #should this be sample minus one?
        x = A*x + B*u
    end
    return C*x
end

#function to extrapolate estimate uncertainty based on dynamic model and process noise (q)
function covariance_extrapolation(p_curr, q)
    return p_curr + q #needs to be updated to reflect dynamic model
end

#function to call the above functions given an LFP measurement and return
#the best estimate of the current state along with the other outputs which are needed for
#further iterations of the kalman filter
function kalman_filter(meas, old_p, old_state_est, r, q, A, B, u, sample)
    #compute the Kalman gain
    K = kalman_gain(r, old_p)
    #update estimate uncertainty
    p_curr = covariance_update(old_p, K)
    #update the state
    curr_est = state_update(old_state_est, meas)
    #extrapolate the state
    future_est = state_extrapolation(curr_est, A, B, u, sample)
    #extrapolate estimate uncertainty
    p_future = covariance_extrapolation(p_curr, q)

    return curr_est, p_curr, future_est, p_future
end
