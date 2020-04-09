% mode = 1 calculates the norm of each column
% mode = 2 calculates norm of each row
function [prob] = wsample(data, mode)

    % calculate rowwise or columnwise norm
    prob = vecnorm(data, 2, mode);
    
    % scale by the sum of norms to get probabilities
    prob = prob / sum(prob);
    
end

