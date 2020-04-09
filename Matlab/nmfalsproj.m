function [lbest, rbest, lowesterror] = nmfalsproj(data, k, niter, reinit) 
    finalerror = -1;
    
    seqerror = double.empty(niter, 0);
    lowesterror = double.empty(1, 0);
    datadim = size(data);
    
    lbest = rand(datadim(1), k);
    rbest = rand(k, datadim(2));
    
    for j = 1:reinit
        lfactor = rand(datadim(1), k);
        rfactor = rand(k, datadim(2));
        
        for i = 1:niter
           %row =  randsample(datadim(1), 1, true, wsample(data, 2));
           %col = randsample(datadim(2), 1, true, wsample(data, 1));
            
           row = randi(datadim(1));
           col = randi(datadim(2));
           
           rfactor(:, col) = pinv(lfactor)*data(:, col);
           lfactor(row, :) = data(row, :)*(pinv(rfactor.').');
           
           rfactor(:, col) = softproj(rfactor(:, col), i);
           lfactor(row, :) = softproj(lfactor(row, :), i);
           
           seqerror(i) = norm(data - lfactor*rfactor) / norm(data);
        end
        
        if (finalerror == -1)
            lowesterror = seqerror;
            lbest = lfactor;
            rbest = rfactor;

        elseif (finalerror > seqerror(niter))
            finalerror = seqerror(niter);
            lowesterror = seqerror;
            lbest = lfactor;
            rbest = rfactor;
        end
    end
    
end

