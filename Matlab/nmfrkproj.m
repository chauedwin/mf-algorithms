function [lbest, rbest, lowesterror] = nmfrkproj(data, k, niter, kiter, reinit) 
    finalerror = -1;
    
    seqerror = double.empty(niter, 0);
    lowesterror = double.empty(1, 0);
    datadim = size(data);
    
    lbest = rand(datadim(1), k);
    rbest = rand(k, datadim(2));
    
    for l = 1:reinit
        lfactor = rand(datadim(1), k);
        rfactor = rand(k, datadim(2));
        
        for i = 1:niter
            approx = lfactor*rfactor;
            
            row =  randsample(datadim(1), 1, true, wsample(approx, 2));
            col = randsample(datadim(2), 1, true, wsample(approx, 1));
            
            for j = 1:kiter
                kaczrow = randsample(datadim(1), 1, true, wsample(lfactor, 2);
                kaczcol = randsample(datadim(2), 1, true, wsample(rfactor, 1);
                
                lfactor(row, :) = lfactor(row, :) + (data(row, kaczcol) - lfactor(row, :)*rfactor(:, kaczcol)) / norm(rfactor(:, kaczcol)^2 * rfactor(:, kaczcol));
                rfactor(:, col) = rfactor(:, col) + (data(kaczrow, col) - lfactor(kaczrow, :)*rfactor(:, col)) / norm(lfactor(kaczrow, :)^2 * lfactor(kaczrow, :));
                
                lfactor(row, :) = softproj(lfactor(row, :), j);
                rfactor(:, col) = softproj(rfactor(:, col), j);
            end
            
            seqerror(i) = norm(data - lfactor*rfactor) / norm(data);
        end
        
        if (finalerror == -1)
            lowesterror = seqerror;
            lbest = lfactor;
            rbest = rfactor;
        elseif (finalerror > seqerror(niter)
            finalerror = seqerror(niter);
            lowesterror = seqerror;
            lbest = lfactor;
            rbest = rfactor;
        end
    end
end