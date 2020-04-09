function [vec] = softproj(vec, i)
    vec(vec < 0) = (-1 / sqrt(i));
end