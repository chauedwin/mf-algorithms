rng(1);

col1 = [0; 0; 9; 5; 3; 2; 1; 0; 0; 0; 0; 0];
col2 = [0; 0; 0; 0; 0; 3; 2; 1; 1; 0; 0; 0];
col3 = [0; 5; 5; 6; 6; 7; 4; 2; 1; 0.5; 0; 0];

factors = [col1, col2, col3];
weights = randi([0,1], 3, 10);

X = factors*weights;

% k = 3;
% niter = 1000;
% A = randi([0,1], 12, 3);
% S = randi([0,1], 3, 10);
% 
% for i = 1:k
%     rowcol = randi(3);
%     S(:, rowcol) = pinv(A'*A)*(A'*X(:, rowcol));
%     A(rowcol, :) = (pinv(S*S')*(S*X(rowcol, :)'))';
% end
% 
% approx = A*S;
% 
% error = norm(approx - X) / norm(X)

[A, S, error] = nmfalsproj(X, 3, 100, 10);


