function eco_test
N = 9;
% fft test
A(1:N,1:N) = single(0);
for j= 1:N
   for i = 1:N
        A(j,i) = single((i-1)+(j-1)*N);
   end
end
B(1:N,1:N) = single(0);
for j= 1:N
   for i = 1:N
        B(j,i) = -single((i-1));
   end
end
A
tic
fft2(A)
toc
ifft2(A)
toc
fft2(A)
toc
% Elapsed time is 0.001175 seconds. ~ 1ms.

z1 = complex(A, A)
z2 = complex(A, B)

% complexDotMultiplication test
z1 .* z2

% complexDotDivision test
z1 ./ z2

% complexMartrixMultiplication test
z1*z2

% complexConvolution test
N = 14;
% fft test
C(1:N,1:N) = single(0);
for j= 1:N
   for i = 1:N
        C(j,i) = single((i-1)+(j-1)*N);
   end
end
z1 = complex(C, C)
z2 = complex(A, B)
convn(z1, z2)
convn(z1, z2, 'valid')

end
