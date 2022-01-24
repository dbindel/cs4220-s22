# Different matmul organizations

# We need LinearAlgebra for the norm function, and Printf for printf.
using LinearAlgebra
using Printf


function matmul_ijk(A, B)
    m, n = size(A)
    n, p = size(B)  
    C = zeros(eltype(A), m, p)
    for i = 1:m
        for j = 1:p
            for k = 1:n
                C[i,k] += A[i,j]*B[j,k]
            end
        end
    end
    return C
end


function matmul_kji(A, B)
    m, n = size(A)
    n, p = size(B)  
    C = zeros(eltype(A), m, p)
    for k = 1:n
        for j = 1:p
            for i = 1:m
                C[i,k] += A[i,j]*B[j,k]
            end
        end
    end
    return C
end


function matmul_timing(n)

  A = rand(n,n)
  B = rand(n,n)
  
  ntrials = ceil(1e7 / n^3)
  gflops = 2 * n^3 * ntrials / 1e9
  
  C = A*B
  Cr = matmul_ijk(A, B)
  Cc = matmul_kji(A, B)
  @printf("%d: %e %e\n", n, norm(C-Cr), norm(C-Cc))

  t1 = @elapsed begin
    for trial = 1:ntrials
      C = A*B
    end
  end

  t2 = @elapsed begin
    for trial = 1:ntrials
      C = matmul_ijk(A, B)
    end
  end

  t3 = @elapsed begin
    for trial = 1:ntrials
      y = matmul_kji(A, B)
    end
  end

  @printf("%d: %g %g %g\n", n, t1, t2, t3)
  return @sprintf("%d %g %g %g", n, gflops/t1, gflops/t2, gflops/t3);
end


# Write a data file with timings for a range of sizes
open("../data/matmul_time_jl.dat", "w") do f
  for d = 128:128:1024
    println(f, matmul_timing(d-1))
    println(f, matmul_timing(d))
    println(f, matmul_timing(d+1))
  end
end
