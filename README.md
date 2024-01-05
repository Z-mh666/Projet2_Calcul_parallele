This project aims to parallelize the 2D finite element code using MPI. We will partition the 2D mesh into 4 or more parts, depending on the number of processes, with each process handling the calculation for its designated portion.

**Objevtives**

-- Parallelize the inversion of the matrix (Jacobi method) and the calculation of L2 norm.//
-- Parallelize the scalar product, as distributing tasks to processes may lead to double-counting the boundaries of each mesh part at least twice.
