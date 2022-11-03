#include "headers.hpp"

extern int myRank;
extern int nbTasks;

//================================================================================
// Solution of the system Au=b with Jacobi
//================================================================================

void jacobi(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit)
{
  if(myRank == 0)
    printf("== jacobi\n");
  
  // Compute the solver matrices
  int size = A.rows();
  Vector Mdiag(size);
  SpMatrix N(size, size);
  for(int k=0; k<A.outerSize(); ++k){
    for(SpMatrix::InnerIterator it(A,k); it; ++it){
      if(it.row() == it.col())
        Mdiag(it.row()) = it.value();
      else
        N.coeffRef(it.row(), it.col()) = -it.value();
    }
  }
  exchangeAddInterfMPI(Mdiag, mesh);

  // Jacobi solver
  double residuNorm = 1e2;
  int it = 0;
  Vector w(size);
  Vector Au(size);
  while (it < maxit){
    
    // Compute N*u
    Vector Nu = N*u;
    exchangeAddInterfMPI(Nu, mesh);
    
    // Update field
    for(int i=0; i<size; i++){
      u(i) = 1/Mdiag(i) * (Nu(i) + b(i));
    }

    // Update residual and iterator
    
    if((it % 10) == 0){
      residuNorm = 0;
      Au = A*u;
      exchangeAddInterfMPI(Au, mesh);
      
      for (int i=0;i<size;i++){
        w(i) = b(i)-Au(i);
        residuNorm += w(i)*w(i);
      }

      residuNorm = sqrt(residuNorm);

      if(myRank == 0){
        printf("\r   %i %e", it, residuNorm);
      }
    }
    it++;
  }
  
  if(myRank == 0){
    printf("\r   -> final iteration: %i (prescribed max: %i)\n", it, maxit);
    printf("   -> final residual: %e (prescribed tol: %e)\n", residuNorm, tol);
  }
  saveToMsh(w, mesh, "merde", "benchmark/merde.msh");
}

void normL2(SpMatrix& M,Vector& v,Mesh& mesh)
{
  double  r = 0;
  Vector Mv = M*v;
  exchangeAddInterfMPI(Mv,mesh);
  for (int i=0;i<nbTasks;i++){
    double rr = v.dot(Mv);
    if (myRank == i){
      MPI_Reduce(&rr, &r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  }
  if (myRank == 0){
    printf("   -> final L2 error: %e\n",sqrt(r));
  }
}


