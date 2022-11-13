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
  Vector res(size);   //residu
  Vector Au(size);    //A*u

  while (/*residuNorm>tol &&*/ it < maxit){
    
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
      res = b-Au;
      double r = para_ps(res,res,mesh);
      MPI_Reduce(&r, &residuNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
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
  
}

void normL2(SpMatrix& M,Vector& v,Mesh& mesh)
{
  double  r = 0;
  Vector Mv = M*v;
  exchangeAddInterfMPI(Mv,mesh);
  double rr = para_ps(v,Mv,mesh);
  MPI_Reduce(&rr, &r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myRank == 0){
    printf("   -> final L2 error: %e\n",sqrt(r));
  }
}

void Gradconj(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit){
  //Phase d'initialisations :
  if(myRank == 0){
    printf("== Conjugate gradient\n");
  }

  // Conjugate gradient solver
  int size = A.rows();
  double residuNorm;
  int it = 0;   //iteration
  Vector res(size);   //residu

  Vector Au = A*u;
  exchangeAddInterfMPI(Au, mesh);
  Vector r = b - Au ;
  Vector p = 1.*r;
  double beta;
  double alpha; 
  double ps_a,ps_b,ps_c;
  double ps_d;

  while ( it < maxit) {
    Vector Ap = A*p ;
    exchangeAddInterfMPI(Ap, mesh);
    ps_a = para_ps(r,p,mesh);
    ps_b = para_ps(Ap,p,mesh);
    alpha = ps_a/ps_b;
    //alpha = r.dot(p)/Ap.dot(p);
    u = u + alpha*p;
    r = r - alpha*Ap;
    ps_c = para_ps(r,r,mesh);
    Vector d = r+alpha*Ap;
    ps_d = para_ps(d,d,mesh);
    beta = ps_c/ps_d;
    //beta = -r.dot(r)/((r+alpha*Ap).dot(r+alpha*Ap));
    //Vector Ar = A*r;
    //exchangeAddInterfMPI(Ar, mesh);
    //ps_c = para_ps(Ar,p,mesh);
    //beta = -(Ar.dot(p))/(Ap.dot(p));
    //beta = -ps_c/ps_b;
    p = r + beta*p;
    
    

    // Update residual and iterator
    if((it % 10) == 0){
      residuNorm = 0;
      Au=A*u;
      exchangeAddInterfMPI(Au, mesh);
      for (int i=0;i<size;i++){
        res(i) = b(i)-Au(i);
        residuNorm += res(i)*res(i);
      }
    
      residuNorm=sqrt(residuNorm);
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
}




