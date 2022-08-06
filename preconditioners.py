"""
Implementation of different preconditioners. Bempp-cl uses mass-matrix preconditioner by default.
"""

#from .preprocess import PARAMS
import bempp.api
from bempp.api import ZeroBoundaryOperator
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from scipy.sparse import diags, bmat, block_diag
from scipy.sparse.linalg import aslinearoperator

def BlockDiagonal0(dirichl_space, neumann_space, A,es,em,k):

    ILD = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal() 
    KLD = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A 
    VLD = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A 
    IHD = sparse.identity(dirichl_space, neumann_space, neumann_space).weak_form().A.diagonal() 
    if k==0:
        KHD = laplace.double_layer(dirichl_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A 
        VHD = laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A 
    else:
        KHD = modified_helmholtz.double_layer(dirichl_space, neumann_space, neumann_space, k, assembler="only_diagonal_part").weak_form().A 
        VHD = modified_helmholtz.single_layer(neumann_space, neumann_space, neumann_space, k, assembler="only_diagonal_part").weak_form().A 
    
    D11 = (0.5*ILD + KLD)  
    D12 = -VLD           
    D21 = (0.5*IHD - KHD) 
    D22 = (em/es)*VHD        
    
    DA = 1/(D11*D22-D21*D12)
    DI11 = D22*DA
    DI12 = -D12*DA
    DI21 = -D21*DA
    DI22 = D11*DA
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12)],
                              [diags(DI21), diags(DI22)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)

def BlockDiagonal(dirichl_space, neumann_space, A,es,em,k):

    ILD = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal() 
    KLD = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    Z12 = ZeroBoundaryOperator(neumann_space, dirichl_space, dirichl_space).weak_form().A 
    Z21 = ZeroBoundaryOperator(dirichl_space, neumann_space, neumann_space).weak_form().A 
    if k==0:
        VHD = laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A 
    else: 
        VHD = modified_helmholtz.single_layer(neumann_space, neumann_space, neumann_space, k, assembler="only_diagonal_part").weak_form().A 
    
    D11 = (0.5*ILD + KLD)  
    D22 = (em/es)*VHD           
    DI11 = (1/D11)
    DI22 = (1/D22)
    
    block_mat_precond = bmat([[diags(DI11), Z12],
                              [Z21, diags(DI22)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)
