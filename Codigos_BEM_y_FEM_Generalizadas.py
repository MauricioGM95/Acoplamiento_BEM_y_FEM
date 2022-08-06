#Seccion de importacion de librerias
import bempp.api
import dolfin
from dolfin import Mesh
import numpy as np 
import trimesh
import time
from readoff import *
from readpqr import *

################################################## Seccion de clases ###################################################

# Clase para mostrar las iteraciones del GMRES
class gmres_counter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if (self.niter /100) == (self.niter // 100):
                print(self.niter,str(rk))                                    
            #if self._disp:
            #    print('iter %3i\trk = %s' % (self.niter, str(rk)))   
                
#Caso ei variable por radio
class Fun_ei_Radio(dolfin.UserExpression):  #funcion dolfin
    def __init__(self,PC,R,em,ei,**kwargs):
        super().__init__(**kwargs)
        self.PC = PC
        self.R = R
        self.em = em
        self.ei = ei
    def eval(self, v, x):
        PC = self.PC
        R = self.R
        em = self.em
        ei = self.ei
        d0 = np.linalg.norm( x - PC, axis=1) #distancia minima a los atomos de la molecula 
        i0 = np.argsort(d0)        
        ro = (R[i0[0]]/d0[i0[0]]+R[i0[1]]/d0[i0[1]]+R[i0[2]]/d0[i0[2]])/(1/d0[i0[0]]+1/d0[i0[1]]+1/d0[i0[2]])+1.4 #Radio efectivo
        ef = (ei-1+55/26)-(5/13)*ro #epsilon efectivo por vertice
        if(ef>em):
            v[0] = ef #Distribucion por radio por tetrahedro
        else:
            v[0] = em #En caso que ef sea menor que em, se cambia por em por sentido fisico de la permitividad
    def value_shape(self):
        return () 

#Caso ei variable por interpolacion lineal
class Fun_ei_Lineal(dolfin.UserExpression):  #funcion dolfin
    def __init__(self,em,es,Ga,Gb,**kwargs):
        super().__init__(**kwargs)
        self.Ga = Ga
        self.Gb = Gb
        self.em = em
        self.es = es
    def eval(self, v, x):
        Ga = self.Ga
        Gb = self.Gb
        em = self.em 
        es = self.es 
        da = np.linalg.norm( x - Ga, axis=1) #distancia minima a la malla inferior
        db = np.linalg.norm( x - Gb, axis=1) #distancia minima a la malla superior
        ia = np.argsort(da)
        ib = np.argsort(db)
        ra = (da[ia[0]]+da[ia[1]]+da[ia[2]])/3
        rb = (db[ib[0]]+db[ib[1]]+db[ib[2]])/3
        v[0] = em+(ra/(ra+rb))*(es-em) #Distribucion lineal por tetrahedro
    def value_shape(self):
        return ()  

#Creacion del potencial de Coulomb en forma de clase para FEM/BEM de 3 Terminos.
class Fun_ucfem(dolfin.UserExpression):  #funcion dolfin
    def __init__(self,PC,Q,em,**kwargs):
        super().__init__(**kwargs)
        self.PC = PC
        self.Q = Q
        self.em = em
    def eval(self, v, x):
        PC = self.PC
        Q = self.Q
        em = self.em
        v[0]= (1/(4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    def value_shape(self):
        return () 
    
################################################## Seccion de funciones ###################################################

#Energia de Coulomb    
def ECoulomb(PQR,em): 
    PC,Q,R = readpqr(PQR)
    C0=0.5*332.064*(1/em)
    EC=0
    for i in range(len(PC)):
        for j in range(len(PC)):    
            if i!=j:
                EC=EC+C0*Q[i]*Q[j]/(np.linalg.norm(PC[i]-PC[j], axis=0))
    return EC

##########################################################################################################################

def Caso_BEMBEM(PQR,Malla,es,em,k,Tol,Md,SF,Asb):       
    start = time.time()  
    #Eleccion de modelo en especifico para BEM/BEM
    if Md=='NR':
        print("Caso BEM/BEM(NoRegularizado)")
    elif Md=='R':
        print("Caso BEM/BEM(Regularizado)")
    elif Md=='3T':
        print("Caso BEM/BEM(3Terminos)")
        
    # Eleccion de los ensamblaje de los operadores de forntera    
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
   
    #Datos de la posicion en el espacio, carga y radios de los atomos de la molecula.
    PC,Q,R = readpqr(PQR) #Eleccion del "pqr" de la molecula
        
    #Generar la malla superficial de la molecula
    #En caso que la malla tenga huecos pequeños, con trimesh se obtiene la informacion de la malla original sin los huecos.
    import trimesh
    V1,F1 = read_off(Malla)
    meshSP = trimesh.Trimesh(vertices = V1, faces= F1)
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split))
    
    vertices_split1 = mesh_split[0].vertices 
    faces_split1 = mesh_split[0].faces   
    grid = bempp.api.grid.grid.Grid(vertices_split1.transpose(), faces_split1.transpose())
    
    #Generar espacios de funcionales del potencial y su derivada
    dirichl_space = bempp.api.function_space(grid, "P", 1)   #Potencial electrostatico en la interfaz.
    neumann_space = bempp.api.function_space(grid, "DP", 0)  #Derivada del potencial electrostatico en la interfaz.

    print("DS dofs: {0}".format(dirichl_space.global_dof_count))
    print("NS dofs: {0}".format(neumann_space.global_dof_count))
    
    #Generar los operadores de frontera
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 5
    #Dominio del soluto Ωm
    IL = bempp.api.operators.boundary.sparse.identity(dirichl_space, dirichl_space, dirichl_space) #1
    KL = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=Assemble) #K
    VL = bempp.api.operators.boundary.laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=Assemble) #V
    #Dominio del solvente Ωs
    IH = bempp.api.operators.boundary.sparse.identity(dirichl_space, neumann_space, neumann_space) #1
    if k==0:
        KH = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, neumann_space, neumann_space, assembler=Assemble) #K
        VH = bempp.api.operators.boundary.laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler=Assemble) #V             
    else:
        KH = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space, neumann_space, neumann_space, k, assembler=Assemble) #K
        VH = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space, neumann_space, neumann_space, k, assembler=Assemble) #V
    
    #Creacion de funcion del potencial de Coulomb y su derivada  
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    U_c = bempp.api.GridFunction(dirichl_space, fun=U_c)

    @bempp.api.complex_callable(jit=False)
    def dU_c(x, n, domain_index, result):
        result[:] = -(1/(4.*np.pi*em))   * np.sum( np.dot( x - PC , n)  * Q / (np.linalg.norm( x - PC , axis=1)**3) )
    dU_c = bempp.api.GridFunction(neumann_space, fun=dU_c)
    
    #Construccion del vector derecho
    if Md=='NR':
        if SF==False:
            # Rhs en Ωm
            rhs_M = (U_c).projections(dirichl_space)
            # Rhs en Ωs
            rhs_S = np.zeros(neumann_space.global_dof_count)
            # La conbinacion del rhs
            rhs = np.concatenate([rhs_M, rhs_S])
        else:
            rhs = [IL*U_c, 0*IH*U_c]
            
    elif Md=='R':
        if SF==False:
            # Rhs en Ωm
            rhs_M = np.zeros(dirichl_space.global_dof_count)
            # Rhs en Ωs
            rhs_S = ((KH - 0.5*IH)*U_c).projections(neumann_space) - (em/es)*(VH*dU_c).projections(neumann_space)
            # La conbinacion del rhs
            rhs = np.concatenate([rhs_M, rhs_S])
        else:
            rhs = [0*IL*U_c, (KH - 0.5*IH)*U_c-(em/es)*(VH*dU_c)]

    elif Md=='3T':
        #Generar los operadores de frontera
        #Dominio para el potencial armonico
        KL0 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, neumann_space, neumann_space) #K
        VL0 = bempp.api.operators.boundary.laplace.single_layer(neumann_space, neumann_space, neumann_space) #V
        
        #Calculo de la derivada del potencial armonico con GMRES
        U_h = -U_c
        rhs0 = ((0.5*IH + KL0)*U_h).projections(neumann_space)
        Auh = VL0.weak_form()
        
        count_iterations = gmres_counter()
        from scipy.sparse.linalg import gmres
        Sol, info = gmres(Auh, rhs0, M=None,callback=count_iterations,tol=Tol)
        print("Numero de iteraciones de GMRES de dU_h: {0}".format(count_iterations.niter))

        dU_h = bempp.api.GridFunction(neumann_space, coefficients=Sol)            
        dirichlet_uh_fun = U_h
        neumann_duh_fun = dU_h
        
        if SF==False:
            # Rhs en Ωm
            rhs_M = np.zeros(dirichl_space.global_dof_count)
            # Rhs en Ωs
            rhs_S = -(em/es)*(VH*dU_c).projections(neumann_space) - (em/es)*(VH*dU_h).projections(neumann_space)
            # La conbinacion del rhs
            rhs = np.concatenate([rhs_M, rhs_S])
        else:
            rhs = [0*IL*U_c, -(em/es)*(VH*dU_c)-(em/es)*(VH*dU_h)]

    #Construccion matriz izquierda 2x2
    if SF==False:
        #Posicion de la matriz 2x2
        blocks = [[None,None],[None,None]]    
        blocks[0][0] = (0.5*IL + KL).weak_form()  #0.5+K
        blocks[0][1] = -VL.weak_form()            #-V
        blocks[1][0] = (0.5*IH - KH).weak_form()  #0.5-K
        blocks[1][1] = (em/es)*VH.weak_form()     #V(em/es) 
        
        blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks))
    
    else:    
        #Posicion de la matriz 2x2
        blocks = bempp.api.BlockedOperator(2,2)    
        blocks[0,0] = (0.5*IL + KL)  #0.5+K
        blocks[0,1] = -VL            #-V
        blocks[1,0] = (0.5*IH - KH)  #0.5-K
        blocks[1,1] = VH*(em/es)     #V(em/es)            
                
    #Se resuelve la solucion de la Ecuacion matricial Ax=B
    #Contador de iteraciones
    count_iterations = gmres_counter()
    
    
    from scipy.sparse.linalg import gmres
    if SF==False:
        # Solucion por GMRES
        start1 = time.time()
        soln, info = gmres(blocked, rhs, M=None, callback=count_iterations,tol=Tol)   
        end1 = time.time() 
        curr_time1 = (end1 - start1)  
        print("Numero de iteraciones de GMRES: {0}".format(count_iterations.niter))
        print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))
        TF=count_iterations.niter
        
        #Calcula todo el dominio global del potencial a partir de la solucion del borde calculado
        soln_u = soln[:dirichl_space.global_dof_count]
        soln_du = soln[dirichl_space.global_dof_count:]
        # Solucion para datos de Dirichlet 
        dirichlet_fun = bempp.api.GridFunction(dirichl_space, coefficients=soln_u)
        # Solucion para datos de Neumann 
        neumann_fun = bempp.api.GridFunction(neumann_space, coefficients=soln_du)
        
    else:
        # Solucion por GMRES
        start1 = time.time()
        soln, info, it_count = bempp.api.linalg.gmres(blocks, rhs, return_iteration_count=True, use_strong_form=True,tol=Tol)
        end1 = time.time() 
        curr_time1 = (end1 - start1)  
        print("Numero de iteraciones de GMRES: {0}".format(it_count))
        print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))
        TF=it_count
        
        #Calcula todo el dominio global del potencial a partir de la solucion del borde calculado
        dirichlet_fun = soln[0]
        neumann_fun = soln[1]
    
    #Calculo del potencial en la posicion de los atomos de la molecula    
    VF = bempp.api.operators.potential.laplace.single_layer(neumann_space, np.transpose(PC)) 
    KF = bempp.api.operators.potential.laplace.double_layer(dirichl_space, np.transpose(PC))
    if Md=='3T':
        uF = VF*(neumann_fun+ neumann_duh_fun) - KF*(dirichlet_fun+dirichlet_uh_fun)
    else:
        uF = VF*neumann_fun - KF*dirichlet_fun

    #Resultado de la energia de solvatacion total
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Energia de Solvatacion en BEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )
    
    end = time.time()
    curr_time = (end - start)   
    print("Tiempo total: {:5.2f} [s]".format(curr_time))

    return E_Solv, curr_time, curr_time1, TF 
    
############################################################################################################################ 

def Caso_FEMBEM(PQR,Malla,es,em,k,Tol,Md,Asb):    
    start = time.time()  
    #Eleccion de modelo en especifico para BEM/BEM
    if Md=='NR':
        print("Caso BEM/BEM(NoRegularizado)")
    elif Md=='R':
        print("Caso BEM/BEM(Regularizado)")
    elif Md=='3T':
        print("Caso BEM/BEM(3Terminos)")
        
    # Eleccion de los ensamblaje de los operadores de forntera    
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
          
    #Datos de la posicion en el espacio, carga y radios de los atomos de la molecula.
    PC,Q,R = readpqr(PQR) #Eleccion del "pqr" de la molecula

    #Generar la malla volumetrica de la molecula
    from dolfin import Mesh
    mesh = Mesh(Malla) #Creacion de la malla volumetrica
    
    #Generar espacios de funcionales del potencial en Fem y su derivada en Bem
    from bempp.api.external import fenics

    fenics_space = dolfin.FunctionSpace(mesh, "CG", 1) #Potencial electrostatico en la interfaz y dominio del soluto
    trace_space, trace_matrix = \
        fenics.fenics_to_bempp_trace_data(fenics_space) # Espacio de la traza para trabajar en BEM y FEM simultaneamente.
    bempp_space = bempp.api.function_space(trace_space.grid, "DP", 0) #Derivada del potencial electrostatico en la interfaz.

    print("FEM dofs: {0}".format(mesh.num_vertices()))
    print("BEM dofs: {0}".format(bempp_space.global_dof_count))
    print("TRA dofs: {0}".format(trace_space.global_dof_count))

    #Generar operadores de frontera de Fem y Bem
    I1 = bempp.api.operators.boundary.sparse.identity(trace_space, bempp_space, bempp_space)
    mass = bempp.api.operators.boundary.sparse.identity(bempp_space, bempp_space, trace_space)
    if k==0:
        K1 = bempp.api.operators.boundary.laplace.double_layer(trace_space, bempp_space, bempp_space)
        V1 = bempp.api.operators.boundary.laplace.single_layer(bempp_space, bempp_space, bempp_space) 
    else:
        K1 = bempp.api.operators.boundary.modified_helmholtz.double_layer(trace_space, bempp_space, bempp_space, k)
        V1 = bempp.api.operators.boundary.modified_helmholtz.single_layer(bempp_space, bempp_space, bempp_space, k)       

    #Definir espacio funcional de Dolfin
    u = dolfin.TrialFunction(fenics_space)
    v = dolfin.TestFunction(fenics_space)
        
    #Creacion del potencial de Coulomb y su derivada
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    U_c = bempp.api.GridFunction(bempp_space, fun=U_c)

    @bempp.api.complex_callable(jit=False)
    def dU_c(x, n, domain_index, result):
        result[:] = -(1/(4.*np.pi*em))   * np.sum( np.dot( x - PC , n)  * Q / (np.linalg.norm( x - PC , axis=1)**3) )
    dU_c = bempp.api.GridFunction(bempp_space, fun=dU_c)
    
    #Construccion del vector derecho
    if Md=='NR':
        #Constante
        C1=1  
        C2=1  
        # EL rhs en Ωm(FEM)
        rhs_f = dolfin.Constant(0.0)
        rhs_fem = dolfin.assemble(rhs_f*v*dolfin.dx)
        for i in range(len(PC)):
            delta = dolfin.PointSource(fenics_space, dolfin.Point(PC[i]), Q[i]/em) 
            delta.apply(rhs_fem)      
        # El rhs en Ωs(BEM)
        rhs_bem = np.zeros(bempp_space.global_dof_count)
                
    elif Md=='R':
        #Constante
        C1=1 
        C2=1
        #Generar otros operadores de frontera
        I2 = bempp.api.operators.boundary.sparse.identity(bempp_space, bempp_space, bempp_space)
        if k==0:
            K2 = bempp.api.operators.boundary.laplace.double_layer(bempp_space, bempp_space, bempp_space)
        else:
            K2 = bempp.api.operators.boundary.modified_helmholtz.double_layer(bempp_space, bempp_space, bempp_space, k)
        
        # EL rhs en Ωm(FEM)
        rhs_fem = np.zeros(mesh.num_vertices())
        # El rhs en Ωs(BEM)
        rhs_bem = -(em/es)*(V1*dU_c).projections(bempp_space) + ((K2 - 0.5*I2)*U_c).projections(bempp_space)       
       
    elif Md=='3T':
        #Constante
        C1=em
        C2=es
        #Definir espacio funcional de Dolfin#
        uh = dolfin.TrialFunction(fenics_space)

        #Calculo del potencial armonico en FEM con dolfin
        def boundary(x, on_boundary):
            return on_boundary
            
        #Definicion de variables de dolfin
        UCP = Fun_ucfem(PC,Q,em,degree=1)
        u_Dir = dolfin.Expression('C', C=UCP, degree=1, domain = mesh) 
        bc = dolfin.DirichletBC(fenics_space, -u_Dir, boundary)
        a = dolfin.inner(dolfin.grad(uh), dolfin.grad(v))*dolfin.dx
        f = dolfin.Constant(0.0)
        L = f*v*dolfin.dx

        #Solucion de la ecuacion
        uh = dolfin.Function(fenics_space)
        dolfin.solve(a == L, uh, bc)

        # EL rhs en Ωm(FEM)
        from bempp.api.external.fenics import FenicsOperator
        nor = dolfin.FacetNormal(mesh)
        B = FenicsOperator(em*dolfin.dot(dolfin.grad(u),nor)*v*dolfin.ds)  
        B_u_dir = dolfin.dot(em*dolfin.grad(u_Dir),nor)*v*dolfin.ds
        rhs_u_dir = dolfin.assemble(B_u_dir)
        
        rhs_fem = - B.weak_form().A @ uh.vector()[:] - rhs_u_dir 
        # El rhs en Ωs(BEM)
        rhs_bem = -(em/es)*(V1*dU_c).projections(bempp_space) 

    # La combinacion de rhs
    rhs = np.concatenate([rhs_fem, rhs_bem])
    
    #Construccion matriz izquierda 2x2 y del vector derecho
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from bempp.api.external.fenics import FenicsOperator
    from scipy.sparse.linalg.interface import LinearOperator      
    blocks = [[None,None],[None,None]]

    trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix*x)
    A = FenicsOperator((C1*dolfin.inner(dolfin.nabla_grad(u),
                                     dolfin.nabla_grad(v)) ) * dolfin.dx)  

    #Posicion de la matriz 2x2
    blocks[0][0] = A.weak_form()  #A
    blocks[0][1] = -trace_matrix.T * C2* mass.weak_form().A  #-ML
    blocks[1][0] = (0.5*I1 - K1).weak_form() * trace_op #0.5-K
    blocks[1][1] = V1.weak_form()*(em/es)   #V

    blocked = BlockedDiscreteOperator(np.array(blocks))

    #Creacion del precondicionador Matriz de Masa para FEM/BEM
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse import diags
    
    P1 = diags(1./blocked[0,0].A.diagonal())
    P2 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space, bempp_space, bempp_space).weak_form())

    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
    
        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1:])
        return np.concatenate([res1, res2])

    p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))

    #Se resuelve la solucion de la Ecuacion matricial Ax=B
    count_iterations = gmres_counter()

    # Solucion por GMRES
    from scipy.sparse.linalg import gmres
    start1 = time.time()
    soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol)  
    end1 = time.time() 
    curr_time1 = (end1 - start1)  
    print("Numero de iteraciones de GMRES: {0}".format(count_iterations.niter))
    print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))
    
    soln_fem = soln[:mesh.num_vertices()]
    soln_bem = soln[mesh.num_vertices():]
    
    #Calcula todo el dominio global del potencial en la interfaz a partir del potencial de FEM y la derivada de BEM calculado
    if Md=='NR': 
        # Solucion para datos de Dirichlet 
        dirichlet_data = trace_matrix * soln_fem
        dirichlet_fun = bempp.api.GridFunction(trace_space, coefficients=dirichlet_data)
        # Solucion para datos de Neumann
        neumann_fun = bempp.api.GridFunction(bempp_space, coefficients=soln_bem)
    
        slpF = bempp.api.operators.potential.laplace.single_layer(bempp_space, np.transpose(PC)) 
        dlpF = bempp.api.operators.potential.laplace.double_layer(trace_space, np.transpose(PC))
        uF = slpF * neumann_fun - dlpF * dirichlet_fun 
    elif Md=='R':
        #Calcula todo el dominio global del potencial con el potencial de FEM
        u = dolfin.Function(fenics_space)
        u.vector()[:] = np.ascontiguousarray(np.real(soln_fem))
        u.set_allow_extrapolation(True) #Extrapola datos que no estan en el corte#
    elif Md=='3T':
        #Calcula todo el dominio global del potencial con el potencial de FEM
        u = dolfin.Function(fenics_space)
        u.vector()[:] = np.ascontiguousarray(np.real(soln_fem))+ uh.vector()[:]
        u.set_allow_extrapolation(True) #Extrapola datos que no estan en el corte#
       
    #Calculo del potencial en la posicion de los atomos de la molecula
    q_uF = 0
    for i in range(len(PC)):
        if Md=='NR':
            Sum1 = (uF[0][i].real)*Q[i]
        else:
            Sum1 = u(PC[i])*Q[i]
        q_uF = q_uF + Sum1
        
    #Resultado de la energia de solvatacion total    
    E_Solv = 0.5*4.*np.pi*332.064*(q_uF)
    print('Energia de Solvatacion en FEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )
    end = time.time()
    curr_time = (end - start)   
    print("Tiempo total: {:5.2f} [s]".format(curr_time))

    return E_Solv,curr_time, curr_time1, count_iterations.niter  
    
###########################################################################################################################    
    
def Caso_BEMBEMBEM(PQR,Malla1,Malla2,es,ei,em,k,Tol,SF,Asb):            
    start = time.time()  
    print("Caso BEM/BEM/BEM")
    
    # Eleccion de los ensamblaje de los operadores de forntera    
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
    
    #Datos de la posicion en el espacio, carga y radios de los atomos de la molecula. 
    PC,Q,R = readpqr(PQR) #Eleccion del "pqr" de la molecula
    
    #Generar las malla superficiales de la molecula
    #En caso que la malla tenga huecos pequeños, con trimesh se obtiene la informacion de la malla original sin los huecos.
    verts1,faces1 =read_off(Malla1)
    meshSP = trimesh.Trimesh(vertices = verts1, faces= faces1)
    mesh_split = meshSP.split()
    vertices_split1 = mesh_split[0].vertices 
    faces_split1 = mesh_split[0].faces   
    grid1 = bempp.api.grid.grid.Grid(vertices_split1.transpose(), faces_split1.transpose()) #Creacion de la malla superficial interior
    
    verts2,faces2 =read_off(Malla2)
    grid2 = bempp.api.grid.grid.Grid(verts2.transpose(), faces2.transpose()) #Creacion de la malla superficial exterior
    
    #Generar espacios de funcionales del potencial y su derivada para el dominio Ωm y Ωi
    dirichl_space1 = bempp.api.function_space(grid1, "P", 1)  #Potencial electrostatico en la interfaz interior.
    neumann_space1 = bempp.api.function_space(grid1, "DP", 0) #Derivada del potencial electrostatico en la interfaz interior.
    dirichl_space2 = bempp.api.function_space(grid2, "P", 1)  #Potencial electrostatico en la interfaz superior.
    neumann_space2 = bempp.api.function_space(grid2, "DP", 0) #Derivada del potencial electrostatico en la interfaz superior.

    print("DS1 dofs: {0}".format(dirichl_space1.global_dof_count))
    print("NS1 dofs: {0}".format(neumann_space1.global_dof_count))
    print("DS2 dofs: {0}".format(dirichl_space2.global_dof_count))
    print("NS2 dofs: {0}".format(neumann_space2.global_dof_count))
    
    #Generar los operadores de frontera
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 5
    #Operadores de identidad
    I1d = bempp.api.operators.boundary.sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1) # 1
    I1n = bempp.api.operators.boundary.sparse.identity(dirichl_space1, neumann_space1, neumann_space1) # 1
    I2d = bempp.api.operators.boundary.sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2) # 1
    I2n = bempp.api.operators.boundary.sparse.identity(dirichl_space2, neumann_space2, neumann_space2) # 1

    #Dominio del soluto Ωm
    K111 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1) #K
    V111 = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1) #V
    Y121 = bempp.api.ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1) #0
    Z121 = bempp.api.ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1) #0

    #Dominio de la capa intermedia Ωi en la interfaz interior
    K211 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1) #K
    V211 = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, neumann_space1, neumann_space1) #V
    K221 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space1, neumann_space1) #K
    V221 = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space1, neumann_space1) #V

    #Dominio de la capa intermedia Ωi en la interfaz superior
    K212 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space2, dirichl_space2) #K
    V212 = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space2, dirichl_space2) #V
    K222 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2) #K
    V222 = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2) #V

    #Dominio del solvente Ωs
    Y312 = bempp.api.ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2) #0
    Z312 = bempp.api.ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2) #0
    if k==0:
        K322 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2) #K
        V322 = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space2, neumann_space2) #V    
    else:        
        K322 = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, k) #K
        V322 = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, k) #V
    
    #Creacion de funcion del potencial de Coulomb    
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uc1 = bempp.api.GridFunction(dirichl_space1, fun=U_c) 

    #Construccion del vector derecho
    if SF==False:
        # Rhs en Ωm
        rhs_M = (Uc1).projections(dirichl_space1) # uc
        # Rhs en Ωi en interfaz inferior
        rhs_I1 = np.zeros(neumann_space1.global_dof_count) #0
        # Rhs en Ωi en interfaz superior
        rhs_I2 = np.zeros(dirichl_space2.global_dof_count) #0
        # Rhs en Ωs
        rhs_S = np.zeros(neumann_space2.global_dof_count) #0
        # La conbinacion del rhs
        rhs = np.concatenate([rhs_M, rhs_I1, rhs_I2, rhs_S])
    else:
        Uc2 = bempp.api.GridFunction(dirichl_space2, fun=U_c) 
        rhs = [I1d*Uc1, 0*I1n*Uc1, 0*I2d*Uc2, 0*I2n*Uc2] 
   
    #Construccion matriz izquierda 4x4
    if SF==False:
        blocks = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]] 

        #Posicion de la matriz 4x4
        blocks[0][0] = (0.5*I1d+K111).weak_form()  # 0.5+K  
        blocks[0][1] = -V111.weak_form()           # -V
        blocks[0][2] = Y121.weak_form()            # 0
        blocks[0][3] = Z121.weak_form()            # 0

        blocks[1][0] = (0.5*I1n-K211).weak_form()  # 0.5-K
        blocks[1][1] = (em/ei)*V211.weak_form()    # (em/ei)V
        blocks[1][2] = K221.weak_form()            # K
        blocks[1][3] = -V221.weak_form()           # -V

        blocks[2][0] = -K212.weak_form()           # -K
        blocks[2][1] = (em/ei)*V212.weak_form()    # (em/ei)V
        blocks[2][2] = (0.5*I2d+K222).weak_form()  # 0.5+K
        blocks[2][3] = -V222.weak_form()           # -V

        blocks[3][0] = Y312.weak_form()            # 0
        blocks[3][1] = Z312.weak_form()            # 0
        blocks[3][2] = (0.5*I2n-K322).weak_form()  # 0.5-K
        blocks[3][3] = (ei/es)*V322.weak_form()    # (ei/es)V

        blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks)) 

    else:
        blocks = bempp.api.BlockedOperator(4,4)  

        #Posicion de la matriz 4x4
        blocks[0,0] = (0.5*I1d+K111) # 0.5+K  [0][0]
        blocks[0,1] = -V111          # -V
        blocks[0,2] = Y121           # 0
        blocks[0,3] = Z121           # 0

        blocks[1,0] = (0.5*I1n-K211) # 0.5-K
        blocks[1,1] = (em/ei)*V211   # (em/ei)V
        blocks[1,2] = K221           # K
        blocks[1,3] = -V221          # -V

        blocks[2,0] = -K212          # -K
        blocks[2,1] = (em/ei)*V212   # (em/ei)V
        blocks[2,2] = (0.5*I2d+K222) # 0.5+K
        blocks[2,3] = -V222          # -V

        blocks[3,0] = Y312           # 0
        blocks[3,1] = Z312           # 0
        blocks[3,2] = (0.5*I2n-K322) # 0.5-K
        blocks[3,3] = (ei/es)*V322   # (ei/es)V
    
    #Se resuelve la solucion de la Ecuacion matricial Ax=B
    #Contador de iteraciones
    count_iterations = gmres_counter()
           
    from scipy.sparse.linalg import gmres
    if SF==False:
        # Solucion por GMRES
        start1 = time.time()
        soln, info = gmres(blocked, rhs, M=None, callback=count_iterations,tol=Tol)  #Sin Precondicionador#
        end1 = time.time() 
        
        # Tiempo de resolver la ecuacion
        curr_time1 = (end1 - start1) 
        print("Numero de iteraciones de GMRES: {0}".format(count_iterations.niter))
        print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))
        TF=count_iterations.niter 
              
        soln_u1 = soln[:dirichl_space1.global_dof_count]
        soln_du1 = soln[dirichl_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count]
        soln_u2 =  soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count]
        soln_du2 = soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count:]
       
        #Calcula todo el dominio global del potencial a partir de la solucion de los borde de ambas interfaces.
        # Solucion de la funcion con Dirichlet en el borde inferior
        dirichlet_fun1 = bempp.api.GridFunction(dirichl_space1, coefficients=soln_u1)
        # Solucion de la funcion con Neumann en el borde inferior
        neumann_fun1 = bempp.api.GridFunction(neumann_space1, coefficients=soln_du1)
        # Solucion de la funcion con Dirichlet en el borde superior
        dirichlet_fun2 = bempp.api.GridFunction(dirichl_space2, coefficients=soln_u2)
        # Solucion de la funcion con Neumann en el borde superior
        neumann_fun2 = bempp.api.GridFunction(neumann_space2, coefficients=soln_du2)
        
    else:
        # Solucion por GMRES
        start1 = time.time()
        soln, info, it_count = bempp.api.linalg.gmres(blocks, rhs, return_iteration_count=True, use_strong_form=True,tol=Tol)
        end1 = time.time() 
        
        # Tiempo de resolver la ecuacion
        curr_time1 = (end1 - start1)        
        print("Numero de iteraciones de GMRES: {0}".format(it_count))
        print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))
        TF=it_count
        
        #Calcula todo el dominio global del potencial a partir de la solucion de los borde de ambas interfaces.
        dirichlet_fun1 = soln[0] 
        neumann_fun1 = soln[1] 
        dirichlet_fun2 = soln[2] 
        neumann_fun2 = soln[3]  
   
    #Calculo del potencial en la posicion de los atomos de la molecula
    VF1 = bempp.api.operators.potential.laplace.single_layer(neumann_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(dirichl_space1, np.transpose(PC))
    uF = VF1*neumann_fun1 - KF1*dirichlet_fun1 
    
    #Resultado de la energia de solvatacion total  
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Energia de Solvatacion en BEM/BEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv))
    
    end = time.time()
    curr_time = (end - start)   
    print("Tiempo total: {:5.2f} [s]".format(curr_time))

    return E_Solv,curr_time, curr_time1, TF
    
########################################################################################################################################
    
def Caso_BEMFEMBEM(PQR,Malla,es,ei,em,k,Tol,Va,Asb):
    start = time.time()         
    
    # Eleccion del modelo de permitividad que se va a trabajar
    if Va=='C':
        print("Caso BEM/FEM/BEM(Constante)")
    elif Va=='VR':
        print("Caso BEM/FEM/BEM(Variable_por_Radio)")
    elif Va=='VL':
        print("Caso BEM/FEM/BEM(Variable_Lineal)")  
       
    # Eleccion de los ensamblaje de los operadores de forntera    
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
        
    #Datos de la posicion en el espacio, carga y radios de los atomos de la molecula. 
    PC,Q,R = readpqr(PQR) #Eleccion del "pqr" de la molecula
    
    #Importar la malla volumetrica del cascaron 
    mesh = Mesh(Malla) #Creacion de la malla volumetrica
    
    #Generar espacios de funcionales globales del potencial en Fem y su derivada en Bem
    from bempp.api.external import fenics
    fenics_space = dolfin.FunctionSpace(mesh, "CG", 1) #Potencial electrostatico en la interfaz y dominio del soluto
    trace_space, trace_matrix = \
        fenics.fenics_to_bempp_trace_data(fenics_space) # Espacio de la traza global para trabajar en BEM y FEM simultaneamente
    
    #Proceso para separar el trace_space para el caso de la malla inferior y la malla superior de forma individual

    #Codigo para identificar vertices y caras de la malla superior e inferior
    faces0 = trace_space.grid.elements
    vertices0 = trace_space.grid.vertices
    meshSP = trimesh.Trimesh(vertices = vertices0.transpose(), faces= faces0.transpose())
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split))
    V1 = len(mesh_split[0].vertices)
    F1 = len(mesh_split[0].faces)

    #Obtencion de la malla superficial inferior
    faces1 = faces0.transpose()[:F1]
    vertices1 = vertices0.transpose()[:V1]
    grid1 = bempp.api.grid.grid.Grid(vertices1.transpose(), faces1.transpose())
    bempp_space1 = bempp.api.function_space(grid1, "DP", 0) #Derivada del potencial electrostatico en la interfaz inferior
    trace_space1 = bempp.api.function_space(grid1, "P", 1)  #Espacio de la traza para trabajar en BEM inferior y FEM simultaneamente.

    #Obtencion de la malla superficial superior
    faces2 = faces0.transpose()[F1:]
    vertices2 = vertices0.transpose()[V1:]
    grid2 = bempp.api.grid.grid.Grid(vertices2.transpose(), (faces2-len(vertices1)).transpose())
    bempp_space2 = bempp.api.function_space(grid2, "DP", 0) #Derivada del potencial electrostatico en la interfaz superior
    trace_space2 = bempp.api.function_space(grid2, "P", 1)  #Espacio de la traza para trabajar en BEM superior y FEM simultaneamente.

    #Visualizacion de elementos
    print("FEM dofs: {0}".format(mesh.num_vertices()))
    print("BEM1 dofs: {0}".format(bempp_space1.global_dof_count))
    print("BEM2 dofs: {0}".format(bempp_space2.global_dof_count))
    print("Tra1 dofs: {0}".format(trace_space1.global_dof_count))
    print("Tra2 dofs: {0}".format(trace_space2.global_dof_count))
    print("TraL dofs: {0}".format(trace_space.global_dof_count))
    
    #Proceso para separar el trace_matrix para el caso de la malla inferior y la malla superior de forma individual
    Nodos = np.zeros(trace_space.global_dof_count)
    Lista_Vertices = []

    #Procedimeito para ubicar los vertices del trace inferior en trace global
    for i in range(len(trace_space1.grid.vertices.T)):
        valores = np.linalg.norm(trace_space1.grid.vertices[:, i] - trace_space.grid.vertices.T,axis= 1)
        index = np.argmin(valores)
        Lista_Vertices.append(index) 
        
    Nodos[Lista_Vertices] = 1
    trace_matrix1 = trace_matrix[Nodos.astype(bool)]
    trace_matrix2 = trace_matrix[np.logical_not(Nodos)]
    
    #Generar los operadores de frontera
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 5
    #Operadores de identidad
    I1 = bempp.api.operators.boundary.sparse.identity(trace_space1, bempp_space1, bempp_space1) # 1
    I2 = bempp.api.operators.boundary.sparse.identity(trace_space2, bempp_space2, bempp_space2) # 1
    mass1 = bempp.api.operators.boundary.sparse.identity(bempp_space1, trace_space1, trace_space1) # 1
    mass2 = bempp.api.operators.boundary.sparse.identity(bempp_space2, trace_space2, trace_space2) # 1

    #Dominio del soluto Ωm en BEM
    K1 = bempp.api.operators.boundary.laplace.double_layer(trace_space1, bempp_space1, bempp_space1, assembler=Assemble) #K
    V1 = bempp.api.operators.boundary.laplace.single_layer(bempp_space1, bempp_space1, bempp_space1, assembler=Assemble) #V
    Z1 = bempp.api.ZeroBoundaryOperator(bempp_space2, bempp_space1, bempp_space1) #0

    #Dominio del solvente Ωs en BEM
    Z2 = bempp.api.ZeroBoundaryOperator(bempp_space1, bempp_space2, bempp_space2) #0
    if k==0:
        K2 = bempp.api.operators.boundary.laplace.double_layer(trace_space2, bempp_space2, bempp_space2, assembler=Assemble) #K
        V2 = bempp.api.operators.boundary.laplace.single_layer(bempp_space2, bempp_space2, bempp_space2, assembler=Assemble) #V
    else:
        K2 = bempp.api.operators.boundary.modified_helmholtz.double_layer(trace_space2, bempp_space2, bempp_space2, k, assembler=Assemble) 
        V2 = bempp.api.operators.boundary.modified_helmholtz.single_layer(bempp_space2, bempp_space2, bempp_space2, k, assembler=Assemble)
    
    #Definir el espacio funcional de Dolfin
    u = dolfin.TrialFunction(fenics_space)
    v = dolfin.TestFunction(fenics_space)
    
    #Creacion de funcion del potencial de Coulomb    
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uca = bempp.api.GridFunction(bempp_space1, fun=U_c)
        
    #Construccion del vector derecho
    # Rhs en Ωm en BEM
    rhs_bem1 = (Uca).projections(bempp_space1)
    # Rhs en Ωi en FEM
    rhs_fem =  np.zeros(mesh.num_vertices()) 
    # Rhs en Ωs en BEM
    rhs_bem2 = np.zeros(bempp_space2.global_dof_count) 
    # La combinacion de rhs
    rhs = np.concatenate([rhs_bem1, rhs_fem, rhs_bem2])
    
    #Eleccion de la variable ei
    if Va=='C':
        EI = ei 
    elif Va=='VR':
        EI =Fun_ei_Radio(PC,R,em,ei,degree=0)
    elif Va=='VL':
        G0 = mesh.coordinates()  #Lista de posicion de los vertices del cascaron
        Ga = G0[:trace_space1.global_dof_count] #Lista de posicion de los vertices de la malla inferior
        Gb = G0[trace_space1.global_dof_count : trace_space1.global_dof_count+ trace_space2.global_dof_count] #Lista de posicion de los vertices de  de la malla superior      
        EI =Fun_ei_Lineal(em,es,Ga,Gb,degree=0)  
    
    #Construccion matriz izquierda 3x3
    from bempp.api.external.fenics import FenicsOperator
    from scipy.sparse.linalg import LinearOperator
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    blocks = [[None,None,None],[None,None,None],[None,None,None]]

    trace_op1 = LinearOperator(trace_matrix1.shape, lambda x:trace_matrix1*x)
    trace_op2 = LinearOperator(trace_matrix2.shape, lambda x:trace_matrix2*x)
    A = FenicsOperator(EI*(dolfin.inner(dolfin.nabla_grad(u),
                                 dolfin.nabla_grad(v)) ) * dolfin.dx)
    
    #Posicion de la matriz 3x3
    blocks[0][0] = V1.weak_form()                     # -V
    blocks[0][1] = (0.5*I1-K1).weak_form()*trace_op1  # 0.5+K
    blocks[0][2] = Z1.weak_form()                     # 0
 
    blocks[1][0] = -trace_matrix1.T *em*mass1.weak_form().A   # -Mg1
    blocks[1][1] =  A.weak_form()                            # A
    blocks[1][2] = -trace_matrix2.T *es*mass2.weak_form().A   # -Mg2
 
    blocks[2][0] = Z2.weak_form()                     # 0
    blocks[2][1] = (0.5*I2-K2).weak_form()*trace_op2  # 0.5-K
    blocks[2][2] = V2.weak_form()                     # V

    blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks)) 
    
    #Creacion del precondicionador Matriz de Masa para BEM/FEM/BEM
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse import diags

    P2 = diags(1./blocked[1,1].A.diagonal())

    P1 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space1, bempp_space1, bempp_space1).weak_form())

    P3 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space2, bempp_space2, bempp_space2).weak_form())

    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        m3 = P3.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
        n3 = P3.shape[1]
 
        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1: n1+n2])
        res3 = P3.dot(x[n1+n2:])
        return np.concatenate([res1, res2, res3])

    p_shape = (P1.shape[0] + P2.shape[0] + P3.shape[0], P1.shape[1] + P2.shape[1] + P3.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))
    
    #Se resuelve la solucion de la Ecuacion matricial Ax=B
    count_iterations = gmres_counter()  
    
    # Solucion por GMRES
    from scipy.sparse.linalg import gmres
    start1 = time.time()
    soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, maxiter=None) 
    end1 = time.time() 
    curr_time1 = (end1 - start1)  
    print("Numero de iteraciones de GMRES: {0}".format(count_iterations.niter)) 
    print("Tiempo total en GMRES: {:5.2f} [s]".format(curr_time1))

    soln_bem1 = soln[:bempp_space1.global_dof_count]
    soln_fem  = soln[bempp_space1.global_dof_count : bempp_space1.global_dof_count + mesh.num_vertices()]
    soln_bem2 = soln[bempp_space1.global_dof_count + mesh.num_vertices():]
   
    #Calcula todo el dominio de Fem y Bem a partir del borde calculado en el paso anterior 
    #(Incluido la funcion si se quiere calcular el valor de los tres dominios)

    #Calcula todo el dominio global del potencial a partir de la solucion de los borde de ambas interfaces.
    #(Incluido la funcion si se quiere calcular el valor de los tres dominios)

    # Calcula la solucion del potencial real en el dominio de FEM en la region intermedia 
    u = dolfin.Function(fenics_space)
    u.vector()[:] = np.ascontiguousarray(np.real(soln_fem)) 

    # Solucion para datos de Dirichlet en la interfaz inferior
    dirichlet_data1 = trace_matrix1 * soln_fem
    dirichlet_fun1 = bempp.api.GridFunction(trace_space1, coefficients=dirichlet_data1)
    # Solucion para datos de Neumann en la interfaz inferior
    neumann_fun1 = bempp.api.GridFunction(bempp_space1, coefficients=soln_bem1)

    # Solucion para datos de Dirichlet en la interfaz superior
    dirichlet_data2 = trace_matrix2 * soln_fem
    dirichlet_fun2 = bempp.api.GridFunction(trace_space2, coefficients=dirichlet_data2)
    # Solucion para datos de Neumann en la interfaz superior
    neumann_fun2 = bempp.api.GridFunction(bempp_space2, coefficients=soln_bem2)

    #Calculo del potencial en la posicion de los atomos de la molecula
    VF1 = bempp.api.operators.potential.laplace.single_layer(bempp_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(trace_space1, np.transpose(PC))
    uF = -VF1*neumann_fun1 + KF1*dirichlet_fun1 

    #Resultado de la energia de solvatacion total 
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real      
    print('Energia de Solvatacion en BEM/FEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )
    end = time.time()
    curr_time = (end - start)   
    print("Tiempo total: {:5.2f} [s]".format(curr_time))
    
    return E_Solv,curr_time, curr_time1, count_iterations.niter

