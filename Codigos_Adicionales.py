import parmed.amber
import numpy as np 
from pathlib import Path
import sys
sys.path.append('/home/mauricioguerrero/Software/bem_electrostatics-master/')
import bem_electrostatics
      
def MallaSuperficial(file,de,f,FE,FS):
    # file: nombre del archivo en formato de texto
    # FE: direccion donde se encuentra el pqr. Ej: 'Moleculas_Grandes/PQR/'
    # FS: direccion donde se guarda la malla.  Ej: 'Moleculas_Grandes/Mallas_S/'
    #Eleccion de creacion de malla
    if f=='n':
        fe= "nanoshaper" 
    elif f=='m':
        fe= "msms"    
    protein = bem_electrostatics.solute(
                solute_file_path="/home/mauricioguerrero/"+FE+file+".pqr", #pqr que se va a mallla
                save_mesh_build_files = True,  
                mesh_build_files_dir = "/home/mauricioguerrero/"+FS, #carpeta para guardar la malla
                mesh_density = de, #densidad(msms) o longitud(nanoshaper)
                mesh_probe_radius = 1.4, #radio de esfera de prueba
                mesh_generator = fe, #metodo para crear la malla
                print_times = True)
    V = protein.mesh.number_of_vertices  #Vertices
    E = protein.mesh.number_of_elements  #Elementos
    return V,E

# Caso para las moleculas de Mobley
def PQR_Moleculas_Pequeñas(file,d,FP,FC,FS):  
    # file: nombre del archivo en formato de texto
    #FP: direccion donde se encuentra la carpeta prmcrd. Ej: 'Moleculas_Pequeñas/prmcrd/'
    #FC: direccion donde se encuentra la carpeta charged_mol2files. Ej: 'Moleculas_Pequeñas/charged_mol2files/'
    #FS: direccion donde se guardara el pqr. Ej: 'Moleculas_Pequeñas/PQR/'
    #d: distacia adicional para el radio para crear la malla con capa de exclusion
    
    #Extraccion de la informacion del radio del '.prmtop'
    mol_param = parmed.amber.AmberParm(FP+ file + '.prmtop')

    N_atom = mol_param.ptr('NATOM')
    atom_type = mol_param.parm_data['ATOM_TYPE_INDEX']
    atom_radius = np.zeros(N_atom)
    atom_depth = np.zeros(N_atom)

    for i in range(N_atom):
        atom_radius[i] = mol_param.LJ_radius[atom_type[i]-1]
        atom_depth[i] = mol_param.LJ_depth[atom_type[i]-1]

    #Extraccion de la informacion de las cargas del '.mol2'
    atom_charges = []
    with open(FC + file +".mol2","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line)==9:
            atom_charges.append(float(line[8]))
        
    #Extraccion de la informacion de la posicion de los atomos '.prmtop'
    atom_posX = []
    atom_posY = []
    atom_posZ = []
    with open(FP+ file +".crd","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line)==6:        
            atom_posX.append(float(line[0]))
            atom_posY.append(float(line[1]))
            atom_posZ.append(float(line[2]))
            atom_posX.append(float(line[3]))
            atom_posY.append(float(line[4]))
            atom_posZ.append(float(line[5]))
        elif len(line)==3:        
            atom_posX.append(float(line[0]))
            atom_posY.append(float(line[1]))
            atom_posZ.append(float(line[2]))
         
    #Creacion del PQR
    if d==0:
        f0=".pqr"
    else:
        f0="R"+str(d)+".pqr"
    
    FF=open(FS+file +f0,"w")
    for i in range(len(atom_radius)):   
        x="{:.3f}".format(atom_posX[i])
        y="{:.3f}".format(atom_posY[i])
        z="{:.3f}".format(atom_posZ[i])
        r="{:.3f}".format(atom_radius[i]+d)
        q="{:.4f}".format(atom_charges[i])
        if len(x)==5:
            x=" "+x
        if len(y)==5:
            y=" "+y
        if len(z)==5:
            z=" "+z
        if len(q)==6:
            q=" "+q
        if len(r)==5:
            r=" "+r
        FF.write("ATOM      1  C1  TMP  1      "+x+"  "+y+"  "+z+"  "+q+" "+r+"\n") #Para leer correctamente la informacion, tiene que tener esta configuracion de escritura.
    FF.close()
    return d

#Modificacion del texto del pqr del 1LYZ y el 1BPI extraidos del PDB2PQR, para que se pueda leer correctamente la informacion
def NuevoPQR(file,d,FE,FS):      
    # file: nombre del archivo en formato de texto
    # FE: direccion donde se encuentra el pqr. Ej: 'Moleculas_Grandes/PQR/'
    # FS: direccion donde se guarda el pqr modificado.  Ej: 'Moleculas_Grandes/Mallas_S/'  
    # d: distacia adicional para el radio para crear la malla con capa de exclusion        
    Tex = []
    posX = []
    posY = []
    posZ = []
    posQ = []
    posR = []            
    with open(FE+file+"Base.pqr","r") as f:
        lines = f.readlines()
        Tex.append(lines)
    for line in lines:
        line = line.split()
        if len(line)==10:    
            posX.append(line[5])
            posY.append(line[6])
            posZ.append(line[7])
            posQ.append(line[8])
            posR.append(line[9]) 
            
    #Creacion del PQR
    if d==0:
        f0=".pqr"
    else:
        f0="R"+str(d)+".pqr"
    
    FF=open(FS+file +f0,"w")
    for i in range(len(posX)):   
        t=Tex[0][i][0:28]  
        x=str(posX[i])
        y=str(posY[i])
        z=str(posZ[i])
        q="{:.4f}".format(float(posQ[i]))
        r="{:.4f}".format(float(posR[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        FF.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")  
    FF.close()
    return d


# Caso para las moleculas de grandes como el 1SE0 y 1BBZ
def PQR_Moleculas_de_Union(file,d,FP,FE,FS):  
# file: nombre del archivo en formato de texto
# FE: direccion donde se encuentra el archivo prmtop y el pdb. Ej: 'Moleculas_de_Union/top/'
# FS: direccion donde se guarda el pqr modificado.  Ej: 'Moleculas_de_Union/PQR/'  
# d: distacia adicional para el radio para crear la malla con capa de exclusion  
    if file=='1SE0-L':
        N1='PROA'
        N2='PROB'
        fileL='L1SE0'
        file1='1SE0'
        k=3
        k1=1
        k2=0
    elif file=='1BBZ-L':
        N1='SH3D'
        N2='PPRO'   
        fileL='L1BBZ'
        file1='1BBZ'
        k=0
        k1=0
        k2=1
    
    #Caso, datos de la proteina y proteina+ligando
    C=[]
    Tex = []
    posX = []
    posY = []
    posZ = []
    posQ = []      
    #Posicion, proteina y ligando
    with open(FE+file+".pdb","r") as f:
        lines = f.readlines() 
        Tex.append(lines)
    for line in lines:
        line = line.split()
        if line[0]== 'ATOM':
            if line[11] == N1: 
                G=0
                posX.append(line[6])
                posY.append(line[7])
                posZ.append(line[8])
            elif line[11] == N2:
                posX.append(line[6])
                posY.append(line[7])
                posZ.append(line[8])    
                if G==0:
                    C.append(int(line[1])-1)
                    G=1
            else:       
                C.append(int(line[1])-1)
                break
    
    #Radio y Carga, proteina y ligando       
    mol_param = parmed.amber.AmberParm(FE+ file + '.prmtop')
    N_atom = C[1] 
    atom_type = mol_param.parm_data['ATOM_TYPE_INDEX']
    atom_charge = mol_param.parm_data['CHARGE']
    posR = np.zeros(N_atom)
    for i in range(N_atom):
        posR[i] = mol_param.LJ_radius[atom_type[i]-1]          
        posQ.append(atom_charge[i])
 
    #Ligando
    CL=[]
    TexL = []
    posXL = []
    posYL = []
    posZL = []
    posQL = []      
    #Posicion del ligando
    with open(FE+fileL+".pdb","r") as f:
        lines = f.readlines() 
        TexL.append(lines)
    for line in lines:
        line = line.split()
        if line[0]== 'ATOM':
            if line[11-k1] == N2: 
                posXL.append(line[6-k1])
                posYL.append(line[7-k1])
                posZL.append(line[8-k1])
            else:       
                CL.append(int(line[1])-1)
                break
    
    #Radio y Carga del ligando    
    mol_paramL = parmed.amber.AmberParm(FE+ fileL + '.prmtop')
    N_atomL = CL[0] 
    atom_typeL = mol_paramL.parm_data['ATOM_TYPE_INDEX']
    atom_chargeL = mol_paramL.parm_data['CHARGE']
    posRL = np.zeros(N_atomL)
    for i in range(N_atomL):
        posRL[i] = mol_paramL.LJ_radius[atom_typeL[i]-1]          
        posQL.append(atom_chargeL[i])

    #Creacion del PQR
    if d==0:
        f0=".pqr"
    else:
        f0="R"+str(d)+".pqr"
    
    FF0=open(FS+file+f0,"w")  #Conjunto
    FF1=open(FS+file1+f0,"w")  #Proteina
    FF2=open(FS+fileL+f0,"w")  #Ligando
    for i in range(len(posX)):  
        t=Tex[0][i+k][0:20]+" "+Tex[0][i+k][22:28]   
        x=str(posX[i])
        y=str(posY[i])
        z=str(posZ[i])
        q="{:.4f}".format(posQ[i])
        r="{:.4f}".format(float(posR[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        if i<C[0]:  
            FF1.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")      
        FF0.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")  
    FF0.close()
    FF1.close()
    
    for i in range(len(posXL)):  
        t=TexL[0][i+k2][0:20]+" "+TexL[0][i+k2][22:28]   
        x=str(posXL[i])
        y=str(posYL[i])
        z=str(posZL[i])
        q="{:.4f}".format(posQL[i])
        r="{:.4f}".format(float(posRL[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        FF2.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")
    FF2.close()
    
    return d