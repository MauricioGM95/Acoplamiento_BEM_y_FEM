import numpy as np
def readpqr(filename):
    """
    Read charge information from pqr file
    Arguments
    ----------
    filename: name of the file that contains the surface information.
               (filename.pqr)
    REAL    : data type.
    Returns
    -------
    pos     : (Nqx3) array, positions of the charges.
    q       : (Nqx1) array, value of the charges.
    """
    REAL =np.float
    with open(filename, 'r') as f:
        lines = f.readlines()

    pos = []
    q = []
    r = []
    for line in lines:
        line = line.split()

        if line[0] == 'ATOM':
            #  grab coordinates and charge from columns
            x, y, z, q0 = [REAL(i) for i in line[5:-1]]
            r0 = REAL(line[-1])
            q.append(q0)
            r.append(r0)
            pos.append([x, y, z])

    pos = np.array(pos)
    q = np.array(q)
    r = np.array(r)
    return pos, q, r