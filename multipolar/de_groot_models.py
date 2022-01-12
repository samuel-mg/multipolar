import numpy as np
import networkx as nx
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import time
import sys
import copy

def init_model(
    A,
    ellite,
    nodes_order,
    order_dict,
    remove_loops=True):
    """
    Created: 15-12-2015
    Modified: 11-01-2021
    """
    ## No self-influencing
    if remove_loops:
        A.tolil().setdiag(0)

    ellite_pos = []
    coords_init = np.zeros((len(nodes_order),len(list(ellite.values())[0])))
    for node in ellite:
        pos = order_dict[node]
        ellite_pos.append(pos)
        coords_init[pos] = ellite[node]
    # print "Ellite positions: ", ellite_pos
    not_ellite_pos = sorted( set(range(len(nodes_order))) - set(ellite_pos) )
    # print "Not ellite positions: ", not_ellite_pos
    Acsc = A.tocsc()
    Acsc_norm = sparse_row_normalize(Acsc)
    assert len(set(ellite_pos).intersection(set(not_ellite_pos))) == 0
    return Acsc_norm, ellite_pos, not_ellite_pos, coords_init

##############################################################################
##############################################################################
## Methods to solve model. Fastest is solve_model_Ax_iterative.
##############################################################################
##############################################################################

## Iterative method x_t+1=A x_t
def solve_model_Ax_iterative(
    G,
    ellite,
    tol=1e-8,
    nodes_order=None,
    order_dict=None,
    verbose=True,
    remove_loops=True
    ):
    """
    Solve the polarization model with the powers method.
    Created: 2020-02-27
    Modified: 2020-02-27
    THE FASTEST METHOD BY MANY ORDERS OF MAGNITUDE.
    """
    if nodes_order is None:
        nodes_order = G.nodes()
    if order_dict is None:
        order_dict = {v:i for i,v in enumerate(nodes_order)}

    print ("Building adjacency matrix...")
    A = nx.adjacency_matrix(G,nodelist=nodes_order)
    A = A.astype(np.float64)

    print ("Initializing the model...")
    A, ellite_pos, not_ellite_pos, coords_init = init_model(A,ellite,nodes_order,order_dict,remove_loops=remove_loops)
    A = A.tolil()
    
    ## Put a 1 in the diagonal of ellite nodes
    for posi in ellite_pos:
        row_ellite = np.zeros(A.shape[1])
        row_ellite[posi] = 1.0
        A[posi,:] = row_ellite
    A = A.tocsr()
        
    x = copy.deepcopy(coords_init)
    res = np.inf
    print ("Solving polarization model with the iterative Ax method")
    t0 = time.time()
    cntr = 0
    while res > tol:
        if verbose:
            print (cntr, res)
        x_new = A.dot(x)
        res = np.sum(np.abs(x_new-x))
        x = copy.deepcopy(x_new)
        cntr += 1
    print ("Solved. Time = ", time.time() - t0)
    return x

## LU method
def solve_model(G,ellite,nodes_order=None, order_dict =None,remove_loops=True):
    """
    Created: 14-12-2015
    Modified: 26-02-2020
    26-02-2020 - Modified to add some prints to keep track in 
    long computations. 
    """
    if type(G) == nx.classes.graph.Graph:
        assert nx.is_connected(G) ## This is not pertinent when working with ellite connected directed networks.
    if nodes_order is None:
        nodes_order = G.nodes()
    if order_dict is None:
        order_dict = dict([ (v,i) for i,v in enumerate(nodes_order) ])

    print ("Obtaining the adjacency matrix...")
    A = nx.adjacency_matrix(G,nodelist=nodes_order)
    A = A.astype(np.float64)

    print ("Initializing DeGroot's model...")
    A, ellite_pos, not_ellite_pos, coords_init = init_model(A,ellite,nodes_order,order_dict,remove_loops=remove_loops)
    
    print ("Building the linear system of equations...") ## Quizas esto se pueda acelerar pasando A a lil

    Afin = A - sprs.eye(A.shape[0])
    #print "Afin",Afin.todense()
    ## print Afin.todense()
    b = -Afin.dot(coords_init)

    Afin = Afin.tolil() ## NEW

    #print "b",b
    Aslice = Afin[not_ellite_pos,:][:,not_ellite_pos]
    #print "Aslice", Aslice.todense()
    bslice = b[not_ellite_pos]
    Aslice = Aslice.tocsc() ## NEW
    bslice = bslice.tocsc() ## NEW
    #print "bslice",bslice
    print ("Computing LU decomposition")
    t0 = time.time()
    LU = splu(Aslice)
    print ("Condition number of the matrix: (%0.03e/%0.03e)"%(np.linalg.cond(Aslice.todense()),1.0/sys.float_info.epsilon))
    print ("Solving with LU")
    final_coords = LU.solve(bslice)
    print ("Solved, t= ", time.time()-t0)
    coord_fin = np.zeros((len(nodes_order),len(list(ellite.values())[0])))
    coord_fin[ellite_pos,:] = coords_init[ellite_pos,:]
    coord_fin[not_ellite_pos,:] = final_coords.reshape(coord_fin[not_ellite_pos,:].shape)
    return coord_fin

## Powers method
def solve_model_powers(G,ellite,tol=sys.float_info.epsilon,nodes_order=None,order_dict=None,remove_loops=True):
    """
    Solve the polarization model with the powers method.
    Created: 2018-11-3
    Modified: 2020-02-27
    2020-02-27 - To include some minor prints and optimizations
    """
    if nodes_order is None:
        nodes_order = G.nodes()
    if order_dict is None:
        order_dict = {v:i for i,v in enumerate(nodes_order)}

    print ("Building adjacency matrix...")
    A = nx.adjacency_matrix(G,nodelist=nodes_order)
    A = A.astype(np.float64)

    print ("Initializing the model...")
    A, ellite_pos, not_ellite_pos, coords_init = init_model(A,ellite,nodes_order,order_dict,remove_loops=remove_loops)
    A = A.tolil()
    
    ## Put a 1 in the diagonal of ellite nodes
    for posi in ellite_pos:
        row_ellite = np.zeros(A.shape[1])
        row_ellite[posi] = 1.0
        A[posi,:] = row_ellite
    A = A.tocsr()
        
    A_pow = copy.deepcopy(A)
    x = copy.deepcopy(coords_init)
    res = np.inf
    print ("Solving polarization model with the matrix power method")
    t0 = time.time()
    while res > tol:
        print (res)
        A_pow = A_pow.dot(A_pow)
        x_new = A_pow.dot(x)
        res = np.max(np.abs(x_new-x))
        x = copy.deepcopy(x_new)
    print ("Solved. Time = ", time.time() - t0)
    return x

## Asynchronous method
def de_groot_async(G,ellite, n_iter = 100, tol = sys.float_info.epsilon, v = False, nodes_order=None, get_vec=False):
    assert type(ellite) == dict
    nodes_list = list(G.nodes)
    listeners = list( set(nodes_list) - set(ellite.keys()) )
    coords_dct = {n: np.zeros(len(list(ellite.values())[0])) for n in nodes_list}
    for n, crds in ellite.items():
        coords_dct[n] = crds
    lstnr_rnd = copy.deepcopy(listeners)
    diff_crds = np.inf
    for itr in range(n_iter):
        if v:
            print (itr, diff_crds, tol)
        np.random.shuffle(lstnr_rnd)
        coords_old = np.array([coords_dct[n] for n in nodes_list])
        for n in lstnr_rnd:
            succs = G[n]
            succs_cords = np.zeros((len(succs),len(list(ellite.values())[0])))
            for i, si in enumerate(succs):
                succs_cords[i,:] = coords_dct[si]
            new_crd = np.mean(succs_cords, axis=0)
            ## assert len(new_crd) == len(coords_dct[si]) ## for tests
            coords_dct[n] = new_crd
        coords_new = np.array([coords_dct[n] for n in nodes_list])
        diff_crds = np.sum(np.abs(coords_new-coords_old))
        if diff_crds < tol:
            break
    if itr == n_iter-1:
        print ("WARNING!!! Max iter achieved with no convergence")
    if get_vec:
        x = []
        for n in nodes_order:
            x.append(coords_dct[n])
        return np.array(x)
    else:
        return coords_dct

##############################################################################
##############################################################################
## Tools
##############################################################################
##############################################################################

## To build regular simplices of dim n
def get_simplex_vertex(n):
    """
    Created: 4-12-2015
    Modified: 4-12-2015
    Gets the coordinates of the hipertetrahedron of dimension n or n-simplex.
    Note: it is not throughly tested.
    """
    x = np.zeros((n+1,n))
    for i in range(n):
        x[i,i] = np.sqrt(1.0-np.dot(x[i,:],x[i,:]))
        for j in range(i+1,n+1):
            # print x
            x[j,i] = (-1.0/n - np.dot(x[j,:],x[i,:]))/x[i,i]
    return x

def sparse_row_normalize(sps_mat):
    """
    Inpired in:
    http://stackoverflow.com/questions/15196289/modify-scipy-sparse-matrix-in-place
    """
    if sps_mat.format != 'csc':
        msg = 'Can only row-normalize in place with csc format, not {0}.'
        msg = msg.format(sps_mat.format)
        raise ValueError(msg)
    row_norm = (np.bincount(sps_mat.indices, weights=sps_mat.data))
    sps_mat.data = sps_mat.data / np.take(row_norm, sps_mat.indices)
    return sps_mat

def assign_poles_to_nodes(
    poles_corresp,
    poles_nodes_dct):
    diff_poles_corresp = set(poles_corresp.keys())
    diff_poles_in_nodes_dct = set(poles_nodes_dct.keys())
    if diff_poles_corresp != diff_poles_in_nodes_dct:
        print (f"Warning!! Different poles in poles-index dict and in\
                poles-nodes dict:\npoles_corresp{diff_poles_corresp}\
                \npoles_nodes_dct{diff_poles_in_nodes_dct}")
        if diff_poles_corresp-diff_poles_in_nodes_dct!=set():
            raise Exception("There are poles in poles_corresp that are not\
                            in poles_nodes_dct. This would cause an excessive number of\
                            dimensions in the polarization computation.")

    ## Make sure numerical labels are consistent
    assert set(poles_corresp.values())==set(range(len(poles_corresp)))

    ## Make sure elite sets are disjoint
    for pole_a, elite_set_a in poles_nodes_dct.items():
        for pole_b, elite_set_b in poles_nodes_dct.items():
            if len(elite_set_b) != len(set(elite_set_b)):
                raise Exception(f"Elite of pole {pole_b} has repeated nodes")
            elite_set_a = set(elite_set_a)
            elite_set_b = set(elite_set_b)
            if set(elite_set_a) != set(elite_set_b):
                if elite_set_a.intersection(elite_set_b):
                    raise Exception(f"Elite sets of poles {pole_a} and {pole_b} share nodes {elite_set_a.intersection(elite_set_b)}")

    poles_coords = get_simplex_vertex(len(poles_corresp)-1)
    elite_nodes = []
    elite_dct = {}
    for pole, nodes in poles_nodes_dct.items():
        # print (pole, len(nodes))
        idx = poles_corresp[pole]
        coords = poles_coords[idx]
        elite_nodes.extend(nodes)
        for n in nodes:
            elite_dct[n] = coords
    # print (len(elite_dct))
    pole_coords_dct = {k:poles_coords[v] for k,v in poles_corresp.items()}
    return elite_dct, elite_nodes, pole_coords_dct

def get_elite_connected(G, elite_nodes,max_path_len=10):
    """
    Created: 2020-03-04
    Modified: 2020-03-04
    A function to obtain all the nodes with at least one directed path to an 
    elite node.
    """
    nodes_list = list(G.nodes())
    elite_index = []

    for i,n in enumerate(nodes_list):
        if n in elite_nodes:
            elite_index.append(i)
        if len(elite_index) == len(elite_nodes):
            break

    print ("calculando las conexiones...")
    A = nx.to_scipy_sparse_matrix(G, nodelist = nodes_list)
    elite_con = A[:,elite_index]
    nonzero_row_indice, _ = elite_con.nonzero()
    conectados_total = set(nonzero_row_indice)
    for i in range(max_path_len):
        print (i,len(conectados_total))
        elite_con = A.dot(elite_con)
        nonzero_row_indice, _ = elite_con.nonzero()
        conectados_total.update(nonzero_row_indice)
    del(A)
    
    conectados_total.update(elite_index)

    conectados_total_id = []
    for i in conectados_total:
        conectados_total_id.append(nodes_list[i])

    return nx.DiGraph(G.subgraph(conectados_total_id))