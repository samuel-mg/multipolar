import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
plt.style.use("default")
import networkx as nx
import copy
import numpy as np
import sympy
from matplotlib import transforms
import matplotlib.colors as colors
# from sklearn.neighbors import KernelDensity
import itertools
from scipy import stats
import scipy as sp
import sys

from multiprocessing import Pool
import pickle

##############################################################################
## General geometry functions
##############################################################################

def project_point_to_line(p1,p2,x):
	"""
	Projects point x onto the straight line that passes through the origin and is parallel
	to the one that joines p1 and p2.
	p1: Pole 1.
	p2: Pole 2.
	x: Position of user.
	"""
	n = (p1-p2) / np.linalg.norm(p1-p2)
	return np.dot(n,x)
def project_point_to_line_v2(a,b,p):
	"""
	Modified: 2020-02-05
	Projects point x onto the straight line that passes through the origin and is parallel
	to the one that joines p1 and p2.
	A: Pole 1.
	B: Pole 2.
	P: Position of user.
	inspired in https://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line
	"""
	a = np.array(a,dtype=np.float64)
	b = np.array(b,dtype=np.float64)
	p = np.array(p,dtype=np.float64)
	ap = p-a
	ab = b-a
	result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
	return result
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
def get_lin_manifold_impl_eq(point_lst):
	"""
	Created: 2019-09-04
	Modified: 2019-09-04
	Get the implicit equations of a linear variety defined by a set of points.
	The vectors joining the n points are assumed to span a n-1 dimensional linear manifold. 
	"""
	## Build cartesian variables vector
	x = [sympy.symbols("x%d"%i) for i in range(len(point_lst[0]))]
	x.append(1)
	x = sympy.Matrix(x)
	## Build parameters vector
	a = [sympy.symbols("a%d"%i) for i in range(len(point_lst[0]))]
	a.append(sympy.symbols("k"))
	a = sympy.Matrix(a)
	## Solve the system of equations of the form k+a1*p1+a2*p2+...
	## where ai are the parameters of the constraint and vectors 
	## {p1,p2,...,pn} are the coordinates of the points
	eqns = []
	for pi in point_lst:
		pi = sympy.Matrix(list(pi)+[1])
		eqns.append(pi.T*a)
	res = sympy.solve(eqns,a)
	## Substitute the computed values of the parameters on the 
	## constraint equation
	manifold_eq = a.T*x
	for ai in a:
		try:
			manifold_eq = manifold_eq.subs(ai,res[ai])
		except KeyError:
			pass
	#print manifold_eq
	## Convert the constraint equation of the manifold to the usual
	## implicit form.
	eq_fin = []
	for i,ai in enumerate(a):
		a_slc = a[:i]+a[(i+1):]
		eq_fin_i = copy.deepcopy(manifold_eq)
		eq_fin_i = eq_fin_i.subs(ai,1)
		for aj in a_slc: 
			eq_fin_i = eq_fin_i.subs(aj,0)
		eq_fin.append(eq_fin_i)
	eq_fin = [i[0] for i in eq_fin]
	#print eq_fin
	## Assess that the resulting set of equations have the right
	## rank
	M, b = sympy.linear_eq_to_matrix(eq_fin,[i for i in x [:-1]])
	if M.rank() != (len(point_lst[0])-len(point_lst)+1):
		print ((point_lst), eq_fin)
		print ("Rank %d. Expected %d.\nUser point falls inside linear manifold spanned by selected poles"%(M.rank(),(len(point_lst[0])-len(point_lst)+1)))
		return None
	return eq_fin
def get_intersect_point(lm1,lm2):
	"""
	Created: 2019-09-04
	Modified: 2019-09-04
	Returns a point of intersection between two linear manifolds (lm) of complementary
	dimensions.
	"""
	dim = len(lm1)-1
	intersect = lm1 + lm2
	x = [sympy.symbols("x%d"%i) for i in range(dim)]
	M, _ = sympy.linear_eq_to_matrix(intersect,x)
	assert M.rank() == dim
	res = sympy.solve(intersect,x)
	return np.array([res[xi] for xi in x],dtype=np.float64)
def build_param_lm_from_points(point_lst):
	"""
	Created: 2019-09-04
	Modified: 2019-09-04
	Build a linear manifold in parametric form from a set of points.
	"""
	p0 = sympy.Matrix(point_lst[0])
	V = []
	for pi in point_lst[1:]:
		pi = sympy.Matrix(pi)
		V.append(pi-p0)
	eq = copy.deepcopy(p0)
	for i, vi in enumerate(V):
		eq += sympy.symbols("l%d"%i)*vi
	return eq
def proyect_nd_to_md(
	pnt_lst,
	proy_from,
	proy_to,
	v = True
	):
	"""
	Created: 2019-09-16
	Modified: 2019-09-16
	Proyect the n-dimensional points contained in pnt_lst over the subspace
	spanned by points proy_to by performing the cuts of that linear manifold
	with the linear manifold spanned by each point and the points of proy_from
	"""
	assert len(proy_from)+len(proy_to) == len(pnt_lst[0])+1
	lm_to = get_lin_manifold_impl_eq(proy_to)
	proy_from = list(proy_from)
	proys_lst = []
	if v:
		print ("Projecting...")
	for i, pnt in enumerate(pnt_lst):
		if i % 100 == 0:
			print (i, len(pnt_lst))
		lm_from = get_lin_manifold_impl_eq(proy_from+[pnt])
		if lm_from is None: ## Better code needed here. Prone to errors.
			continue
		intrsc = get_intersect_point(lm_from,lm_to)
		proys_lst.append(intrsc)
	return proys_lst

def are_points_coplanar(points,dim,tol=1e-7):
	"""
	Created: 2019-09-16
	Modified: 2019-09-26
	Checks if the dimension of the space spanned by the points is exactly dim.
	Used to check if 4 points in 3D are coplanar, if 5 are co-hyperplanar
	in 4D or if 3 points are colinear in 2D.
	"""
	p0 = points[0]
	vects = [pi-p0 for pi in points[1:]]
	rnk = np.linalg.matrix_rank(vects,tol=tol)
	if rnk == dim:
		return True
	else:
		return False

## Numerical (instead of symbolical) versions that are faster:
def get_lin_manifold_num(
	point_lst,
	tol_diff = 1e-10,
	tol_det = 1e-10,
	tol_rnk = 1e-10
						):
	"""
	Created: 24-01-2020
	Modified: 24-01-2020
	Returns the coefficients of the equations of the subspace spanned
	by the n points living in D-dimensional space of point_lst; 
	that, is, the q equations (with q=D-n+1) of the form
	a1 x1 + a2 x2 + ... + aD xD - b = 0 where we are looking for the
	ai (and b) and are given the coordinates x1, x2, ..., xD of the points.
	Inspired in: 
	https://math.stackexchange.com/questions/2244129/solution-of-a-linear-underdetermined-system-of-equations
	There is another method proposed here:
	https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
	"""
	point_lst = np.array(point_lst)
	A = np.array([list(p)+[-1] for p in point_lst])
	#### debug
	#print (A)
	####
	D = len(point_lst[0])
	n = len(point_lst)
	q = D-n+1
	ok = 0
	eqs = []
	## Make sure that the points are different
	for i,pi in enumerate(point_lst):
		for pj in point_lst[i+1:]: 
			pipj = pi-pj
			#assert np.sqrt(np.dot(pipj,pipj)) > tol_diff
			if np.sqrt(np.dot(pipj,pipj)) < tol_diff:
				#print ("Points too close")
				#print (point_lst)
				return None
	for cols_A1 in itertools.combinations(range(D+1), n):
		cols_A1 = list(cols_A1)
		## In a subfunction
		A1 = copy.deepcopy(A[:,cols_A1])
		if np.abs(np.linalg.det(A1)) < tol_det:
			#### debug
			#print (A1)
			####
			continue
		cols_A2 = [i for i in range(D+1) if i not in cols_A1]
		A2 = copy.deepcopy(A[:,cols_A2])
		x2 = np.ones(len(cols_A2))
		b3 = -np.dot(A2,x2)
		##
		x1 = np.linalg.solve(A1,b3)
		## Build complete solution
		x_full = np.zeros(D+1) + np.nan
		x_full[cols_A1] = x1
		x_full[cols_A2] = x2
		eqs_temp = copy.deepcopy(eqs)
		eqs_temp.append(x_full)
		eqs_rnk = np.linalg.matrix_rank(eqs_temp,tol=tol_rnk)
		if eqs_rnk == len(eqs_temp):
			eqs = copy.deepcopy(eqs_temp)
		elif eqs_rnk > len(eqs_temp):
			print (point_lst)
			raise Exception("WTF happened here")
		if len(eqs) == q:
			ok = 1
			break
	if len(eqs) == q:
		ok = 1
	if not ok:
		print ("Points:", point_lst)
		print (q, eqs)
		raise Exception("WTF happenned here")
	return eqs
def get_lin_manifold_num_v2(
	point_lst,
	tol_diff = 1e-10,
						):
	"""
	Created: 04-02-2020
	Modified: 04-02-2020
	Returns the coefficients of the equations of the subspace spanned
	by the n points living in D-dimensional space of point_lst; 
	that, is, the q equations (with q=D-n+1) of the form
	a1 x1 + a2 x2 + ... + aD xD - b = 0 where we are looking for the
	ai (and b) and are given the coordinates x1, x2, ..., xD of the points.
	Computed searching for an orthonormal base that fulfills Ax = 0 
	where A is the point matrix and x the coefficient vector.
	"""
	point_lst = np.array(point_lst)
	## Make sure that the points are different
	for i,pi in enumerate(point_lst):
		for pj in point_lst[i+1:]:
			pipj = pi-pj
			#assert np.sqrt(np.dot(pipj,pipj)) > tol_diff
			if np.sqrt(np.dot(pipj,pipj)) < tol_diff:
				#print ("Points too close")
				#print (point_lst)
				return "close"
	A = np.array([list(p)+[-1] for p in point_lst])
	## I had many problems with ill-conditioned matrices, so I check
	## if the condition number is so small that I should not even
	## try to compute anything.
	if np.linalg.cond(A) > 0.1/sys.float_info.epsilon: 
		return "illcond"
	eqs = sp.linalg.null_space(A).T
	return eqs
def get_intersect_point_num(lm1,lm2):
	"""
	Created: 24-01-2020
	Modified: 24-01-2020
	Get the intersection point of two linear manifolds defined
	by the coefficient matrices of their 
	"""
	Ab = np.concatenate((lm1,lm2),axis=0)
	A = Ab[:,:-1]
	b = Ab[:,-1]
	res = np.linalg.solve(A,b)
	return res
def proyect_nd_to_md_num(
	pnt_lst,
	proy_from,
	proy_to):
	"""
	Created: 2020-01-24
	Modified: 2020-01-24
	Proyect the n-dimensional points contained in pnt_lst over the subspace
	spanned by points proy_to by performing the cuts of that linear manifold
	with the linear manifold spanned by each point and the points of proy_from
	"""
	assert len(proy_from)+len(proy_to) == len(pnt_lst[0])+1
	lm_to = get_lin_manifold_num_v2(proy_to)
	if lm_to is None:
		raise Exception("Projection manifold ill defined (at least two points are the same)")
	proy_from = list(proy_from)
	proys_lst = []
	nocontr_cntr = 0
	illcond_cntr = 0
	for pnt in pnt_lst:
		full_frm = proy_from+[pnt]
		lm_from = get_lin_manifold_num_v2(full_frm)
		if type(lm_from) is str:
			if lm_from == "close":
				#print ("Point with no contribution of projection poles. Skipping.")
				nocontr_cntr += 1
				continue
			if lm_from == "illcond":
				#print ("Ill-conditioned matrix. Skipping.")
				illcond_cntr += 1
				continue
		intrsc = get_intersect_point_num(lm_from,lm_to)
		proys_lst.append(intrsc)
	print (f"No contribution: {nocontr_cntr} ({100.0*nocontr_cntr/len(pnt_lst):.1f}%)")
	print (f"Ill-conditioned: {illcond_cntr} ({100.0*illcond_cntr/len(pnt_lst):.1f}%)")
	return proys_lst

def proyect_nd_to_md_num_PAR(
	pnt_lst,
	proy_from,
	proy_to,
	get_bad_pnts=False):
	"""
	Created: 2020-01-24
	Modified: 2020-03-04
	Proyect the n-dimensional points contained in pnt_lst over the subspace
	spanned by points proy_to by performing the cuts of that linear manifold
	with the linear manifold spanned by each point and the points of proy_from
	PARALLEL VERSION.
	"""
	assert len(proy_from)+len(proy_to) == len(pnt_lst[0])+1
	lm_to = get_lin_manifold_num_v2(proy_to)
	if lm_to is None:
		raise Exception("Projection manifold ill defined (at least two points are the same)")
	proy_from = list(proy_from)
	pool = Pool(processes=20,maxtasksperchild=5)
	func_args = [(pnt, proy_from, lm_to) for pnt in pnt_lst]
	proys_lst = pool.map(proyect_nd_to_md_num_WORKER,func_args)
	pool.close()
	pool.join()
	nocontr_cntr = 0
	illcond_cntr = 0
	bad_pnts = []
	for i, elem in enumerate(proys_lst):
		if type(elem) is str:
			bad_pnts.append(i)
			if elem == "close":
				nocontr_cntr += 1
			elif elem == "illcond":
				illcond_cntr += 1
	proys_lst_final = [i for i in proys_lst if type(i) != str]
	print (f"No contribution: {nocontr_cntr} ({100.0*nocontr_cntr/len(pnt_lst):.1f}%)")
	print (f"Ill-conditioned: {illcond_cntr} ({100.0*illcond_cntr/len(pnt_lst):.1f}%)")
	if not get_bad_pnts:
		return proys_lst_final
	else:
		return proys_lst_final, bad_pnts

def proyect_nd_to_md_num_WORKER(
	args
	):
	"""
	Created: 2020-03-04
	Modified: 2020-03-04
	Inner loop function to parallelize proyect_nd_to_md.
	"""
	pnt, proy_from, lm_to = args
	full_frm = proy_from+[pnt]
	lm_from = get_lin_manifold_num_v2(full_frm)
	if type(lm_from) is str:
		if lm_from == "close":
			#print ("Point with no contribution of projection poles. Skipping.")
			return lm_from
		if lm_from == "illcond":
			#print ("Ill-conditioned matrix. Skipping.")
			return lm_from
	intrsc = get_intersect_point_num(lm_from,lm_to)
	return intrsc

def new_2d_coords(
	pnt_lst,
	plane_pnts
	):
	"""
	Created: 2019-09-16
	Modified: 2019-09-16
	Compute the coordinates of pnt_lst in a new reference system spanned by 
	the points in plane_pnts. The new coordinate system is built such that 
	the points with the new coords can be plot in the triangle of the 2D 
	visualization. That is, it is assumed that the plane_pnts form a triangle
	and the new coordinate system is centered at the baricenter of an equilateral
	triangle with a vertex on coordinates (x=1,y=0). That vertex corresponds
	to the first point of the list plane_pnts. The top left vertex is the
	second point of plane_pnts and the bottom left vertex is the third point.
	
	The points of pnt_lst must be coplanar to the other three points. That
	is, previously to use this, use proyect_nd_to_md to proyect the points
	to the plane spanned by the points in plane_pnts.
	"""
	assert len(plane_pnts) == 3
	## Compute baricenter of three points, which will be the origin
	bc = (plane_pnts[0]+plane_pnts[1]+plane_pnts[2])/3.0
	## Vectors of the new base
	v1 = plane_pnts[0]-bc
	v2 = (plane_pnts[1]-plane_pnts[2])*(np.linalg.norm(v1)/np.linalg.norm(plane_pnts[1]-plane_pnts[2]))
	new_coords = []
	for pnt in pnt_lst:
		assert are_points_coplanar([pnt]+plane_pnts, 2)
		b = bc-pnt
		M_proy = np.array([v1,v2,b])
		M_proy_mtrx = sympy.Matrix(M_proy.T)
		prms = sympy.Matrix([sympy.symbols("a0"),sympy.symbols("a1"),1.0])
		## Ecuaciones a resolver para sacar las nuevas coordenadas
		eqs = M_proy_mtrx*prms
		print ("OJO!! Me quedo con las dos primeras ecuaciones porque solo necesito 2\
				para encontrar las componentes a0 y a1, pero si no son linealmente independientes,\
				esto no sale bien. REVISAR.")
		new_coords_i = sympy.solve([eqs[0],eqs[1]],[sympy.symbols("a0"),sympy.symbols("a1")])
		new_coords_i = [new_coords_i[sympy.symbols("a0")],new_coords_i[sympy.symbols("a1")]]
		new_coords.append(new_coords_i)
	return new_coords
		
def new_3d_coords(
	pnt_lst,
	tetra_pnts
	):
	"""
	Created: 2019-09-16
	Modified: 2019-09-16
	Compute the coordinates of pnt_lst in a new reference system spanned by 
	the points in tetra_pnts. The new coordinate system is built such that 
	the points with the new coords can be plot in the tetrahedron of the 3D 
	visualization. That is, it is assumed that the tetra_pnts form a tetrahedron
	and the new coordinate system is centered at the baricenter of an equilateral
	tetrahedron with a vertex on coordinates (x=1,y=0,z=0). That vertex corresponds
	to the first point of the list tetra_pnts.
	
	The points of pnt_lst must be "coplanar" to the other three points. That
	is, previously to use this, use proyect_nd_to_md to proyect the points
	to the space spanned by the points in tetra_pnts.
	"""
	assert len(tetra_pnts) == 4
	lm_plane = get_lin_manifold_impl_eq(tetra_pnts)
	pnt_lst = list(pnt_lst)
	## Compute baricenter of three points, which will be the origin
	bc = np.zeros(size=len(tetra_pnts[0]))
	for ttr_pnt in tetra_pnts:
		bc += ttr_pnt
	bc = bc/4.0
	## Vectors of the new base
	v1 = tetra_pnts[0]-bc ## x-axis vector
	v2 = tetra_pnts[1]-(tetra_pnts[2]+tetra_pnts[3])/2.0 ## y axis vector
	v2 = v2*(np.linalg.norm(v1)/np.linalg.norm(v2)) ## Normalizing
	v3 = tetra_pnts[2]-tetra_pnts[3] ## z axis vector
	v3 = v3*(np.linalg.norm(v1)/np.linalg.norm(v3)) ## Normalizing
	new_coords = []
	for pnt in pnt_lst:
		assert are_points_coplanar([pnt]+tetra_pnts, 3)
		b = bc-pnt
		M_proy = np.array([v1,v2,v3,b])
		M_proy_mtrx = sympy.Matrix(M_proy.T)
		prms = sympy.Matrix([sympy.symbols("a0"),sympy.symbols("a1"),sympy.symbols("a2"),1.0])
		## Ecuaciones a resolver para sacar las nuevas coordenadas
		eqs = M_proy_mtrx*prms
		print ("OJO!! Me quedo con las tres primeras ecuaciones porque solo necesito 3\
				para encontrar las componentes a0, a1 y a2, pero si no son linealmente independientes,\
				esto no sale bien. REVISAR.") 
		new_coords_i = sympy.solve([eqs[0],eqs[1],eqs[2]],[sympy.symbols("a0"),sympy.symbols("a1"),sympy.symbols("a2")])
		new_coords_i = [new_coords_i[sympy.symbols("a0")],new_coords_i[sympy.symbols("a1")],new_coords_i[sympy.symbols("a2")]]
		new_coords.append(new_coords_i)
	return new_coords

def new_1d_coord(
	pnt_lst,
	line_pnts
	):
	"""
	Created: 2019-09-16
	Modified: 2019-09-16
	Converts the points in pnt_lst, which lie in the line spanned by the 
	two points, to a coordinate system where -1 corresponds to the first
	point of line_pnts and +1 to the second point.
	"""
	assert len(line_pnts) == 2
	new_coords = []
	p1 = line_pnts[0]
	p2 = line_pnts[1]
	D = np.linalg.norm(p2-p1)
	for pnt in pnt_lst:
		if not are_points_coplanar([pnt]+line_pnts, 1):
			print ([pnt]+line_pnts)
			input()
		d = np.linalg.norm(pnt-p1)
		proy_val = 2.0*(d/D)-1.0
		new_coords.append(proy_val)
	return new_coords

## Numerical versions of the 2d and 3d previous functions

def new_2d_coords_num(
	pnt_lst,
	plane_pnts,
	eps = 1e-9
	):
	"""
	Created: 2020-02-04
	Modified: 2020-02-04
	Faster version of new_2d_coords that does not use sympy.
	
	Compute the coordinates of pnt_lst in a new reference system spanned by 
	the points in plane_pnts. The new coordinate system is built such that 
	the points with the new coords can be plot in the triangle of the 2D 
	visualization. That is, it is assumed that the plane_pnts form a triangle
	and the new coordinate system is centered at the baricenter of an equilateral
	triangle with a vertex on coordinates (x=1,y=0). That vertex corresponds
	to the first point of the list plane_pnts. The top left vertex is the
	second point of plane_pnts and the bottom left vertex is the third point.
	
	The points of pnt_lst must be coplanar to the other three points. That
	is, previously to use this, use proyect_nd_to_md to project the points
	to the plane spanned by the points in plane_pnts.
	"""
	assert len(plane_pnts) == 3
	## Compute baricenter of three points, which will be the origin
	bc = (plane_pnts[0]+plane_pnts[1]+plane_pnts[2])/3.0
	## Vectors of the new base
	## Ojo! Una vez pense que habia que normalizarlos. No hay que hacerlo porque
	## las caras triangulares de los simplex de dimension superior a 2 son mas
	## pequenios, pero yo quiero poder dibujar mis proyecciones en el simplex
	## estander de 2d, entonces el tamanio de los vectores v1 y v2 TIENE QUE 
	## SER MAS PEQUENIO, adaptandose a cada dimensions. De esta manera, las
	## nuevas coordenadas 2d (a1 y a2) toman valores hasta 1 en vez de hasta
	## 0.9 y pico. De hecho, el v2, como se ve, hay que reescalarlo
	## tomando el tamanio que toma v1 de referencia, porque tamanio v1 es correcto.
	v1 = plane_pnts[0]-bc
	v2 = (plane_pnts[1]-plane_pnts[2])*(np.linalg.norm(v1)/np.linalg.norm(plane_pnts[1]-plane_pnts[2]))
	new_coords = []
	for pnt in pnt_lst:
		assert are_points_coplanar([pnt]+plane_pnts, 2)
		b = pnt-bc
		M = np.array([v1,v2]).T
		## I fit a least squares because the system is overdetermined
		res = np.linalg.lstsq(M,b) 
		new_coords_i = res[0]
		err = res[1]
		assert err < eps ## The fit should be almost perfect
		new_coords.append(new_coords_i)
	return np.array(new_coords)

def new_3d_coords_num(
	pnt_lst,
	tetra_pnts,
	eps = 1e-9
	):
	"""
	Created: 2020-02-04
	Modified: 2020-02-04

	Faster version of new_2d_coords that does not use sympy.

	Compute the coordinates of pnt_lst in a new reference system spanned by 
	the points in tetra_pnts. The new coordinate system is built such that 
	the points with the new coords can be plot in the tetrahedron of the 3D 
	visualization. That is, it is assumed that the tetra_pnts form a tetrahedron
	and the new coordinate system is centered at the baricenter of an equilateral
	tetrahedron with a vertex on coordinates (x=1,y=0,z=0). That vertex corresponds
	to the first point of the list tetra_pnts.
	
	The points of pnt_lst must be "coplanar" to the other three points. That
	is, previously to use this, use proyect_nd_to_md to proyect the points
	to the space spanned by the points in tetra_pnts.
	"""
	print ("THIS FUNCTION HAS NOT BEEN TESTED. USE WITH CARE")
	input()
	assert len(tetra_pnts) == 4
	## Compute baricenter of three points, which will be the origin
	bc = np.zeros(size=len(tetra_pnts[0]))
	for ttr_pnt in tetra_pnts:
		bc += ttr_pnt
	bc = bc/4.0
	## Vectors of the new base
	v1 = tetra_pnts[0]-bc ## x-axis vector
	v2 = tetra_pnts[1]-(tetra_pnts[2]+tetra_pnts[3])/2.0 ## y axis vector
	v2 = v2*(np.linalg.norm(v1)/np.linalg.norm(v2)) ## Normalizing
	v3 = tetra_pnts[2]-tetra_pnts[3] ## z axis vector
	v3 = v3*(np.linalg.norm(v1)/np.linalg.norm(v3)) ## Normalizing
	new_coords = []
	for pnt in pnt_lst:
		assert are_points_coplanar([pnt]+tetra_pnts, 3)
		b = pnt-bc
		M = np.array([v1,v2,v3]).T
		## I fit a least squares because the system is overdetermined
		res = np.linalg.lstsq(M,b) 
		new_coords_i = res[0]
		err = res[1]
		assert err < eps ## The fit should be almost perfect
		new_coords.append(new_coords_i)
	return np.array(new_coords)

######

def axis_1d_projection(
	pnt_lst,
	pole_set_A,
	pole_set_B
	):
	"""
	Created: 2020-02-05
	Modified: 2020-02-05
	Projection along the axis that joins the center of two groups of nodes.
	Useful to see left-right axis, new-old parties, 1 vs rest, etc.
	"""
	pole_set_A = np.array(pole_set_A)
	pole_set_B = np.array(pole_set_B)
	assert len(pole_set_A.shape)==2 ## Make sure it is a 2d array to not fuck the np.mean
	assert len(pole_set_B.shape)==2 ## Make sure it is a 2d array to not fuck the np.mean
	## Compute baricenters
	bcA = np.mean(pole_set_A,axis=0)
	bcB = np.mean(pole_set_B,axis=0)
	## Project everything in the line that joins A with B
	prjs = []
	for pnt in pnt_lst:
		prjs_i = project_point_to_line_v2(bcA,bcB,pnt)
		prjs.append(prjs_i)
	prjs_new = new_1d_coord(
		prjs,
		[bcA,bcB]
		)
	return prjs_new

###### Orthogonal projection functions

def proj_ortog(
	data,
	new_base,
	orig=None
	):
	"""
	Created: 2020-04-10
	Modified: 2020-04-10
	Returns the orthogonal projection of the points in data to the vectors in
	new_base.
	"""
	data = np.array(data)
	new_base = np.array(new_base)
	assert data.shape[1] == new_base.shape[1]
	assert data.shape[1] >= new_base.shape[0]
	if orig is None:
		orig = np.zeros(data.shape[1])
	n_points = data.shape[0]
	new_dim = new_base.shape[0]
	projections = np.zeros((n_points, new_dim))
	for i, pnt in enumerate(data):
		pri = np.dot(new_base, pnt-orig)
		projections[i,:] = pri
	return projections

def axis_1d_proj_ortog(
	data,
	pole_set_A,
	pole_set_B,
	):
	"""
	Created: 2020-04-10
	Modified: 2020-04-10
	Created from their non-orthogonal analogs.
	"""
	pole_set_A = np.array(pole_set_A)
	pole_set_B = np.array(pole_set_B)
	assert len(pole_set_A.shape)==2 ## Make sure it is a 2d array to not fuck the np.mean
	assert len(pole_set_B.shape)==2 ## Make sure it is a 2d array to not fuck the np.mean
	bcA = np.mean(pole_set_A,axis=0)
	bcB = np.mean(pole_set_B,axis=0)
	orig = (bcA+bcB)/2.0
	vec_dir = (bcB - orig)
	## Now, I normalize (divide by the norm) AND divide again by the norm so 
	## the projections are not the "actual" projections but a reescaled ones
	## that lie between the origin and the "tip" of the vec_dir vector. The
	## result is that I divide by vec_dir*vec_dir=norm(vec_dir)**2.0.
	## Because of that, it is necessary to have, not only the right direction
	## of the vector, but also THE RIGHT NORM before this "normalization"
	## (because I'm not building a unit vector)	
	## Otherwise, I would be dividing by an incorrect norm.
	vec_norm = vec_dir/np.dot(vec_dir,vec_dir)
	projs = proj_ortog(data,[vec_norm],orig=orig)
	return projs.flatten()

def proj_ortog_2d(
	data,
	plane_pnts):
	"""
	Created: 2020-04-10
	Modified: 2020-04-10
	Created from their non-orthogonal analogs.
	"""
	plane_pnts = np.array(plane_pnts)
	assert len(plane_pnts) == 3
	## Compute baricenter of three points, which will be the origin
	bc = (plane_pnts[0]+plane_pnts[1]+plane_pnts[2])/3.0
	## Vectors of the new base
	## Ojo! Una vez pense que habia que normalizarlos. No hay que hacerlo porque
	## las caras triangulares de los simplex de dimension superior a 2 son mas
	## pequenios, pero yo quiero poder dibujar mis proyecciones en el simplex
	## estander de 2d, entonces el tamanio de los vectores v1 y v2 TIENE QUE 
	## SER MAS PEQUENIO, adaptandose a cada dimensions. De esta manera, las
	## nuevas coordenadas 2d (a1 y a2) toman valores hasta 1 en vez de hasta
	## 0.9 y pico. De hecho, el v2, como se ve, hay que reescalarlo
	## tomando el tamanio que toma v1 de referencia, porque tamanio v1 es correcto.
	v1 = plane_pnts[0]-bc
	v2 = (plane_pnts[1]-plane_pnts[2])*(np.linalg.norm(v1)/np.linalg.norm(plane_pnts[1]-plane_pnts[2]))
	## Now, I normalize (divide by the norm) AND divide again by the norm so 
	## the projections are not the "actual" projections but a reescaled ones
	## that lie between the origin and the "tip" of the v1 and
	## v2 vectors. The result is that I divide by vec*vec=norm(vec)**2.0.
	## Because of that, it is necessary to have, not only the right direction
	## of the vector, but also THE RIGHT NORM before this "normalization"
	## (because I'm not building a unit vector)
	## Otherwise, I would be dividing by an incorrect norm.
	v1 = v1/np.dot(v1,v1) 
	v2 = v2/np.dot(v2,v2)
	new_base = [v1,v2]
	projs = proj_ortog(data,new_base,orig=bc)
	return projs

def proj_ortog_3d(
	data,
	tetra_pnts):
	"""
	Created: 2020-04-10
	Modified: 2020-04-10
	Created from their non-orthogonal analogs.
	"""
	# print ("THIS FUNCTION HAS NOT BEEN CHECKED. USE WITH CARE")
	# input()
	assert len(tetra_pnts) == 4
	tetra_pnts = np.array(tetra_pnts)
	## Compute baricenter of three points, which will be the origin
	bc = np.zeros(len(tetra_pnts[0]))
	for ttr_pnt in tetra_pnts:
		bc += ttr_pnt
	bc = bc/4.0
	## Vectors of the new base
	v1 = tetra_pnts[0]-bc ## x-axis vector
	v2 = tetra_pnts[1]-(tetra_pnts[2]+tetra_pnts[3])/2.0 ## y axis vector
	v2 = v2*(np.linalg.norm(v1)/np.linalg.norm(v2)) ## Normalizing
	v3 = tetra_pnts[2]-tetra_pnts[3] ## z axis vector
	v3 = v3*(np.linalg.norm(v1)/np.linalg.norm(v3)) ## Normalizing
	v1 = v1/np.dot(v1,v1)
	v2 = v2/np.dot(v2,v2)
	v3 = v3/np.dot(v3,v3)
	new_base = [v1,v2,v3]
	projs = proj_ortog(data,new_base,orig=bc)
	return projs

def vis_1d_proj_along_PCA(
	data,
	bins=200,
	show_values=True,
	show_poles_proj=True,
	poles_lr_order=None
	):
	"""
	Created: 2020-04-17
	Modified: 2020-06-29
	Add poles_lr_order list to fix the order of two of the poles from left to
	right (the advice would be to take the poles that are expected to be the
	most extreme). The format is [pole_left, pole_right] with the pole_xx the
	corresponding numerical label.
	"""
	data = np.array(data)
	gl_var = myvar_v3(data)
	# print ("Global variance: ", gl_var)
	l, var_exp, U = my_pca(data)
	# print ("Check if PCA equals to global variance (yes it does)", l, sum(l))
	pca_main_dir = U[:,-1]
	pca_main_dir = pca_main_dir / np.linalg.norm(pca_main_dir)
	if poles_lr_order:
		assert len(poles_lr_order) == 2
		dim = len(data[0])
		poles_coord = get_simplex_vertex(dim)
		poles_proj = proj_ortog(poles_coord,[pca_main_dir],orig=None)
		## Now, check if the poles are ordered and, if not, change the orientation
		## of the vector to point to the opposite direction.
		l_pole = poles_proj[poles_lr_order[0]]
		r_pole = poles_proj[poles_lr_order[1]]
		if l_pole > r_pole:
			pca_main_dir = -1.0*pca_main_dir
			## Assert that now I have the right direction
			poles_proj = proj_ortog(poles_coord,[pca_main_dir],orig=None)
			l_pole = poles_proj[poles_lr_order[0]]
			r_pole = poles_proj[poles_lr_order[1]]
		assert l_pole < r_pole
	proj = proj_ortog(data,[pca_main_dir],orig=None)
	prj_var = np.var(proj)
	fig = plt.figure(figsize=(.4*8,.4*6))
	ax = plt.axes()
	sns.distplot(proj,bins=bins)
	plt.xlabel("$x$")
	plt.ylabel("PDF")
	if show_poles_proj:
		dim = len(data[0])
		poles_coord = get_simplex_vertex(dim)
		poles_proj = proj_ortog(poles_coord,[pca_main_dir],orig=None)
		for i, xi in enumerate(poles_proj):
			plt.axvline(xi,ls=":",lw=0.5,color="0.7",zorder=0)
			ylims = ax.get_ylim()
			plt.text(xi, ylims[1], i,
				 ha='center',
				 va='bottom',
				 color = "0.7"
				)
	else:
		print ("WARNING! Poles' projections won't be shown")
	if show_values:
		plt.text(0.5, .93, f"$\\sigma^2_{{global}}=${gl_var:.02f}; $\\sigma^2_{{proj}}={prj_var:.02f}$\nExp. variance={100.0*l[-1]/sum(l):.00f}%",
				 ha='center',
				 va='top',
				 fontsize=8,
				 zorder=20,
				 transform=ax.transAxes,
				 bbox = dict(boxstyle='round', facecolor='w', alpha=0.5, lw=1)
				)
	plt.tight_layout()
	return fig

def myvar_v3(data):
    data = np.array(data)
    m = np.mean(data,axis=0)
    comps = data-m
    comps = comps*comps
    assert comps.shape == data.shape
    return np.sum(comps)/(1.0*len(data))

##############################################################################
## 2D triangle visualization
##############################################################################

def vis_all_AvB_projs(
	data,
	show_hist=False,
	save_fig=None,
	save_computation=None,
	load_computation=None,
	pickle_fig=None,
	project_close_only=False,
	show_pca=True,
	pca_align_mag="angle",
	show_var=True,
	proj_type="orthogonal",
	sharey=True,
	**kwargs
	):
	"""
	Created: 2020-02-06
	Modified: 2020-06-24
	Plots the opinions projected along all the possible axis of the
	form poles set A vs poles set B.
	
	For example, in a 3-pole system, it will show the 1v2, 1v3, 2v3,
	1v23, 2v13 and 3v12 axis.

	Modified to show histogram as default and pass kwargs to plot_AvB.
	"""
	dim = len(data[0])
	n_poles = dim+1
	
	if dim == 1:
		fig = plt.figure(figsize=(.4*8,.4*6))
		sns.distplot(data,bins=np.linspace(-1,1,30),hist=show_hist)
		plt.xlim(-1,1)
		plt.xlabel("$x$")
		plt.ylabel("PDF")
		plt.tight_layout()
		if save_fig:
			plt.savefig(save_fig+".pdf")
			plt.savefig(save_fig+".png",dpi=600,transparent=True)
		if pickle_fig:
			with open(pickle_hist,"wb") as f:
				pickle.dump(fig, f)
		return

	pls_coord = get_simplex_vertex(dim)

	## Build the figure (meter en funcion aparte)
	tot_num_projs = 0
	for subset_size in range(2,n_poles+1):
		max_set_A_size = int(subset_size/2.0)
		for _ in range(sp.special.comb(n_poles, subset_size,exact=True)):
			for size_A in range(1,max_set_A_size+1):
				for _ in range(sp.special.comb(subset_size, size_A,exact=True)):
					## Tiene que haber una forma mas elegante de hacer esto
					if subset_size/(1.0*size_A) == 2:
						tot_num_projs += 0.5
					else:
						tot_num_projs += 1
	tot_num_projs = int(tot_num_projs)

	print (tot_num_projs)
	cols = int(np.ceil(np.sqrt(tot_num_projs)))
	rows = int(np.ceil(tot_num_projs/(1.0*cols)))
	try:
		figsize = {3:(5,3),4:(8,6),5:(12*1.3,9*1.3)}[n_poles]
	except KeyError:
		figsize = (16,16)
	fig, axes = plt.subplots(rows,cols,sharex=True,sharey=sharey,figsize=figsize)

	plt_cntr = 0
	clr_cntr = 0
	unique_divisions = set()
	if project_close_only:
		_, pole_assignment = assign_to_closest_pole(data)

	if show_pca:
		_, var_exp, U = my_pca(data)
		## Array to store the alignment of each polarization axis with the 
		## principal components
		pca_alignment = np.zeros((tot_num_projs,U.shape[1]))
		pca_angles = np.zeros((tot_num_projs,U.shape[1]))
		print ("PCA explained variance: ",var_exp)
	ax_lst = []
	var_lst = []
	for subset_size in range(2,n_poles+1):
		## How many different kinds of divisions I will have
		max_set_A_size = int(subset_size/2.0)
		## All posible subsets of size grp_size
		for size_A in range(1,max_set_A_size+1):
			print (f"{size_A} vs {subset_size-size_A}")
			color = "C%d"%clr_cntr
			clr_cntr += 1
			for i, subset in enumerate(itertools.combinations(range(n_poles), subset_size)):
				## All possible divisions of the subset
				for i, poles_A_idx in enumerate(itertools.combinations(subset, size_A)):
					poles_B_idx = sorted(set(subset)-set(poles_A_idx))

					## Tiene que haber manera + elegante de hacer esto
					division_i = frozenset([frozenset(poles_A_idx) , frozenset(poles_B_idx)])
					if division_i in unique_divisions:
						continue
					else:
						unique_divisions.add(division_i)
					print (sorted(poles_A_idx), poles_B_idx)
					if show_pca:
						pca_al_i, pca_angle_i = get_alignment_AvB_axis_to_PCA(
									pls_coord,
									poles_A_idx,
									poles_B_idx,
									U)
						pca_alignment[plt_cntr,:] = pca_al_i
						pca_angles[plt_cntr,:] = pca_angle_i

					if project_close_only:
						data_final = filter_data_closer_poles(
							data,
							pole_assignment,
							set(poles_A_idx).union(set(poles_B_idx)))
					else:
						data_final = data

					## Plot
					yi = int(plt_cntr/cols)
					xi = int(plt_cntr%cols)
					ax = axes[yi,xi]
					plt.sca(ax)
					ax, var = plot_AvB(
						data_final,
						poles_A_idx,
						poles_B_idx,
						color = color,
						show_hist = show_hist,
						rotation=90,
						ax=ax,
						save_computation=save_computation,
						load_computation=load_computation,
						show_var=show_var,
						proj_type=proj_type,
						hist_bins=np.linspace(-1,1,61),
						**kwargs
						)
					ax_lst.append(ax)
					if show_var:
						var_lst.append(var)

					plt_cntr += 1
	if show_var:
		var_max = max(var_lst)
		for i, ax in enumerate(ax_lst):
			plt.sca(ax)
			var = var_lst[i]
			rect = patches.Rectangle(
				(0.35,0.77),
				0.3,0.17,
				linewidth=1,
				edgecolor='k',
				transform=ax.transAxes,
				facecolor='none')
			ax.add_patch(rect)
			rect = patches.Rectangle(
				(0.35,0.77),
				0.3*var/var_max,0.17,
				edgecolor='none',
				transform=ax.transAxes,
				facecolor='r',
				alpha=0.4,
				zorder=0)
			ax.add_patch(rect)
	ax = axes[-1,0]
	plt.sca(ax)
	plt.xlabel("$x$")
	plt.ylabel("PDF")

	## Evitar mostrar ejes vacios en los axes sobrantes
	for i in range(plt_cntr,cols*rows):
		yi = int(i/cols)
		xi = int(i%cols)
		ax = axes[yi,xi]
		plt.sca(ax)
		plt.axis("off")
	plt.tight_layout()
	if show_pca:
		pca_al_max = np.argmax(pca_alignment,axis=0)
		#### DEBUG
		# print ("pca_alignment\n",pca_alignment)
		# print ("pca_al_max\n",pca_al_max)
		####
		assert np.sum(pca_al_max == np.argmin(pca_angles,axis=0)) == len(pca_al_max)
		for i,idx in enumerate(pca_al_max):
			ax = ax_lst[idx]
			plt.sca(ax)
			if pca_align_mag == "cosine":
				plt.text(0.5, 1.0, f"{var_exp[i]*100:.00f}%/{pca_alignment[idx,i]*100:.00f}%",
				 color="r",
				 ha='center',
				 va='bottom',
				 transform=ax.transAxes,
				 #bbox = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor=color, lw=1)
				)
			elif pca_align_mag == "angle":
				plt.text(0.5, 1.0, f"{var_exp[i]*100:.00f}%/{pca_angles[idx,i]:.00f}º",
				 color="r",
				 ha='center',
				 va='bottom',
				 transform=ax.transAxes,
				 #bbox = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor=color, lw=1)
				)
			else:
				raise Exception(f"{pca_align_mag} is not a valid value for pca_align_mag")
	if save_fig:
		plt.savefig(save_fig+".pdf")
		plt.savefig(save_fig+".png",transparent=True,dpi=600)
	if pickle_fig:
		with open(pickle_fig,"wb") as f:
			pickle.dump(fig,f)
	return fig

def plot_AvB(
	data,
	poles_A_idx,
	poles_B_idx,
	ax = None,
	color = None,
	show_hist = False,
	hist_bins = np.linspace(-1,1,100),
	rotation = 90,
	xlbl=None,
	ylbl=None,
	save_computation = None,
	load_computation = None,
	show_var=False,
	proj_type = "orthogonal"
	):
	"""
	Created: 2020-02-06
	Modified: 2020-02-06
	Projects the points to the subspace spanned by the chosen poles
	(all of them, both from set A and set B) from the rest of the poles
	and then projects orthogonally to the line that joins the barycenter
	of set A with barycenter of set B.
	
	The input for poles_X_idx is a list or tuple of integer indices to 
	choose the pole coordinates from the output of get_simplex_vertex.
	"""
	if ax is None:
		ax = plt.axes()
	plt.sca(ax)

	if load_computation:
		with open(load_computation+f"_{poles_A_idx}vs{poles_B_idx}.p","rb") as f:
			prjs_1d = pickle.load(f)
	else:
		prjs_1d = proj_AvB(data,poles_A_idx,poles_B_idx,proj_type=proj_type)
	if save_computation:
		with open(save_computation+f"_{poles_A_idx}vs{poles_B_idx}.p","wb") as f:
			pickle.dump(prjs_1d,f)
	## Plot
	sns.distplot(prjs_1d,bins=hist_bins,color=color,hist=show_hist)
	## Write the labels of the poles that are compared in their 
	## respective extremes
	plt.text(0.1, 0.9, ";".join(map(str,poles_A_idx)),
			 rotation=rotation,
			 ha='left',
			 va='top', 
			 transform=ax.transAxes,
			 bbox = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor=color, lw=1)
			)
	plt.text(0.9, 0.9, ";".join(map(str,poles_B_idx)),
			 rotation=rotation,
			 ha='right',
			 va='top', 
			 transform=ax.transAxes,
			 bbox = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor=color, lw=1)
			)
	if show_var:
		var = np.var(prjs_1d)
		plt.text(0.5, 0.77, f"{var:.02f}",
			 ha='center',
			 va='bottom',
			 transform=ax.transAxes,
			 #bbox = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor=color, lw=1)
			)
	plt.xlim(-1,1)
	plt.xlabel(xlbl)
	plt.ylabel(ylbl)
	if show_var:
		return ax, var
	else:
		return ax, None

def proj_AvB(data,
	poles_A_idx,
	poles_B_idx,
	get_bad_pnts=False,
	proj_type = "orthogonal"
	):
	"""
	Created: 2020-02-06
	Modified: 2020-02-06
	Projects the points to the subspace spanned by the chosen poles
	(all of them, both from set A and set B) from the rest of the poles
	and then projects orthogonally to the line that joins the barycenter
	of set A with barycenter of set B.
	
	The input for poles_X_idx is a list or tuple of integer indices to 
	choose the pole coordinates from the output of get_simplex_vertex.
	"""
	assert type(poles_A_idx[0]) is int
	assert type(poles_B_idx[0]) is int
	dim = len(data[0])
	n_poles = dim+1
	pls_coord = get_simplex_vertex(dim)
	
	poles_B = np.array([pls_coord[i] for i in poles_B_idx])
	poles_A = np.array([pls_coord[i] for i in poles_A_idx])

	if proj_type == "from_pole":
		poles_AB = np.concatenate((poles_A,poles_B),axis=0)
		## Proyecto primero sobre el subespacio de poles_A y poles_B
		## Para ello, proyecto desde el resto de polos hasta la variedad
		## lineal definida por poles_A y poles_B juntos
		poles_rest_idx = set(range(n_poles))-set(poles_A_idx)-set(poles_B_idx)
		poles_rest = np.array([pls_coord[i] for i in poles_rest_idx])
		if not get_bad_pnts:
			proj_subesp = proyect_nd_to_md_num_PAR(
				data,
				proy_from=poles_rest,
				proy_to=poles_AB)
		else:
			proj_subesp, bad_pnts = proyect_nd_to_md_num_PAR(
				data,
				proy_from=poles_rest,
				proy_to=poles_AB,
				get_bad_pnts=get_bad_pnts)
		## Luego calculo la proyeccion sobre el eje A vs B
		prjs_1d = axis_1d_projection(proj_subesp,poles_A,poles_B)
		if not get_bad_pnts:
			return prjs_1d
		else:
			return prjs_1d, bad_pnts
	elif proj_type == "orthogonal":
		prjs_1d = axis_1d_proj_ortog(data,poles_A,poles_B)
		return prjs_1d
	else:
		raise Exception(f"{proj_type} is not a valid value for proj_type")

def vis_2d_projections(
	data,
	proj_type = "orthogonal",
	show_points = False,
	show_2d_kde = False,
	show_1d_kde = False,
	save_fig_data = False,
	load_fig_data = False,
	show_global_pca = True,
	show_proj_pca = False,
	project_close_only = False,
	poles_lbls = None,
	**kwargs
	):
	"""
	Created: 2020-02-05
	Modified: 2020-06-29
	To project >2d opinion spaces to the 2d triangular faces of the simplex.
	kwargs are passed to the vis_triangle function.
	I WAS USING SAME VARIABLE i TWICE. STILL IT SEEMED TO WORK ALL RIGHT??
	"""
	data = np.array(data)
	dim = len(data[0])
	if not poles_lbls:
		poles_lbls = range(dim+1)
	else:
		assert len(poles_lbls) == dim+1
	try:
		pca_vec_width = {1:None,2:1.0,3:0.7,4:0.5}[dim]
	except KeyError:
		pca_vec_width = 0.5
	try:
		hist_1d_offset = {1:None,2:0.01,3:0.025,4:0.03}[dim]
	except KeyError:
		hist_1d_offset = 0.03
	## Pa porsi soy torpe y llamo a esta funcion cuando no tiene sentido (para dimensiones <3)
	if dim == 1:
		sns.distplot(data,kde=show_1d_kde,norm_hist=True)
		plt.xlabel("$x$")
		plt.ylabel("PDF")
		#raise Exception("These are not the droids, ehem function, you are looking for")
		return
	if dim == 2:
		if show_global_pca:
			_, var_exp, U = my_pca(data)
			print ("PCA explained variance: ",var_exp)
			## OJO! AL MULTIPLICAR exp_var POR EL VECTOR UNITARIO DE DIRECCION
			## DE MAXIMA VARIANZA LA BRUJULA DE POLARIZACION PARA UN SISTEMA 
			## TRIPOLAR (EL QUE YA ESTA EN 2d) NO TIENE LA MISMA INTERPRETACION
			## QUE PARA UN SISTEMA CON MAS POLOS (caso A: varianza explicada.
			## caso B: tamanio proyeccion)
			# exp_var = var_exp[-1]
			# pol_vect = exp_var*U[:,-1]
			pol_vect = U[:,-1]
		else:
			pol_vect = None
		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111)
		vis_triangle(
			data,
			ax=ax,
			show_points=show_points,
			show_2d_kde=show_2d_kde,
			show_1d_kde=show_1d_kde,
			show_pca=show_proj_pca,
			pol_vect = pol_vect,
			project_close_only=project_close_only,
			proj_type=proj_type,
			pca_vec_width=pca_vec_width,
			hist_1d_offset=hist_1d_offset,
			poles_lbls=poles_lbls,
			**kwargs
		)
		return fig
	n_poles = dim+1

	pls_coord = get_simplex_vertex(dim)

	## Build the axes to plot
	n_tris = sp.special.comb(n_poles,3,exact=True)
	cols = int(np.ceil(np.sqrt(n_tris)))
	rows = int(np.ceil(n_tris/(1.0*cols)))
	fig, axes = plt.subplots(rows,cols, frameon=False, gridspec_kw={"wspace":0.0,"hspace":0.25})

	if show_global_pca:
		print ("Computing global polarization compass...")
		_, exp_var, U = my_pca(data)
		print ("Explained variance: ", exp_var)
		global_pol_vect = U[:,-1]

	if project_close_only:
		## To project only data points that are closer to the projection
		## poles than to any other to avoid irrelevant projections (users that
		## are much closer to a given pole and whose contribution to any 
		## other pole is residual)
		_, pole_assignment = assign_to_closest_pole(data)

	for i, poles_to_idx in enumerate(itertools.combinations(range(n_poles), 3)):
		poles_from_idx = set(range(n_poles))-set(poles_to_idx)
		poles_from = np.array([pls_coord[j] for j in poles_from_idx])
		poles_to = np.array([pls_coord[j] for j in poles_to_idx])
		if project_close_only:
			data_final = filter_data_closer_poles(
				data,
				pole_assignment,
				poles_to_idx)
		else:
			data_final = data
		if proj_type == "from_pole":
			## project to 2D
			proj_2d = proyect_nd_to_md_num_PAR(
				data_final,
				poles_from,
				poles_to)
			## Change of reference to "standard" triangle
			new_2d_coord = new_2d_coords_num(
				proj_2d,
				poles_to
				)
		elif proj_type == "orthogonal":
			new_2d_coord = proj_ortog_2d(data_final,poles_to)
		else:
			raise Exception(f"{proj_type} is not a valid value for proj_type")
		## Plot
		yi = int(i/cols)
		xi = int(i%cols)
		ax = axes[yi,xi]
		plt.sca(ax)
		if save_fig_data:
			save_fig_data_final = save_fig_data+"_%d"%i
		else:
			save_fig_data_final = save_fig_data
		if load_fig_data:
			load_fig_data_final = load_fig_data+"_%d"%i
		else:
			load_fig_data_final = load_fig_data
		if show_global_pca:
			bc = (poles_to[0]+poles_to[1]+poles_to[2])/3.0
			v1 = poles_to[0]-bc
			v2 = (poles_to[1]-poles_to[2])*(np.linalg.norm(v1)/np.linalg.norm(poles_to[1]-poles_to[2]))
			v1 = v1 / np.linalg.norm(v1)
			v2 = v2 / np.linalg.norm(v2)
			a1 = np.dot(global_pol_vect,v1)
			a2 = np.dot(global_pol_vect,v2)
			pol_vect = [a1,a2]
		else:
			pol_vect = None
		vis_triangle(
			new_2d_coord,
			ax=ax,
			poles_lbls=[poles_lbls[pi] for pi in poles_to_idx],
			show_points=show_points,
			show_2d_kde=show_2d_kde,
			show_1d_kde=show_1d_kde,
			save_fig_data=save_fig_data_final,
			load_fig_data=load_fig_data_final,
			show_pca=show_proj_pca,
			pol_vect=pol_vect,
			project_close_only=project_close_only,
			proj_type=proj_type,
			pca_vec_width=pca_vec_width,
			hist_1d_offset=hist_1d_offset,
			**kwargs
		)

	for axvec in axes:
		for ax in axvec:
			plt.sca(ax)
			ax.axis("equal")
			plt.axis('off')
			plt.xlim(-1.5,1.5)
			plt.ylim(-1.5,1.5)
	return fig

def vis_triangle(
	pnt_lst,
	poles_lbls = None,
	ax = None,
	show_1d_hist = True,
	hist_1d_bins = 60,
	hist_1d_offset = 0.025,
	show_1d_kde = False,
	show_2d_kde = False,
	show_2d_hist = True,
	show_2d_contours = False,
	show_poles = True,
	show_points = True,
	show_colorbar = False,
	pnt_alpha = 1.0,
	pnt_color = "w",
	pnt_size = 3,
	kde_grid_div = 200,
	kde_levels = 500,
	hist2d_bins = 200,
	cntr_levels = 20,
	save_fig_data = False,
	load_fig_data = False,
	pol_vect = None,
	pol_vect_clr = "w",
	show_pca = False,
	pca_vec_width = 0.5,
	project_close_only = False,
	proj_type="orthogonal",
	show_barycenter = False,
	show_center = False
	):
	"""
	Created: 2019-09-16
	Modified: 2020-06-24
	2D visualization of an equilateral triangle with "opinions" (points) over it.
	Modified function to ALWAYS show triangle even if show_poles is False.
	Modified to show barycenter, neutral point (0,0) and white contours.
	"""
	if show_2d_kde and show_2d_hist:
		raise Exception("Choose either 2D KDE or 2D histogram, both make no sense.")
	if save_fig_data:
		fig_data = {}
	if load_fig_data:
		with open(load_fig_data,"rb") as f:
			fig_data_old = pickle.load(f)
	if ax is None:
		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111)
	assert len(pnt_lst[0]) == 2
	if poles_lbls is None:
		poles_lbls = range(3)
	assert len(poles_lbls) == 3
	vs = get_simplex_vertex(2)
	for i in range(3):
		#plt.plot([vs[i,0]],[vs[i,1]],"ko",ms=15,zorder=19)
		# plt.text(vs[i,0],vs[i,1],
		#		  poles_lbls[i],
		#		  ha="center",
		#		  va="center",
		#		  color="white",
		#		  zorder=20)
		for j in range(3):
			x = vs[[i,j],0] ## Select the x coord of two poles
			y = vs[[i,j],1] ## Select the y coord of two poles
			plt.plot(x,y,'k-',zorder=0)
	if show_poles:
		## Alternative way of labeling the corners
		plt.text(vs[0,0],vs[0,1],
						 poles_lbls[0],
						 ha="left",
						 va="center",
						 color="black",
						 weight="bold",
						 zorder=20) 
		plt.text(vs[1,0],vs[1,1],
						 poles_lbls[1],
						 ha="right",
						 va="bottom",
						 color="black",
						 weight="bold",
						 zorder=20)
		plt.text(vs[2,0],vs[2,1],
						 poles_lbls[2],
						 ha="right",
						 va="top",
						 color="black",
						 weight="bold",
						 zorder=20)									 
	pnt_lst = np.array(pnt_lst)
	if show_points:
		print ("Plotting points...")
		plt.plot(pnt_lst[:,0],pnt_lst[:,1],"o",
		ms=pnt_size,
		color=pnt_color,
		markeredgewidth=0.5,
		markeredgecolor="k",
		alpha=pnt_alpha)
	## Build and plot 1d KDE and histogram
	if show_1d_kde or show_1d_hist:
		print ("Projecting 2D points to the sides of triangle...")
		if load_fig_data:
			pry12_1d, pry13_1d, pry23_1d = fig_data_old["1d_projs"]
		else:
			pry12_1d, pry13_1d, pry23_1d = vis_triangle_1d_proy_helper_num(pnt_lst,project_close_only,proj_type=proj_type)
		if save_fig_data:
			fig_data["1d_projs"] = (pry12_1d,pry13_1d,pry23_1d)
	if show_1d_kde:
		print ("Plotting 1D kde...")
		if load_fig_data:
			proj_1d_kdes = plot_kde_on_triangle_vis(
				pry12_1d, pry13_1d, pry23_1d,
				ax,
				precomp=True,
				fig_data_old = fig_data_old["1d_kdes"]
				)
		else:
			proj_1d_kdes = plot_kde_on_triangle_vis(
				pry12_1d, pry13_1d, pry23_1d,
				ax)
		if save_fig_data:
			fig_data["1d_kdes"] = proj_1d_kdes
	if show_1d_hist:
		print ("Plotting 1D histogram...")
		plot_hist_on_triangle_vis(
			pry12_1d, pry13_1d, pry23_1d,
			ax,
			bins = hist_1d_bins,
			offset=hist_1d_offset)
	if show_2d_kde:
		print ("Plotting 2D kde...")
		if load_fig_data:
			ax, xx_2dkde, yy_2dkde, f_2dkde = plot_2d_kde_tri(
						fig_data_old["2d_kde"],
						ax,
						grid_div = kde_grid_div,
						levels = kde_levels,
						precomp=True)
		else:
			ax, xx_2dkde, yy_2dkde, f_2dkde = plot_2d_kde_tri(pnt_lst,ax,
					   grid_div = kde_grid_div,
					   levels = kde_levels)
		if save_fig_data:
			fig_data["2d_kde"] = (xx_2dkde, yy_2dkde, f_2dkde)
		if show_colorbar:
			plt.sca(ax)
			if show_1d_hist or show_1d_kde:
				cbaxes = fig.add_axes([0.87, 0.275, 0.025, 0.45])  # This is the position for the colorbar
				cb = plt.colorbar(cax = cbaxes)
				cb.ax.tick_params(labelsize=14)
				cbaxes.yaxis.set_ticks_position('right')

				def fmt(s):
					try:
						n = "{:.1f}".format(float(s))
					except:
						n = ""
					return n

				cb.ax.set_yticklabels([fmt(label.get_text()) for label in cb.ax.get_yticklabels()])
				cb.set_label(label='Density of users', size=14, labelpad=5)
			else:
				cbaxes = fig.add_axes([0.32, 0.275, 0.025, 0.45])  # This is the position for the colorbar
				cb = plt.colorbar(cax = cbaxes)
				cb.ax.tick_params(labelsize=14)
				cbaxes.yaxis.set_ticks_position('left')
				def fmt(s):
					try:
						n = "{:.1f}".format(float(s))
					except:
						n = ""
					return n
				cb.ax.set_yticklabels([fmt(label.get_text()) for label in cb.ax.get_yticklabels()])
				cb.set_label(label='Density of users', size=14, labelpad=-60)
	if show_2d_hist:
		print ("Plotting 2D histogram...")
		ax = plot_2d_hist_triangle(
			pnt_lst,
			ax=ax,
			hist2d_bins=hist2d_bins)
		if show_colorbar:
			if show_1d_hist or show_1d_kde:
				cbaxes = fig.add_axes([0.87, 0.275, 0.025, 0.45])  # This is the position for the colorbar
				cb = plt.colorbar(cax = cbaxes)
				cb.ax.tick_params(labelsize=14)
				cb.set_label(label='No. of users', size=14, labelpad=5)
				cbaxes.yaxis.set_ticks_position('right')
			else:
				cbaxes = fig.add_axes([0.32, 0.275, 0.025, 0.45])  # This is the position for the colorbar
				cb = plt.colorbar(cax = cbaxes)
				cb.ax.tick_params(labelsize=14)
				cb.set_label(label='No. of users', size=14, labelpad=-60)
				cbaxes.yaxis.set_ticks_position('left')
	if show_2d_contours:
		print ("Plotting 2D contours...")
		if load_fig_data:
			ax, xx_2dkde, yy_2dkde, f_2dkde = plot_2d_cntr_tri(
						fig_data_old["2d_kde"],
						ax,
						grid_div = kde_grid_div,
						nlevels = cntr_levels,
						precomp=True)
		else:
			ax, xx_2dkde, yy_2dkde, f_2dkde = plot_2d_cntr_tri(pnt_lst,ax,
					   grid_div = kde_grid_div,
					   nlevels = cntr_levels)
		if save_fig_data:
			fig_data["2d_kde"] = (xx_2dkde, yy_2dkde, f_2dkde)
	plt.sca(ax)
	if pol_vect is not None:
		print ("Plotting polarization compass...")
		assert len(pol_vect) == 2
		pol_proj = 1.0*np.linalg.norm(pol_vect)
		assert pol_proj <= 1.0
		ux, uy = pol_vect / pol_proj
		plt.annotate('', 
			xy=(-.5*ux,-.5*uy), 
			xytext=(.5*ux,.5*uy), 
			arrowprops=dict(
				arrowstyle='<|-|>',
				shrinkA=0, shrinkB=0,
				color=pol_vect_clr,
				lw=pca_vec_width))
		## To avoid showing the perpendicular lines if the alignment is perfect
		## Which should only happen in 2D
		if np.abs(pol_proj-1.0) > 1e-10:
			plt.annotate('',
				xy=(-.5*pol_proj*ux,-.5*pol_proj*uy), 
				xytext=(.5*pol_proj*ux,.5*pol_proj*uy), 
				arrowprops=dict(
					arrowstyle='|-|, widthA=0.5,widthB=0.5',
					shrinkA=0, shrinkB=0,
					color=pol_vect_clr,
					lw=pca_vec_width))
	if show_pca:
		_, var_exp, U = my_pca(pnt_lst)
		print ("Projected PCA explained variance: ",var_exp)
		exp_var = var_exp[-1]
		ux, uy = U[:,-1]
		plt.plot([-.5*ux,.5*ux],[-.5*uy,.5*uy],"-w")
		#plt.plot(-.5*ux,-.5*uy,"ow")
		#plt.plot(.5*ux,.5*uy,"ow")
		plt.plot(-.5*exp_var*ux,-.5*exp_var*uy,"ow")
		plt.plot(.5*exp_var*ux,.5*exp_var*uy,"ow")
	if show_center:
		plt.plot(0,0,
	        "o",
	        markerfacecolor="none",
	        markeredgewidth=1,
	        markeredgecolor="w",
	        ms=4)
	if show_barycenter:
		bcntr = np.mean(pnt_lst,axis=0)
		plt.plot(bcntr[0],bcntr[1],"sw",ms=4)
	plt.axis("off")
	plt.axis("equal")
	plt.xlim(-1.5,1.5)
	plt.ylim(-1.5,1.5)
	if save_fig_data:
		with open(save_fig_data,"wb") as f:
			pickle.dump(fig_data,f)
	return ax

def plot_2d_hist_triangle(
		data,
		ax=None,
		hist2d_bins=200):
	"""
	Created: 2020-03-02
	Modified: 2020-03-02
	2D histogram triangular visualization.
	2021-12-10: low->lower
	"""
	if ax is None:
		ax = plt.axes()
	data = np.array(data)
	assert data.shape[1] == 2 ## Make sure that I have 2d data
	poles_2d = get_simplex_vertex(2)
	H,xedges,yedges = np.histogram2d(data[:,0],data[:,1],bins=hist2d_bins)
	xcenters = (xedges[1:]+xedges[:-1])/2.0
	ycenters = (yedges[1:]+yedges[:-1])/2.0
	##################################
	## Build a triangular mask
	msk = np.zeros_like(H) + False
	for i, xi in enumerate(xcenters):
		for j, yj in enumerate(ycenters):
			bar1 =\
			((poles_2d[1,1]-poles_2d[2,1])*(xi-poles_2d[2,0])+\
			 (poles_2d[2,0]-poles_2d[1,0])*(yj-poles_2d[2,1])) /\
			((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
			 (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
			bar2 =\
			((poles_2d[2,1]-poles_2d[0,1])*(xi-poles_2d[2,0])+\
			 (poles_2d[0,0]-poles_2d[2,0])*(yj-poles_2d[2,1])) /\
			((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
			 (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
			bar3 = 1.0-bar1-bar2
			yes_no = bar1 >= -0.0 and bar1 <= 1.0 and bar2 >= -0.0 and bar2 <= 1.0 and bar3 >= -0.0 and bar3 <= 1.0
			msk[i,j] = not yes_no
	##################################
	####### To avoid having white pixels in the histogram
	H[H==0] = 1.0
	#######
	H_msk = np.ma.masked_array(H, mask=msk)
	myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
	plt.sca(ax)
	plt.imshow(
		H_msk.T,
		origin='lower',
		extent=myextent,
		interpolation='nearest',
		aspect='auto',
		cmap="inferno",
		norm=colors.LogNorm()
	)
	return ax

def compute_2d_kde_tri(
	data,
	grid_div = 100,
	mask_not_tri = True,
	pnts_bynd=False):
	"""
	Created: 2020-06-24
	Modified: 2020-06-24
	Function to compute 2D gaussian KDE. To make functions more modular.
	"""
	data = np.array(data)
	poles_2d = get_simplex_vertex(2)
	
	x = data[:, 0]
	y = data[:, 1]
	if pnts_bynd:
		xmin, xmax = 1.5*min(data[:,0]), 1.5*max(data[:,0])
		ymin, ymax = 1.5*min(data[:,1]), 1.5*max(data[:,1])
	else:
		xmin, xmax = min(data[:,0]), max(data[:,0])
		ymin, ymax = min(data[:,1]), max(data[:,1])

	# Peform the kernel density estimate
	xx, yy = np.mgrid[xmin:xmax:grid_div*1j, ymin:ymax:grid_div*1j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	print ("Computing 2D kernel...")
	kernel = stats.gaussian_kde(values)
	krnl_vals = kernel(positions).T

	if mask_not_tri:
		############# Mask for points outside triangle using baricentric coordinates
		print ("Masking points outside triangle...")
		bar1 =\
		((poles_2d[1,1]-poles_2d[2,1])*(positions[0,:]-poles_2d[2,0])+\
		 (poles_2d[2,0]-poles_2d[1,0])*(positions[1,:]-poles_2d[2,1])) /\
		((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
		 (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
		bar2 =\
		((poles_2d[2,1]-poles_2d[0,1])*(positions[0,:]-poles_2d[2,0])+\
		 (poles_2d[0,0]-poles_2d[2,0])*(positions[1,:]-poles_2d[2,1])) /\
		((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
		 (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
		bar3 = 1.0-bar1-bar2
		mask = np.logical_and(bar1>=-0.0,bar1<=1.0)
		mask = np.logical_and(mask,bar2>=-0.0)
		mask = np.logical_and(mask,bar2<=1.0)
		mask = np.logical_and(mask,bar3>=-0.0)
		mask = np.logical_and(mask,bar3<=1.0)
		print (sum(mask))
		#############

		krnl_vals_ma = np.ma.masked_array(krnl_vals, mask=np.logical_not(mask))
	else:
		krnl_vals_ma = krnl_vals
	f = krnl_vals_ma.reshape(xx.shape)
	return xx, yy, f

def plot_2d_kde_tri(data,ax,
				   grid_div = 100,
				   levels = 500,
				   precomp=False):
	"""
	Created: 2020-02-03
	Modified: 2020-02-03
	2D KDE visualization of a cloud of points inside the triangle defined by 
	the three poles.
	"""
	plt.sca(ax)
	if not precomp:
		assert len(data[0]) == 2
		xx,yy,f = compute_2d_kde_tri(data,grid_div)
		# data = np.array(data)
		# poles_2d = get_simplex_vertex(2)
		
		# x = data[:, 0]
		# y = data[:, 1]
		# xmin, xmax = min(data[:,0]), max(data[:,0])
		# ymin, ymax = min(data[:,1]), max(data[:,1])

		# # Peform the kernel density estimate
		# xx, yy = np.mgrid[xmin:xmax:grid_div*1j, ymin:ymax:grid_div*1j]
		# positions = np.vstack([xx.ravel(), yy.ravel()])
		# values = np.vstack([x, y])
		# print ("Computing 2D kernel...")
		# kernel = stats.gaussian_kde(values)
		# krnl_vals = kernel(positions).T

		# ############# Mask for points outside triangle using baricentric coordinates
		# print ("Masking points outside triangle...")
		# bar1 =\
		# ((poles_2d[1,1]-poles_2d[2,1])*(positions[0,:]-poles_2d[2,0])+\
		#  (poles_2d[2,0]-poles_2d[1,0])*(positions[1,:]-poles_2d[2,1])) /\
		# ((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
		#  (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
		# bar2 =\
		# ((poles_2d[2,1]-poles_2d[0,1])*(positions[0,:]-poles_2d[2,0])+\
		#  (poles_2d[0,0]-poles_2d[2,0])*(positions[1,:]-poles_2d[2,1])) /\
		# ((poles_2d[1,1]-poles_2d[2,1])*(poles_2d[0,0]-poles_2d[2,0])+\
		#  (poles_2d[2,0]-poles_2d[1,0])*(poles_2d[0,1]-poles_2d[2,1]))
		# bar3 = 1.0-bar1-bar2
		# mask = np.logical_and(bar1>=-0.0,bar1<=1.0)
		# mask = np.logical_and(mask,bar2>=-0.0)
		# mask = np.logical_and(mask,bar2<=1.0)
		# mask = np.logical_and(mask,bar3>=-0.0)
		# mask = np.logical_and(mask,bar3<=1.0)
		# print (sum(mask))
		# #############

		# krnl_vals_ma = np.ma.masked_array(krnl_vals, mask=np.logical_not(mask))
		# f = krnl_vals_ma.reshape(xx.shape)
	else:
		xx, yy, f = data

	print ("Plotting contours...")
	# Contourf plot
	cfset = plt.contourf(xx, yy, f,
						levels=levels, 
						cmap='inferno')
	
	return ax, xx, yy, f

def plot_2d_cntr_tri(data,ax,
				   grid_div = 100,
				   nlevels = 20,
				   precomp=False):
	"""
	Created: 2020-02-03
	Modified: 2020-02-03
	2D KDE visualization of a cloud of points inside the triangle defined by 
	the three poles.
	"""
	plt.sca(ax)
	if not precomp:
		assert len(data[0]) == 2
		xx,yy,f = compute_2d_kde_tri(data,grid_div)
	else:
		xx, yy, f = data

	## 2 thirds of the levels will be logarithmically spaced for the whole range
	## and 1 third will be plot in the higher values
	levels = np.logspace(0.3*np.log10(np.min(f)),np.log10(np.max(f))*0.95,nlevels)
	# levels2 = np.logspace(np.log10(0.5*np.max(f)),np.log10(np.max(f)),nlevels/3)
	# levels = np.sort(np.concatenate((levels1,levels2)))
	# levels = np.linspace(np.min(f),np.max(f)*.95,nlevels)

	## Include contours associated to local maxima
	# gradients = np.gradient(f)
	# x_grad = gradients[0]
	# y_grad = gradients[1]
	# xmsk = np.isclose(x_grad,0,atol=1e-4,rtol=1e-4)
	# ymsk = np.isclose(y_grad,0,atol=1e-4,rtol=1e-4)
	# fullmsk = np.logical_and(xmsk,ymsk)
	# local_max_indx = np.where(fullmsk==True)
	# local_max = sorted([f[i,j] for i,j in zip(local_max_indx[0],local_max_indx[1])],reverse=True)
	# local_max = np.array(local_max[:5])*.99
	# levels = np.concatenate((levels,local_max))
	# print (levels)
	# levels = np.sort(levels)


	print ("Plotting contours...")
	# Contourf plot
	plt.contour(
		xx, 
		yy,
		f,
		levels=levels,
		colors="w",
		linewidths=0.25,
		)
	return ax, xx, yy, f

def plot_kde_on_triangle_vis(
	pry12_1d, pry13_1d, pry23_1d,
	ax,
	colors_lst = ["C2","C0","C1"],
	alpha = 0.5,
	precomp=False,
	fig_data_old = None
	):
	"""
	Created: 2019-09-22
	Modified: 2020-02-06
	Modificado para corregir el calculo del KDE gaussiano, que tenia una
	bandwidth fija en vez de una en funcion de los datos.
	"""
	fig_data = {}
	plt.sca(ax)
	vs = get_simplex_vertex(2)
	p1 = vs[0]
	p2 = vs[1]
	p3 = vs[2]
	##############
	## 12
	## Calcular KDE
	#*X = np.array(pry12_1d)[:,np.newaxis]
	#*X_plot = np.linspace(-1, 1, 100)[:,np.newaxis]
	#*kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
	#*log_dens = kde.score_samples(X_plot)
	if not precomp:
		X = np.array(pry12_1d)
		X_plot = np.linspace(-1, 1, 100)
		kde = stats.gaussian_kde(X)
		y_kde = kde(X_plot)
	else:
		X_plot = fig_data_old["X_plot"]
		y_kde = fig_data_old["pry12_kde"]
	fig_data["X_plot"] = X_plot
	fig_data["pry12_kde"] = y_kde

	## I re-escale the KDE plot to avoid having parts of the plot outside the figure's limits
	y_kde = 0.5*y_kde/np.max(y_kde)

	## pintar KDE sobre plot anterior   
	mid_pnt = (p1+p2)/2.0
	l = mid_pnt[0] - np.linalg.norm(p1-p2)/2.0
	r = mid_pnt[0] + np.linalg.norm(p1-p2)/2.0

	## I have to invert the axis of the x values because in the 
	## triangle pole 1 is on the right and pole 2 on the left
	## and the 2d polarization projection is always computed 
	## with -1 in the smaller pole and 1 in the larger pole.
	X_plt_trns = np.linspace(l,r,100)[::-1,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],-30)+ax.transData

	#*y_trns = np.exp(log_dens)-min(np.exp(log_dens))+mid_pnt[1]
	y_trns = y_kde+mid_pnt[1]

	plt.fill_between(X_plt_trns[:,0],mid_pnt[1],y_trns ,transform=tr,alpha=alpha,color=colors_lst[0],linewidth=0.0)

	#############
	## 13
	## Calcular KDE
	#*X = np.array(pry13_1d)[:,np.newaxis]
	#*X_plot = np.linspace(-1, 1, 100)[:,np.newaxis]
	#*kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
	#*log_dens = kde.score_samples(X_plot)
	if not precomp:
		X = np.array(pry13_1d)
		X_plot = np.linspace(-1, 1, 100)
		kde = stats.gaussian_kde(X)
		y_kde = kde(X_plot)
	else:
		X_plot = fig_data_old["X_plot"]
		y_kde = fig_data_old["pry13_kde"]
	fig_data["pry13_kde"] = y_kde

	## I re-escale the KDE plot to avoid having parts of the plot outside the figure's limits
	y_kde = 0.5*y_kde/np.max(y_kde)

	## pintar KDE sobre plot anterior

	mid_pnt = (p1+p3)/2.0
	l = mid_pnt[0] - np.linalg.norm(p1-p3)/2.0
	r = mid_pnt[0] + np.linalg.norm(p1-p3)/2.0

	X_plt_trns = np.linspace(l,r,100)[:,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],-150)+ax.transData

	#*y_trns = np.exp(log_dens)-min(np.exp(log_dens))+mid_pnt[1]
	y_trns = y_kde+mid_pnt[1]

	plt.fill_between(X_plt_trns[:,0],mid_pnt[1],y_trns ,transform=tr,alpha=alpha,color=colors_lst[1],linewidth=0.0)
	
	############
	## 23
	## Calcular KDE
	#*X = np.array(pry23_1d)[:,np.newaxis]
	#*X_plot = np.linspace(-1, 1, 100)[:,np.newaxis]
	#*kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
	#*log_dens = kde.score_samples(X_plot)
	if not precomp:
		X = np.array(pry23_1d)
		X_plot = np.linspace(-1, 1, 100)
		kde = stats.gaussian_kde(X)
		y_kde = kde(X_plot)
	else:
		X_plot = fig_data_old["X_plot"]
		y_kde = fig_data_old["pry23_kde"]
	fig_data["pry23_kde"] = y_kde

	## I re-escale the KDE plot to avoid having parts of the plot outside the figure's limits
	y_kde = 0.5*y_kde/np.max(y_kde)

	## pintar KDE sobre plot anterior

	mid_pnt = (p2+p3)/2.0
	p23_l = mid_pnt[0] - np.linalg.norm(p2-p3)/2.0
	p23_r = mid_pnt[0] + np.linalg.norm(p2-p3)/2.0

	## I have to invert the axis of the x values because, when I flip 
	## the KDE plot, pole 3 is on the bottom and pole 2 is on the top
	## and the 2d polarization projection is always computed 
	## with -1 in the smaller pole and 1 in the larger pole, resulting
	## in that the original "left" side of the KDE is on the bottom 
	## and the original "right" side, on the top.
	X_plt_trns = np.linspace(p23_l,p23_r,100)[::-1,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],90)+ax.transData

	y_trns = y_kde

	plt.fill_between(X_plt_trns[:,0],0, y_trns,transform=tr,alpha=alpha,color=colors_lst[2],linewidth=0.0)
	return fig_data

def plot_hist_on_triangle_vis(
	pry12_1d, pry13_1d, pry23_1d,
	ax,
	colors_lst = ["C2","C0","C1"],
	alpha = 0.5,
	bins = 60,
	offset = 0.025, ## Para separar un poco los histogramas de los bordes del triangulo.
	):
	"""
	Created: 2020-03-03
	Modified: 2020-03-03
	"""
	plt.sca(ax)
	vs = get_simplex_vertex(2)
	p1 = vs[0]
	p2 = vs[1]
	p3 = vs[2]
	##############
	## 12
	bins_edges = np.linspace(-1,1,bins+1)
	## Sometimes I have values like 1.00000004 that fall outside of the interval. 
	## This fixes that issue:
	pry12_1d = np.clip(pry12_1d,-1,1) 
	y, _ = np.histogram(pry12_1d,bins=bins_edges,density=True)
	y = 0.5*y/np.max(y)
	
	## pintar histograma sobre plot anterior   
	mid_pnt = (p1+p2)/2.0
	l = mid_pnt[0] - np.linalg.norm(p1-p2)/2.0
	r = mid_pnt[0] + np.linalg.norm(p1-p2)/2.0

	## I have to invert the axis of the x values because in the 
	## triangle pole 1 is on the right and pole 2 on the left
	## and the 2d polarization projection is always computed 
	## with -1 in the smaller pole and 1 in the larger pole.
	x_plt = np.linspace(l,r,bins+1)
	x_plt = (x_plt[1:]+x_plt[:-1])/2.0
	## If I don't do this the first and last bars of the histogram are thinner
	## than the rest (they have half the width).
	x_plt_width = x_plt[1]-x_plt[0]
	x_plt[0] = x_plt[0] - x_plt_width
	x_plt[-1] = x_plt[-1] + x_plt_width
	## Final vector
	X_plt_trns = x_plt[::-1,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],-30)+ax.transData

	y_trns = y+mid_pnt[1]+offset

	plt.fill_between(X_plt_trns[:,0],mid_pnt[1]+offset,y_trns ,
		transform=tr,
		alpha=alpha,
		color=colors_lst[0],
		linewidth=0.0,
		step = "mid",
		clip_on = False,
		)

	#############
	## 13
	bins_edges = np.linspace(-1,1,bins+1)
	pry13_1d = np.clip(pry13_1d,-1,1)
	y, _ = np.histogram(pry13_1d,bins=bins_edges,density=True)
	y = 0.5*y/np.max(y)

	## pintar histograma sobre plot anterior
	mid_pnt = (p1+p3)/2.0
	l = mid_pnt[0] - np.linalg.norm(p1-p3)/2.0
	r = mid_pnt[0] + np.linalg.norm(p1-p3)/2.0
	x_plt = np.linspace(l,r,bins+1)
	x_plt = (x_plt[1:]+x_plt[:-1])/2.0
	## If I don't do this the first and last bars of the histogram are thinner
	## than the rest (they have half the width).
	x_plt_width = x_plt[1]-x_plt[0]
	x_plt[0] = x_plt[0] - x_plt_width
	x_plt[-1] = x_plt[-1] + x_plt_width
	## Final vector
	X_plt_trns = x_plt[:,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],-150)+ax.transData

	#*y_trns = np.exp(log_dens)-min(np.exp(log_dens))+mid_pnt[1]
	y_trns = y+mid_pnt[1]+offset

	plt.fill_between(X_plt_trns[:,0],mid_pnt[1]+offset,y_trns,
		transform=tr,
		alpha=alpha,
		color=colors_lst[1],
		linewidth=0.0,
		step = "mid",
		clip_on = False,
		)
	
	############
	## 23
	bins_edges = np.linspace(-1,1,bins+1)
	pry23_1d = np.clip(pry23_1d,-1,1)
	y, _ = np.histogram(pry23_1d,bins=bins_edges,density=True)
	y = 0.5*y/np.max(y)

	## pintar histograma sobre plot anterior
	mid_pnt = (p2+p3)/2.0
	p23_l = mid_pnt[0] - np.linalg.norm(p2-p3)/2.0
	p23_r = mid_pnt[0] + np.linalg.norm(p2-p3)/2.0

	## I have to invert the axis of the x values because, when I flip 
	## the KDE plot, pole 3 is on the bottom and pole 2 is on the top
	## and the 2d polarization projection is always computed 
	## with -1 in the smaller pole and 1 in the larger pole, resulting
	## in that the original "left" side of the KDE is on the bottom 
	## and the original "right" side, on the top.
	x_plt = np.linspace(p23_l,p23_r,bins+1)
	x_plt = (x_plt[1:]+x_plt[:-1])/2.0
	## If I don't do this the first and last bars of the histogram are thinner
	## than the rest (they have half the width).
	x_plt_width = x_plt[1]-x_plt[0]
	x_plt[0] = x_plt[0] - x_plt_width
	x_plt[-1] = x_plt[-1] + x_plt_width
	## Final vector
	X_plt_trns = x_plt[::-1,np.newaxis]

	tr = transforms.Affine2D().rotate_deg_around(mid_pnt[0],mid_pnt[1],90)+ax.transData

	y_trns = y+offset

	plt.fill_between(X_plt_trns[:,0],0+offset, y_trns,
		transform=tr,
		alpha=alpha,
		color=colors_lst[2],
		linewidth=0.0,
		step = "mid",
		clip_on = False,
		)
	return ax

def vis_triangle_1d_proy_helper(pnt_lst):
	"""
	Created: 2019-09-22
	Modified: 2019-09-22
	Get the projection of the points inside the triangle to the sides of the
	triangle.
	"""
	vs = get_simplex_vertex(2)
	## Proyect
	pry12 = proyect_nd_to_md(
		pnt_lst,
		[vs[2]],
		[vs[0],vs[1]])
	pry13 = proyect_nd_to_md(
		pnt_lst,
		[vs[1]],
		[vs[0],vs[2]])
	pry23 = proyect_nd_to_md(
		pnt_lst,
		[vs[0]],
		[vs[1],vs[2]])
	## New coord system in 1d
	pry12_1d = new_1d_coord(
		pry12,
		[vs[0],vs[1]]
		)
	pry13_1d = new_1d_coord(
		pry13,
		[vs[0],vs[2]]
		)
	pry23_1d = new_1d_coord(
		pry23,
		[vs[1],vs[2]]
		)
	return pry12_1d, pry13_1d, pry23_1d

## Use faster numerical computations for projecting
def vis_triangle_1d_proy_helper_num(
	pnt_lst,
	project_close_only=False,
	proj_type="orthogonal"
	):
	"""
	Created: 2019-09-22
	Modified: 2020-03-04
	Get the projection of the points inside the triangle to the sides of the
	triangle.
	Modified to introduce the parallelized version.
	"""
	if project_close_only:
		_, pole_assignment = assign_to_closest_pole(pnt_lst)
	vs = get_simplex_vertex(2)
	## Proyect
	if proj_type == "from_pole":
		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[0,1])
		else:
			pnt_lst_proj = pnt_lst
		pry12 = proyect_nd_to_md_num_PAR(
			pnt_lst_proj,
			[vs[2]],
			[vs[0],vs[1]])

		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[0,2])
		else:
			pnt_lst_proj = pnt_lst

		pry13 = proyect_nd_to_md_num_PAR(
			pnt_lst_proj,
			[vs[1]],
			[vs[0],vs[2]])

		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[1,2])
		else:
			pnt_lst_proj = pnt_lst

		pry23 = proyect_nd_to_md_num_PAR(
			pnt_lst_proj,
			[vs[0]],
			[vs[1],vs[2]])
		## New coord system in 1d
		pry12_1d = new_1d_coord(
			pry12,
			[vs[0],vs[1]]
			)
		pry13_1d = new_1d_coord(
			pry13,
			[vs[0],vs[2]]
			)
		pry23_1d = new_1d_coord(
			pry23,
			[vs[1],vs[2]]
			)
	elif proj_type == "orthogonal":
		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[0,1])
		else:
			pnt_lst_proj = pnt_lst

		pry12_1d = axis_1d_proj_ortog(
			pnt_lst_proj,
			[vs[0]],[vs[1]]
			)

		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[0,2])
		else:
			pnt_lst_proj = pnt_lst

		pry13_1d = axis_1d_proj_ortog(
			pnt_lst_proj,
			[vs[0]],[vs[2]]
			)

		if project_close_only:
			pnt_lst_proj = filter_data_closer_poles(
				pnt_lst,
				pole_assignment,
				[1,2])
		else:
			pnt_lst_proj = pnt_lst

		pry23_1d = axis_1d_proj_ortog(
			pnt_lst_proj,
			[vs[1]],[vs[2]]
			)
	else:
		raise Exception(f"{proj_type} is not a valid value for proj_type")
	return pry12_1d, pry13_1d, pry23_1d

def assign_to_closest_pole(data):
	"""
	Created: 2020-03-20
	Modified: 2020-03-20
	Assigns each data point to its closest pole.
	"""
	n_points = len(data)
	dim = len(data[0])
	n_poles = dim + 1
	poles_coords = get_simplex_vertex(dim)
	dists = np.zeros((n_points,n_poles))
	for i, pole in enumerate(poles_coords):
		disti = np.linalg.norm(data - pole,axis=1)
		dists[:,i] = disti
	poles_assingment = np.argmin(dists,axis=1)
	return dists, poles_assingment

def filter_data_closer_poles(
	data,
	pole_assignment,
	poles_idx):
	"""
	Created: 2020-03-20
	Modified: 2020-03-20
	"""
	msk = np.zeros(len(data)) + False
	for idx in poles_idx:
		msk = np.logical_or(msk, pole_assignment==idx)
	return data[msk]

def get_alignment_AvB_axis_to_PCA(
	poles_coord,
	poles_A_idx,
	poles_B_idx,
	PCA):
	"""
	Created: 2020-03-23
	Modified: 2020-03-23
	"""
	n_dim = PCA.shape[1]
	assert len(poles_coord[0]) == n_dim
	assert len(poles_coord) >= max(set(poles_A_idx).union(set(poles_B_idx)))
	poles_A_coord = [poles_coord[pi] for pi in poles_A_idx]
	poles_B_coord = [poles_coord[pi] for pi in poles_B_idx]
	bcA = np.mean(poles_A_coord,axis=0)
	bcB = np.mean(poles_B_coord,axis=0)
	vecAB = bcB - bcA
	vecAB = vecAB/np.linalg.norm(vecAB)
	assert np.isclose(np.linalg.norm(vecAB),1.0)
	for i in range(n_dim):
		assert np.isclose(np.linalg.norm(PCA[:,i]),1.0)
	PCA_projs = [np.dot(vecAB,PCA[:,i]) for i in range(n_dim)]
	## I dont mind the sign of the projection, only its magnitude
	PCA_projs = [np.abs(i) for i in PCA_projs]
	## Angles in degrees, not radians
	PCA_angles = [np.arccos(i)*180.0/np.pi for i in PCA_projs]
	return PCA_projs, PCA_angles

##############################################################################
## RADAR plot
##############################################################################

def vis_radar(
    values,
    sizes = None,
    pole_lbls = None,
    yticks = [.2,.4,.6,.8],
    pls_font=24,
    tcks_font=14,
    color="C0",
    size_min=50,
    size_max=1000,
    ax=None
    ):
    if not ax:
        fig = plt.figure(figsize=(5,5))
    else:
    	fig = None
    n_poles = len(values)

    # number of variable
    if pole_lbls:
        categories= pole_lbls
    else:
        categories= list(range(n_poles))

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = copy.deepcopy(values)
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [i / float(n_poles) * 2 * np.pi for i in range(n_poles)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)    
    ax.grid(linewidth=1,ls=":")

    # Draw one axe per variable + add labels yet
    plt.xticks(angles[:-1], categories, color='k',fontsize=pls_font)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # Draw 4 ticks
    plt.yticks(yticks,yticks,fontsize=tcks_font,color="0.6")
    plt.ylim(0,1)

    # Plot data
    if sizes is None:
        ax.plot(angles, values, "o-",linewidth=3,color=color,zorder=0)
    else:
        rescale = rescale_fun_gen(sizes,size_min,size_max)
        sizes = [rescale(i) for i in sizes]
        ax.plot(angles,values,"-",linewidth=3,color=color,zorder=0)
        ax.scatter(angles,values,s=sizes,color=color,zorder=0)

    # Fill area
    ax.fill(angles, values, alpha=0.2,color=color,zorder=0)
    plt.tight_layout()
    if fig:
    	return fig, ax
    else:
    	return ax

def rescale_fun_gen(
	vals,
	a = 50,
	b = 1000,
	):
	max_v = max(vals)
	min_v = min(vals)
	rescale = lambda x: ((x-min_v)/(1.0*max_v-min_v))*(b-a) + a
	return rescale

def rescale_fun_gen_v2(
	vals,
	av_target=15
	):
	"""
	Rescales sizes so that the average is av_target.
	"""
	av_orig = np.mean(vals)
	av_ratio = av_target/(1.0*av_orig)
	rescale = lambda x: x*av_ratio
	assert np.isclose(np.mean([rescale(v) for v in vals]), av_target)
	return rescale

def vis_averages_radar(data):
	"""
	Created: 2020-04-16
	Modified: 2020-04-16
	"""
	data = np.array(data)
	dim = len(data[0])
	n_poles = dim+1
	_, pole_assignment = assign_to_closest_pole(data)
	av_lst = []
	size_lst = []
	for pi in range(n_poles):
		data_filter = filter_data_closer_poles(
					data,
					pole_assignment,
					[pi])
		av = np.mean(data_filter,axis=0)
		assert len(av) == dim
		av_lst.append(av)
		size_lst.append(len(data_filter))
	values = list(map(np.linalg.norm,av_lst))
	fig, ax = vis_radar(
	    values,
	    sizes=size_lst,
	    pole_lbls = None,
	    yticks = [.2,.4,.6,.8],
	    ax=None
	    )
	return fig, ax

def vis_averages_DFB(data):
	"""
	Created: 2020-04-16
	Modified: 2020-04-16
	same as vis_averages_radar but using the function
	vis_dist_from_barycenter
	"""
	data = np.array(data)
	dim = len(data[0])
	n_poles = dim+1
	_, pole_assignment = assign_to_closest_pole(data)
	av_lst = []
	size_lst = []
	for pi in range(n_poles):
		data_filter = filter_data_closer_poles(
					data,
					pole_assignment,
					[pi])
		av = np.mean(data_filter,axis=0)
		assert len(av) == dim
		av_lst.append(av)
		size_lst.append(len(data_filter))
	values = list(map(np.linalg.norm,av_lst))
	fig, ax = vis_dist_from_barycenter(
		values,
		sizes=size_lst
		)
	return fig, ax

def vis_dist_from_barycenter(
	values,
	sizes = None,
	ax = None,
	pole_lbls = None,
	size_ref= 15
	):
	"""
	Created: 2020-06-29
	Modified: 2020-06-29
	Like vis_radar but with a more easily interpretable classic layout
	and with a different strategy for sizing the bubbles.
	"""
	if ax is None:
		fig = plt.figure(figsize=(.35*9,.35*6))
		ax = plt.axes()
	else:
		fig = None
		plt.sca(ax)
	xtcks = list(range(len(values)))
	if pole_lbls is None:
		pole_lbls = xtcks
	if sizes is None:
		sizes = np.zeros(len(values)) + size_ref**2.0

	assert len(values) == len(sizes)
	rescale = rescale_fun_gen_v2(sizes,size_ref**2.0)
	sizes_resc = [rescale(i) for i in sizes]

	plt.stem(xtcks,values,
			 linefmt="k-",
			 markerfmt=".k",
			 basefmt ="none",
			 use_line_collection=True)
	plt.scatter(xtcks,values,
				s=sizes_resc,
				c=sizes,
				cmap="viridis",
			   edgecolor="k")
	plt.xticks(xtcks,xtcks)
	plt.ylim(-0.1,1.2)
	plt.ylabel("Distance")
	plt.xlabel("Poles")

	xlims = ax.get_xlim()
	plt.axhline(1,ls="--",lw=2,color="0.6",zorder=0)
	plt.axhline(0,ls="--",lw=2,color="0.6",zorder=0)

	plt.text(xlims[1]+0.05,0, "Neutral",
		 ha='left',
		 va='center',
		 color = "0.6"
		)
	plt.text(xlims[1]+0.05,1, "Pole",
		 ha='left',
		 va='center',
		 color = "0.6"
		)
	plt.tight_layout()
	if fig:
		return fig, ax
	else:
		return ax

##############################################################################
## PCA visualization
##############################################################################

def vis_2d_pca(
	pnt_lst,
	pca_to_plot = [0,1], ## Which PCA should be plotted
	dir_signs = [1,1], ## To change direction of axis if needed
	hist2d_bins = 200,
	grid_div = 100,
	nlevels = 20,
	pole_lbls = None,
	xlim = None,
	ylim = None
	):

	assert len(pca_to_plot) == 2
	assert len(dir_signs) == 2
	assert len(set(pca_to_plot)) == len(pca_to_plot)
	assert max(pca_to_plot) <= len(pnt_lst[0]) - 1

	## Compute PCA
	l, var_exp, U = my_pca(pnt_lst)

	## Get poles coordinates
	dim = len(pnt_lst[0])
	poles_coord = get_simplex_vertex(dim)

	## Compute center of mass (average)
	cm_coord = np.mean(pnt_lst,axis=0).reshape(1,dim)
	
	## Project data and poles to chosen directions
	data_prj = np.zeros((2,len(pnt_lst)))
	poles_prj = np.zeros((2,len(poles_coord)))
	cm_prj = np.zeros((2,1))

	for i,pca_i in enumerate(pca_to_plot):
	    pca_dir = dir_signs[i]*U[:,-(pca_i+1)]
	    pca_dir = pca_dir / np.linalg.norm(pca_dir)
	    data_prj[i,:] = proj_ortog(pnt_lst,[pca_dir],orig=None)[:,0]
	    poles_prj[i,:] = proj_ortog(poles_coord,[pca_dir],orig=None)[:,0]
	    cm_prj[i,:] = proj_ortog(cm_coord,[pca_dir],orig=None)[:,0]

	data = np.array(list(zip(data_prj[0,:],data_prj[1,:])))

	assert data.shape[1] == 2 ## Make sure that I have 2d data

	## Compute 2D histogram
	H,xedges,yedges = np.histogram2d(data[:,0],data[:,1],bins=hist2d_bins)
	xcenters = (xedges[1:]+xedges[:-1])/2.0
	ycenters = (yedges[1:]+yedges[:-1])/2.0

	H[H==0] = 1.0 ## To avoid problems with bins with zero counts when I show them in log scale

	## Figure

	## Setup layout
	fig = plt.figure(constrained_layout=True,figsize=(.6*8,.6*6))
	fig.patch.set_alpha(0.0)

	widths = [1, .3]
	heights = [.3, 1]
	spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
	                          height_ratios=heights)

	ax_main = fig.add_subplot(spec[1, 0])
	ax_main.set_facecolor("k")
	plt.sca(ax_main)

	myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]

	## Show histogram
	plt.imshow(
	    H.T,
	    origin='lower',
	    extent=myextent,
	    interpolation='nearest',
	    aspect='auto',
	    cmap="inferno",
	    norm=colors.LogNorm()
	)

	## Show contours
	xx,yy,f = compute_2d_kde_tri(data, grid_div, mask_not_tri=False, pnts_bynd=True)
	# levels = np.logspace(-10,np.log10(np.max(f)),nlevels) ## 2015
	# levels = np.logspace(-6,0.95*np.log10(np.max(f)),nlevels) ## 28A
	levels = np.logspace(0.3*np.log10(np.min(f)),np.log10(np.max(f))*0.95,nlevels)

	plt.contour(
	    xx, 
	    yy,
	    f,
	    levels=levels,
	    colors="w",
	    linewidths=0.25,
	    )

	# Plot Poles
	if pole_lbls is None:
		pole_lbls = range(dim+1)

	for i, prj_i in enumerate(poles_prj.T):
	    plt.plot(prj_i[0],prj_i[1],"ow",ms=5,mec="w",mfc="none")
	    ax_main.annotate(pole_lbls[i],(prj_i[0],prj_i[1]),
	                      ha="left",
	                      va="bottom",
	                      color="w",
	                      fontweight="bold",
	                      fontsize=10,
	#                       textcoords='offset points',
	                      zorder=20)
	    
	## Plot center
	plt.plot(0,0,"+w")

	## Plot CM
	plt.plot(cm_prj[0],cm_prj[1],"sw")

	## Axis labels
	plt.xlabel(f"PC {pca_to_plot[0]+1}")
	plt.ylabel(f"PC {pca_to_plot[1]+1}")

	# plt.axis("equal")

	if ylim:
		plt.ylim(ylim)
	if xlim:
		plt.xlim(xlim)

	## Projected 1D histogram into the first chosen PC
	ax_up = fig.add_subplot(spec[0, 0],sharex=ax_main)
	plt.setp(ax_up.get_xticklabels(), visible=False)
	plt.sca(ax_up)
	sns.distplot(data_prj[0,:],bins=71,norm_hist=True,color="dimgrey")

	for i, xi in enumerate(poles_prj[0,:]):
	    plt.axvline(xi,ls=":",lw=1.5,color="0.7",zorder=0)
	    ylims = ax_up.get_ylim()
	    plt.text(xi, ylims[1], i,
	         ha='center',
	         va='bottom',
	         color = "k"
	        )
	plt.locator_params(axis='y', nbins=4)
	plt.ylabel("PDF")

	## Projected 1D histogram into the second chosen PC
	ax_right = fig.add_subplot(spec[1, 1],sharey=ax_main)
	plt.setp(ax_right.get_yticklabels(), visible=False)
	plt.sca(ax_right)
	sns.distplot(data_prj[1,:],bins=71,vertical=True,norm_hist=True,color="dimgrey")

	for i, xi in enumerate(poles_prj[1,:]):
	    plt.axhline(xi,ls=":",lw=1.5,color="0.7",zorder=0)
	    xlims = ax_right.get_xlim()
	    plt.text(1.05*xlims[1],xi, i,
	         ha='left',
	         va='center',
	         color = "k"
	        )

	plt.locator_params(axis='x', nbins=4)
	plt.xlabel("PDF")

	return fig

##############################################################################
## Misc
##############################################################################

def get_GCC(G):
	gcc_nodes = max(nx.connected_components(G), key=len)
	G_gcc = G.subgraph(gcc_nodes)
	return copy.deepcopy(nx.Graph(G_gcc))

def node_size_from_vals(
	nodes,
	vals_dct,
	min_sz = 1,
	max_sz = 100,
	):
	max_v = max(vals_dct.values())
	min_v = min(vals_dct.values())
	rescale = lambda x: ((x-min_v)/(1.0*max_v-min_v))*(max_sz-min_sz) + min_sz
	sizes = []
	for n in nodes:
		v = vals_dct[n]
		sz = rescale(v)
		sizes.append(sz)
	return sizes

def add_colorbar_to_plot(
	ax,
	vmin,
	vmax,
	cmap="viridis"
	):
	plt.sca(ax)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
	sm._A = []
	plt.colorbar(sm)

def plot_nice_nw_vis(
	G,
	positions, 
	nodes,
	sizes,
	colors,
	show_lables=True,
	ax = None,
	cmap="Set1",
	vmax=None,
	vmin=None,
	lw = .5,
	edge_alpha=0.4,
	elite = None,
	elite_color = "r",
	elite_shape = "s"
	):
	if ax is None:
		ax = plt.axes()
		plt.axis('off')
	## Borde alrededor de nodos
	nx.draw_networkx_nodes(
		G,
		positions, 
		nodelist=nodes,
		node_color="k",
		node_size=sizes, 
		alpha=.8,
		linewidths=2.0)
	## Nodos
	if not elite:
		nx.draw_networkx_nodes(
			G,
			positions, 
			nodelist=nodes,
			node_color=colors,
			vmin=vmin,
			vmax=vmax,
			node_size=sizes, 
			cmap=cmap,
			alpha=1.0,
			linewidths=1.0)
	## Elite nodes
	else:
		nodes_not_elite = [n for n in nodes if n not in elite]
		colors_not_elite = np.zeros(len(nodes_not_elite))
		sizes_not_elite = np.zeros(len(nodes_not_elite))
		for i, ni in enumerate(nodes):
			for j, nj in enumerate(nodes_not_elite):
				if ni == nj:
					colors_not_elite[j] = colors[i]
					sizes_not_elite[j] = sizes[i]
					break
		nx.draw_networkx_nodes(
			G,
			positions, 
			nodelist=nodes_not_elite,
			node_color=colors_not_elite,
			vmin=vmin,
			vmax=vmax,
			node_size=sizes_not_elite, 
			cmap=cmap,
			alpha=1.0,
			linewidths=1.0)
		elite_sizes = np.zeros(len(elite))
		elite_color = np.zeros(len(elite))
		for i, ni in enumerate(nodes):
			for j, nj in enumerate(elite):
				if nj == ni:
					elite_sizes[j] = sizes[i]
					elite_color[j] = colors[i]
					break
		## Borde alrededor de nodos elite
		nx.draw_networkx_nodes(
			G,
			positions, 
			nodelist=elite,
			node_color="w",
			node_size=elite_sizes, 
			alpha=1.0,
			linewidths=0.0)
		nx.draw_networkx_nodes(
			G,
			positions,
			node_shape = elite_shape,
			nodelist=elite,
			node_color=elite_color,
			node_size=elite_sizes, 
			vmin=vmin,
			vmax=vmax,
			cmap=cmap,
			alpha=1.0) 
	## Etiquetas
	if show_lables:
		nx.draw_networkx_labels(
			G, 
			positions, 
			labels=None, 
			font_size=1.5, 
			font_color='k', 
			font_family='sans-serif', 
			font_weight='bold', 
			alpha=1.0, 
			)
	## Enlaces
	nx.draw_networkx_edges(
		G, 
		positions, 
		nodelist=nodes,
		edge_color="k",
		width = lw,
		alpha=edge_alpha)
	return ax

def draw_pie(dist,
			 xpos, 
			 ypos, 
			 size, 
			 ax=None):
	"""
	https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
	"""
	if ax is None:
		fig, ax = plt.subplots(figsize=(10,8))

	# for incremental pie slices
	cumsum = np.cumsum(dist)
	cumsum = cumsum/ cumsum[-1]
	pie = [0] + cumsum.tolist()

	for r1, r2 in zip(pie[:-1], pie[1:]):
		angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
		x = [0] + np.cos(angles).tolist()
		y = [0] + np.sin(angles).tolist()

		xy = np.column_stack([x, y])

		ax.scatter([xpos], [ypos], marker=xy, s=size)

	return ax

def my_pca(data):
	"""
	Created: 2020-03-20
	Modified: 2020-03-20
	PCA is just diagonalizing the covariance matrix. 
	"""
	assert type(data) == np.ndarray
	## I have to transpose because fucking numpy assumes that each row is 
	## a dimension.
	cov = np.cov(data.T)
	l, U = np.linalg.eigh(cov)
	var_exp = l/np.sum(l)
	return l, var_exp, U