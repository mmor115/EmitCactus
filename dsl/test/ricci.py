from dsl.use_indices import *
from dsl.sympywrap import mkMatrix, do_inv, do_sympify
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin
from emit.code.code_tree import Centering
from generators.wizards import CppCarpetXWizard

# Create a set of grid functions
gf = ThornDef("TestRicci", "Ricci")

# Declare gfs
g = gf.decl("g", [li, lj], Centering.VVC)
x,y,z = gf.mk_coords()
G = gf.decl("Affine", [ua, lb, lc], Centering.VVC)
Ric = gf.decl("Ric", [la, lb], Centering.VVC)

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)
gf.add_sym(Ric[la,lb], la, lb)

diagonal = True
if diagonal:
    gDD00 = gf.decl("gDD00", [], Centering.VVC)
    gf.mk_subst(g[la,lb],mkMatrix(
    [[1+y**2+z**2, 0, 0],
     [0    ,1+x**2+z**2, 0],
     [0    ,0, 1+x**2+y**2]]))
else:
    gf.mk_subst(g[la, lb])

gmat = gf.get_matrix(g[la,lb])
imat = do_inv(gmat)
opt1 = False
if opt1:
    gf.mk_subst(g[ua,ub]) # g[u0,u0] -> gUU00
else:
    gf.mk_subst(g[ua, ub], imat)
#print(gf.find_symmetries(div(g[la,lb], lc)))
#print(gf.find_symmetries(div(g[la,lb], lc, ld)))
#exit(0)
#gf.mk_subst(iter3[lc, la, lb], alt=div(g[la, lb], lc))
gf.mk_subst(div(g[la, lb], lc)) # div(g[l0,l1],l2) -> gDD01_dD2
gf.mk_subst(div(g[ua, ub], lc))
opt2 = False
if opt2:
    gf.mk_subst(G[la, lb, lc])
    gf.mk_subst(G[ua, lb, lc])
else:
    gf.mk_subst(G[la, lb, lc], div(g[lb, lc], la) + div(g[la, lc], lb) - div(g[la, lb], lc))
    gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])
    
gf.mk_subst(Ric[la, lb])
#gf.mk_subst(div(G[ud, la, lb], lc))

fun = gf.create_function("setGL", ScheduleBin.Analysis)

if opt1:
    fun.add_eqn(g[ua, ub], imat)
if opt2:
    fun.add_eqn(G[la, lb, lc], div(g[lb, lc], la) + div(g[la, lc], lb) - div(g[la, lb], lc))
    fun.add_eqn(G[ua, lb, lc], g[ua, ud] * G[ld, lb, lc])

#gf.mk_subst(Ric[li, lj],
fun.add_eqn(Ric[li, lj],
             div(G[ua, li, lj], la) - div(G[ua, la, li], lj) + 
             G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

# Ensure the equations make sense
fun.bake()

CppCarpetXWizard(gf).generate_thorn()
