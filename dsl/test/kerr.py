from dsl.use_indices import *
from dsl.sympywrap import mkMatrix, do_sympify, do_inv, do_diff, do_sympify
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin, sympify
from emit.code.code_tree import Centering
from generators.wizards import CppCarpetXWizard

set_dimension(4)

# Create a set of grid functions
gf = ThornDef("TestKerr", "Kerr")

# Declare gfs
g = gf.decl("g", [li, lj], Centering.VVC)
G = gf.decl("Affine", [ua, lb, lc], Centering.VVC)
Ric = gf.decl("Ric", [la, lb], Centering.VVC)

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(Ric[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)

a = gf.add_param("a", default=0.5, desc="The black hole spin")
#a = do_sympify(0)
m = gf.add_param("m", default=0.5, desc="The black hole mass")
t, r, th, phi = gf.mk_coords()

sigma = r**2 + a**2*cos(th)**2
delta = r**2 - 2*m*r + a**2

gtt = -(1-2*m*r/sigma)
grr = sigma/delta
gqq = sigma
gpp = (r**2 + a**2 + (2*m*r**2**a**2/sigma)*sin(th)**2)*sin(th)**2
gtp = -4*m*r*a*sin(th)**2/sigma

gmat = mkMatrix([
[gtt,   0,   0, gtp],
[  0, grr,   0,   0],
[  0,   0, gqq,   0],
[gtp,   0,   0, gpp]])

gf.mk_subst(g[la,lb], gmat)
imat = do_inv(gmat)
print("imat:",imat)
gf.mk_subst(g[ua, ub], imat)
# s=a, j=b, k=c
gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])

#pairs = [(l0,u0),(l1,u1),(l2,u2),(l3,u3)]
#result = dict()
#for a in pairs:
#    for b in pairs:
#        for c in pairs:
#            for d in pairs:
#                term0 = g[d[1],a[1]]*(div(g[a[0], b[0]], c[0]) + div(g[a[0],c[0]],b[0]) - div(g[b[0],c[0]],a[0]))/2
#                tup = (d[1],b[0],c[0])
#                if tup not in result:
#                    result[tup] = do_sympify(0)
#                result[tup] += do_sympify(gf.do_subs(term0))
#for tup in result:
#    if result[tup] != do_sympify(0):
#        xx = gf.do_subs(G[tup[0], tup[1], tup[2]])
#        print(tup,"=>",result[tup], do_sympify(result[tup]-xx))

#exit(0)
gf.mk_subst(Ric[li, lj],
             div(G[ua, li, lj], la) - div(G[ua, la, li], lj) + 
             G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])
exit(0)

def tst(x):
    y = gf.do_subs(x)
    print(x)
    print("  ->",y)
    return y

tst(G[u1,l0,l0] - g[u1,u1]*G[l1,l0,l0])
tst(G[l0,l0,l1] + div(g[l0,l0],l1))
tst(G[u0,l0,l1] - g[u0,u0]*G[l0,l0,l1])
tst(G[u1,l1,l1] - g[u1,u1]*G[l1,l1,l1])
tst(G[l1,l1,l1] - div(g[l1,l1],l1))
tst(G[u2,l1,l2] - g[u2,u2]*G[l2,l1,l2])
tst(G[u2,l1,l2] - 1/r)
tst(G[u3,l1,l3] - g[u3,u3]*G[l3,l1,l3])
tst(G[u3,l1,l3] - 1/r)
tst(G[u0,l0,l1] + 1/gtt*do_diff(gtt,r))
# Evaluating G[ua,la,li] G[ub,l0,l0]
inp1a = do_sympify(0)
for a in [u0,u1,u2,u3]:
    paira = lookup_pair[a]
    inp1a += gf.do_subs(div(G[a, l0, l0], paira))
inp1b = gf.do_subs(div(G[u1,l0,l0],l1))
print("inp1:",do_sympify(inp1a - inp1b))

inp2a = do_sympify(0)
for a in [u0,u1,u2,u3]:
    paira = lookup_pair[a]
    inp2a += gf.do_subs(div(G[a,paira,l0],l0))
print("inp2:",do_sympify(inp2a))

inp3a = do_sympify(0)
for a in [u0,u1,u2,u3]:
    paira = lookup_pair[a]
    for b in [u0,u1,u2,u3]:
        pairb = lookup_pair[b]
        inp3a += gf.do_subs(G[a,paira,pairb]*G[b,l0,l0])
print("inp3a:", inp3a)

inp4a = do_sympify(0)
for a in [u0,u1,u2,u3]:
    paira = lookup_pair[a]
    for b in [u0,u1,u2,u3]:
        pairb = lookup_pair[b]
        inp4a += gf.do_subs(G[a,l0,pairb]*G[b,paira,l0])
print("inp4a:", inp4a)

print("zero:", do_sympify(inp1a-inp2a+inp3a-inp4a))
