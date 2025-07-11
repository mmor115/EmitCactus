if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, inv, sympify
    from sympy import Expr, cos, sin
    from EmitCactus.emit.tree import Centering

    set_dimension(4)

    # Create a set of grid functions
    gf = ThornDef("TestKerr", "Kerr")

    spin = False
    a: Expr
    if spin:
        # This is very slow
        a = gf.add_param("a", default=0.5, desc="The black hole spin")
    else:
        a = sympify(0)
    m = gf.add_param("m", default=0.5, desc="The black hole mass")
    t, r, th, phi = gf.mk_coords()

    sigma = r ** 2 + a ** 2 * cos(th) ** 2
    delta = r ** 2 - 2 * m * r + a ** 2

    gtt = -(1 - 2 * m * r / sigma)
    grr = sigma / delta
    gqq = sigma
    gpp = (r ** 2 + a ** 2 + (2 * m * r ** 2 * a ** 2 / sigma) * sin(th) ** 2) * sin(th) ** 2
    gtp = -4 * m * r * a * sin(th) ** 2 / sigma

    Z = sympify(0)
    gmat = mkMatrix([
        [gtt, Z, Z, gtp],
        [Z, grr, Z, Z],
        [Z, Z, gqq, Z],
        [gtp, Z, Z, gpp]])

    # Declare gfs
    g = gf.decl("g", [li, lj], symmetries=[(li, lj)], centering=Centering.VVC, substitution_rule=gmat)
    G = gf.decl("Affine", [ua, lb, lc], symmetries=[(lb, lc)], centering=Centering.VVC, substitution_rule=None)
    Ric = gf.decl("Ric", [la, lb], symmetries=[(la, lb)], centering=Centering.VVC, substitution_rule=None)

    imat = inv(gmat)
    gf.add_substitution_rule(g[ua, ub], imat)

    gf.add_substitution_rule(G[la, lb, lc], (D(g[la, lb], lc) + D(g[la, lc], lb) - D(g[lb, lc], la)) / 2)
    gf.add_substitution_rule(G[ud, lb, lc], g[ud, ua] * G[la, lb, lc])
    gf.add_substitution_rule(Ric[li, lj],
                             D(G[ua, li, lj], la) - D(G[ua, la, li], lj) +
                             G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

    for i in range(4):
        for j in range(i + 1, 4):
            ixi = [l0, l1, l2, l3][i]
            ixj = [l0, l1, l2, l3][j]
            print("Checking:", Ric[ixi, ixj])
            assert gf.do_subs(Ric[ixi, ixj]) == sympify(0)
