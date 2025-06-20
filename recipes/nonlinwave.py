if __name__ == "__main__":
    """
    The waveequation! It can't be solved too many times.
    """

    from EmitCactus import *
    from sympy import Expr, Idx, sin, Indexed
    import nrpy.helpers.conditional_file_updater as cfu
    from math import pi

    # If we change our configuration, this will show us diffs of the
    # new output and the old.
    cfu.verbose = True


    def flat_metric(_: Indexed, i: int, j: int) -> Expr:
        if i == 2 or j == 2:
            return sympify(0)
        elif i == j:
            return sympify(1)
        else:
            return sympify(0)


    # Create a set of grid functions
    gf = ThornDef("TestWave", "WaveEqn")

    # Use a NRPy calculated stencil instead
    # of simply calling functions such as divx()
    gf.set_derivative_stencil(3)

    # Declare gfs
    v_t = gf.decl("v_t", [], centering=Centering.VVC)
    v = gf.decl("v", [], centering=Centering.VVC, rhs=v_t)
    u_t = gf.decl("u_t", [], centering=Centering.VVC)
    u = gf.decl("u", [], centering=Centering.VVC, rhs=u_t)

    # Declare the metric
    g = gf.decl("g", [li, lj], symmetries=[(li, lj)], substitution_rule=flat_metric)

    # Declare params
    spd = gf.add_param("spd", default=1.0, desc="The wave speed")
    kx = gf.add_param("kx", default=pi / 20, desc="The wave number in the x-direction")
    ky = gf.add_param("ky", default=pi / 20, desc="The wave number in the y-direction")

    # Fill in values
    gf.add_substitution_rule(g[ui, uj], flat_metric)

    x, y, z = gf.mk_coords()

    # Add the equations we want to evolve.
    fun = gf.create_function("newwave_evo", ScheduleBin.Evolve)
    fun.add_eqn(v_t, u)
    fun.add_eqn(u_t, spd ** 2 * g[ui, uj] * D(v, li, lj) + g[ui,uj]*D(v,li)*D(v, lj))
    print('*** ThornFunction wave_evo:')
    fun.bake()

    # Dump
    fun.dump()

    # Show tensortypes
    fun.show_tensortypes()

    # Again for wave_init
    fun = gf.create_function("newwave_init", ScheduleBin.Init)
    fun.add_eqn(v, sin(kx * x) * sin(ky * y))
    fun.add_eqn(u, sympify(0))  # kx**2 * ky**2 * sin(kx * x) * sin(ky * y))
    print('*** ThornFunction wave_init:')
    fun.bake()
    fun.dump()
    fun.show_tensortypes()

    fun = gf.create_function("refine", ScheduleBin.EstimateError)
    regrid_error = gf.decl("regrid_error", [], centering=Centering.CCC, from_thorn='CarpetX')
    #fun.add_eqn(regrid_error, 2*v*v)
    fun.add_eqn(regrid_error, 9/((x-20)**2 + (y-20)**2))
    fun.bake(do_cse=False)

    CppCarpetXWizard(gf).generate_thorn()
