import typing
from typing import Optional, List, Set, Tuple, Collection

import sympy as sy
from typing_extensions import Unpack

from EmitCactus.dsl.carpetx import ExplicitSyncBatch
from EmitCactus.dsl.eqnlist import TemporaryReplacement
from EmitCactus.dsl.use_indices import ThornDef, ThornFunction, ScheduleBin, ScheduleTarget
from EmitCactus.emit.ccl.interface.interface_tree import InterfaceRoot, HeaderSection, IncludeSection, FunctionSection, \
    VariableSection
from EmitCactus.emit.ccl.param.param_tree import ParamRoot, Param, ParamAccess, ParamType, ParamRange, \
    KeywordParamRange, StringParamRange, IntParamRange, IntParamDescWildcard, IntParamDescRange, IntParamOpenLowerBound, \
    IntParamOpenUpperBound, RealParamRange, RealParamDescWildcard, RealParamDescRange, RealParamOpenLowerBound, \
    RealParamOpenUpperBound
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleRoot, StorageLine, ScheduleBlock, StorageDecl, Intent, \
    GroupOrFunction, AtOrIn, StorageSection, ScheduleSection, IntentRegion
from EmitCactus.emit.code.code_tree import CodeRoot, CodeElem, IncludeDirective, UsingNamespace, Using, \
    ConstConstructDecl, IdExpr, VerbatimExpr, ConstAssignDecl, BinOpExpr, BinOp, FloatLiteralExpr, SympyExpr, \
    ThornFunctionDecl, DeclareCarpetXArgs, DeclareCarpetParams, UsingAlias, ConstExprAssignDecl, CarpetXGridLoopCall, \
    CarpetXGridLoopLambda
from EmitCactus.emit.tree import String, Identifier, Bool, Integer, Float, Language, Verbatim, Centering
from EmitCactus.generators.cactus_generator import CactusGenerator, CactusGeneratorOptions, InteriorSyncMode
from EmitCactus.generators.generator_exception import GeneratorException
from EmitCactus.util import OrderedSet


class CppCarpetXGeneratorOptions(CactusGeneratorOptions, total=False):
    explicit_syncs: Collection[ExplicitSyncBatch]


class CppCarpetXGenerator(CactusGenerator):
    boilerplate_includes: List[Identifier] = [Identifier(s) for s in
                                              ["cctk.h", "cctk_Arguments.h", "cctk_Parameters.h",
                                               "loop_device.hxx", "simd.hxx", "defs.hxx", "vect.hxx",
                                               "cmath", "tuple"]]
    boilerplate_namespace_usings: List[Identifier] = [Identifier(s) for s in ["Arith", "Loop"]]
    boilerplate_usings: List[Identifier] = [Identifier(s) for s in ["std::cbrt", "std::fmax", "std::fmin", "std::sqrt"]]

    # TODO: We want to be able to
    #  specify a header file with these
    #  or alternate defs.
    boilerplate_setup: str = "#define CARPETX_GF3D5"
    boilerplate_div_macros: str = """
        #define access(GF) (GF(p.mask, GF ## _layout, p.I))
        #define store(GF, VAL) (GF.store(p.mask, GF ## _layout, p.I, VAL))
        #define stencil(GF, IX, IY, IZ) (GF(p.mask, GF ## _layout, p.I + IX*p.DI[0] + IY*p.DI[1] + IZ*p.DI[2]))
        #define CCTK_ASSERT(X) if(!(X)) { CCTK_Error(__LINE__, __FILE__, CCTK_THORNSTRING, "Assertion Failure: " #X); }
    """.strip().replace('    ', '')

    options: CppCarpetXGeneratorOptions

    def __init__(self, thorn_def: ThornDef, **options: Unpack[CppCarpetXGeneratorOptions]) -> None:
        super().__init__(thorn_def, options)

        unbaked_fns = {name for name, fn in thorn_def.thorn_functions.items() if not fn.been_baked}
        if len(unbaked_fns) > 0:
            raise GeneratorException(f"One or more functions have not been baked. Namely: {unbaked_fns}")

    def get_src_file_name(self, which_fn: str) -> str:
        assert which_fn in self.thorn_def.thorn_functions

        return f'{self.thorn_def.name}_{which_fn}.cpp'

    def get_sync_batch_fn_src_file_name(self, which: ExplicitSyncBatch) -> str:
        return f'{self.thorn_def.name}_{which.name}.cpp'

    def generate_makefile(self) -> str:
        srcs = [self.get_src_file_name(fn_name) for fn_name in OrderedSet(self.thorn_def.thorn_functions.keys())]

        for sync_batch in self.options.get('explicit_syncs', list()):
            srcs.append(self.get_sync_batch_fn_src_file_name(sync_batch))

        return f'SRCS = {" ".join(srcs)}\n\nSUBDIRS = '

    def generate_schedule_ccl(self) -> ScheduleRoot:
        storage_lines: list[StorageLine] = list()
        schedule_blocks: list[ScheduleBlock] = list()

        for group in self.variable_groups.keys():
            storage_lines.append(StorageLine([
                StorageDecl(
                    Identifier(group),
                    Integer(1)
                )
            ]))

        for fn_name, fn in self.thorn_def.thorn_functions.items():
            schedule_bin, at_or_in = self._resolve_schedule_target(fn.schedule_target)

            reads: list[Intent] = list()
            writes: list[Intent] = list()
            syncs: set[Identifier] = OrderedSet()

            for var, spec in fn.eqn_list.read_decls.items():
                if var in fn.eqn_list.inputs and (var_name := str(var)) not in self.vars_to_ignore:
                    qualified_var_name = self._get_qualified_var_name(var_name)

                    reads.append(Intent(
                        name=Identifier(qualified_var_name),
                        region=spec
                    ))

            for var, spec in fn.eqn_list.write_decls.items():
                if var in fn.eqn_list.outputs and (var_name := str(var)) not in self.vars_to_ignore:
                    qualified_var_name = self._get_qualified_var_name(var_name)
                    qualified_var_id = Identifier(qualified_var_name)

                    writes.append(Intent(
                        name=qualified_var_id,
                        region=spec
                    ))

                    if spec is IntentRegion.Interior and (
                            self.options['interior_sync_mode'] is InteriorSyncMode.Always
                            or (self.options['interior_sync_mode'] is not InteriorSyncMode.HandsOff
                                and var not in self.thorn_def.rhs.values())
                    ):
                        # todo: There's currently a bug s.t. single-variable groups are not reflected in var2base or groups.
                        # assert var_name in self.thorn_def.var2base
                        syncs.add(Identifier(self._get_qualified_group_name_from_var_name(var_name)))

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(fn_name),
                at_or_in=at_or_in,
                schedule_bin=schedule_bin,
                description=String(f'Function `{fn_name}` generated by EmitCactus.'),
                lang=Language.C,
                reads=reads,
                writes=writes,
                sync=list(syncs),
                before=[Identifier(s) for s in fn.schedule_before],
                after=[Identifier(s) for s in fn.schedule_after]
            ))

        if 'extra_schedule_blocks' in self.options:
            for block in self.options['extra_schedule_blocks']:
                schedule_blocks.append(block)

        if self.options['interior_sync_mode'] is InteriorSyncMode.MixedRhs:
            new_explicit_syncs: List[ExplicitSyncBatch] = list(self.options.get('explicit_syncs', list()))
            new_explicit_syncs.append(
                ExplicitSyncBatch(
                    vars=self.thorn_def.get_state(),
                    schedule_target=ScheduleBin.PostStep,
                    name="StateSync"
                )
            )
            self.options['explicit_syncs'] = new_explicit_syncs

        if (sync_batch_items := self.options.get('explicit_syncs', None)) is not None:
            for sync_batch in sync_batch_items:
                schedule_bin, at_or_in = self._resolve_schedule_target(sync_batch.schedule_target)
                var_names = [str(v) for v in sync_batch.vars]

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(sync_batch.name),
                    at_or_in=at_or_in,
                    schedule_bin=schedule_bin,
                    description=String(f'Empty function for explicitly SYNCing state variables. Generated by EmitCactus.'),
                    lang=Language.C,
                    sync=[Identifier(self._get_qualified_group_name_from_var_name(var_name)) for var_name in var_names],
                    before=[Identifier(s) for s in sync_batch.schedule_before],
                    after=[Identifier(s) for s in sync_batch.schedule_after]
                ))

        return ScheduleRoot(
            storage_section=StorageSection(storage_lines),
            schedule_section=ScheduleSection(schedule_blocks)
        )

    @staticmethod
    def _resolve_schedule_target(schedule_target: ScheduleTarget) -> tuple[Identifier, AtOrIn]:
        schedule_bin: Identifier
        at_or_in: AtOrIn

        if isinstance(schedule_target, ScheduleBlock):
            schedule_bin = schedule_target.name
            at_or_in = AtOrIn.In
        else:
            assert isinstance(schedule_target, ScheduleBin)
            at_or_in = AtOrIn.At if schedule_target.is_builtin else AtOrIn.In

            if schedule_target is ScheduleBin.Init:
                schedule_bin = Identifier('initial')
            elif schedule_target is ScheduleBin.Analysis:
                schedule_bin = Identifier('analysis')
            elif schedule_target is ScheduleBin.EstimateError:
                schedule_bin = Identifier('ODESolvers_EstimateError')
            elif schedule_target is ScheduleBin.Evolve:
                schedule_bin = Identifier('ODESolvers_RHS')
            elif schedule_target is ScheduleBin.DriverInit:
                schedule_bin = Identifier('ODESolvers_Initial')
            elif schedule_target is ScheduleBin.PostStep:
                schedule_bin = Identifier('ODESolvers_PostStep')
            else:
                raise NotImplementedError(f'Bad ScheduleBin enum member {schedule_target}')

        return schedule_bin, at_or_in

    def generate_interface_ccl(self) -> InterfaceRoot:
        inherits_from = {Identifier(inherited_thorn) for inherited_thorn in self.thorn_def.base2thorn.values()}

        # We always want to inherit from CarpetX even if no vars explicitly need it
        inherits_from.add(Identifier('Driver'))

        return InterfaceRoot(
            HeaderSection(
                implements=Identifier(self.thorn_def.name),
                inherits=[*inherits_from],
                friends=[]
            ),
            IncludeSection([]),
            FunctionSection([]),
            VariableSection(list(self.variable_groups.values()))
        )

    def generate_param_ccl(self) -> ParamRoot:
        params: list[Param] = list()

        for param_name, param_def in self.thorn_def.params.items():
            py_param_type: type = param_def.get_type()
            py_param_range = param_def.values
            py_param_default = param_def.default

            param_type: ParamType
            param_range: Optional[ParamRange]
            param_default: String | Integer | Float | Bool

            if py_param_type is set:
                param_type = ParamType.Keyword

                assert type(py_param_range) is set[str]
                param_range = KeywordParamRange([String(s) for s in py_param_range], String(''))

                assert type(py_param_default) is str
                param_default = String(py_param_default)
            elif py_param_type is str:
                param_type = ParamType.String

                if py_param_range is None:
                    param_range = StringParamRange([String('')], String(''))
                else:
                    assert type(py_param_range) is str
                    param_range = StringParamRange([String(py_param_range)], String(''))

                assert type(py_param_default) is str
                param_default = String(py_param_default)
            elif py_param_type is int:
                param_type = ParamType.Int

                if py_param_range is None:
                    param_range = IntParamRange(IntParamDescWildcard(), String(''))
                else:
                    assert type(py_param_range) is tuple[int, int]
                    lo_i, hi_i = typing.cast(tuple[int, int], py_param_range)
                    param_range = IntParamRange(IntParamDescRange(
                        IntParamOpenLowerBound(Integer(lo_i)),
                        IntParamOpenUpperBound(Integer(hi_i))
                    ), String(''))

                assert type(py_param_default) is int
                param_default = Integer(py_param_default)
            elif py_param_type is float:
                param_type = ParamType.Real

                if py_param_range is None:
                    param_range = RealParamRange(RealParamDescWildcard(), String(''))
                else:
                    assert type(py_param_range) is tuple[float, float]
                    lo_f, hi_f = typing.cast(tuple[float, float], py_param_range)
                    param_range = RealParamRange(RealParamDescRange(
                        RealParamOpenLowerBound(Float(lo_f)),
                        RealParamOpenUpperBound(Float(hi_f))
                    ), String(''))

                assert type(py_param_default) is float
                param_default = Float(py_param_default)
            elif py_param_type is bool:
                param_type = ParamType.Bool
                param_range = None

                assert type(py_param_default) is bool
                param_default = Bool(py_param_default)
            else:
                raise GeneratorException(f"Didn't expect parameter type {py_param_type}")

            assert param_type is not None
            assert param_range is not None or param_type is ParamType.Bool

            params.append(Param(
                param_access=ParamAccess.Restricted,
                param_type=param_type,
                param_name=Identifier(param_name),
                param_desc=String(param_def.desc),
                range_descriptions=[param_range] if param_range is not None else [],
                default_value=param_default
            ))

        return ParamRoot(params)

    def generate_function_code(self, which_fn: str) -> CodeRoot:
        nodes: list[CodeElem] = list()
        thorn_fn: ThornFunction = self.thorn_def.thorn_functions[which_fn]
        fn_name: str = thorn_fn.name

        assert thorn_fn.been_baked

        nodes.append(Verbatim(self.boilerplate_setup))

        # Includes, usings...
        for include in self.boilerplate_includes:
            nodes.append(IncludeDirective(include))

        # div{x,y,z} macros
        nodes.append(Verbatim(self.boilerplate_div_macros))

        for ns in self.boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(Using(self.boilerplate_usings))

        # Each variable needs to have a corresponding decl of the form
        # `const GF3D5layout ${VAR_NAME}_layout(${LAYOUT_NAME}_layout);`
        # for the div macros to work; layout here really means centering.

        layout_decls: list[CodeElem] = list()
        declared_layouts: set[Centering] = set()
        var_centerings: dict[str, Centering] = dict()

        used_var_names = [str(v) for v in thorn_fn.eqn_list.variables]

        # This loop builds the centering decls for each used var
        for var_name in self.var_names:

            # If a var is not used in this function, skip it
            if var_name not in used_var_names:
                continue

            var_centering: Optional[Centering]

            # Try looking up the var's centering directly...
            if (var_centering := self.thorn_def.centering.get(var_name, None)) is not None:
                pass
            # Otherwise, try looking it up by the var's base...
            elif (var_base := self.thorn_def.var2base.get(var_name, None)) is not None:
                var_centering = self.thorn_def.centering.get(var_base, None)

            # If this var doesn't have a defined centering, skip it.
            if var_centering is None:
                continue

            assert var_centering is not None

            var_centerings[var_name] = var_centering

            # Make sure the referenced layout has a preceding, corresponding decl of the form
            # `const GF3D5layout ${LAYOUT_NAME}_layout(cctkGH, {$I, $J, $K});`
            if var_centering not in declared_layouts:
                declared_layouts.add(var_centering)

                i, j, k = var_centering.int_repr
                centering_init_list = f'{{{i}, {j}, {k}}}'

                layout_decls.append(ConstConstructDecl(
                    Identifier('GF3D5layout'),
                    Identifier(f'{var_centering.string_repr}_layout'),
                    [IdExpr(Identifier('cctkGH')), VerbatimExpr(Verbatim(centering_init_list))]
                ))

            # Now build the var's centering decl.
            # layout_decls.append(ConstConstructDecl(
            #     Identifier('GF3D5layout'),
            #     Identifier(f'{var_name}_layout'),
            #     [IdExpr(Identifier(f'{var_centering.string_repr}_layout'))]
            # ))

            layout_decls.append(Verbatim(f'#define {var_name}_layout {var_centering.string_repr}_layout'))

        # Figure out which centering to pass to grid.loop_int_device<...>
        # All of this function's outputs need to have the same centering. If they do, use that centering.
        output_centerings = {var_centerings[str(var)] for var in thorn_fn.eqn_list.outputs if str(var) in self.var_names}

        if None in output_centerings or len(output_centerings) == 0:
            raise GeneratorException(f"All output vars must have a centering: {thorn_fn.name} {output_centerings}")

        if len(output_centerings) > 1:
            raise GeneratorException(f"Output vars have mixed centerings")

        # Got the centering.
        output_centering: Centering
        [output_centering] = output_centerings

        output_regions = {spec for var, spec in thorn_fn.eqn_list.write_decls.items() if str(var) in self.var_names}

        if None in output_regions or len(output_regions) == 0:
            raise GeneratorException(f"All output vars must have a write region.")

        if len(output_regions) > 1:
            raise GeneratorException(
                f"Output vars for '{which_fn}' have mixed write regions: {list(thorn_fn.eqn_list.write_decls.items())}"
            )

        output_region: IntentRegion
        [output_region] = output_regions

        input_var_strs = [str(i) for i in thorn_fn.eqn_list.inputs]

        # x, y, and z are special, but x is extra special
        xyz_decls = [
            ConstAssignDecl(Identifier('auto'), Identifier(s), IdExpr(Identifier(f'p.{s}'))) for s in ['y', 'z']
            if s in input_var_strs
        ]

        if 'x' in input_var_strs:
            xyz_decls = [
                ConstAssignDecl(
                    Identifier('vreal'),
                    Identifier('x'),
                    BinOpExpr(
                        IdExpr(Identifier('p.x')),
                        BinOp.Add,
                        BinOpExpr(
                            VerbatimExpr(Verbatim('Arith::iota<vreal>()')),
                            BinOp.Mul,
                            IdExpr(Identifier('p.dx'))
                        )
                    )
                ),
                ConstAssignDecl(
                    Identifier('vreal'),
                    Identifier('t'),
                    IdExpr(Identifier('cctk_time'))
                )
            ] + xyz_decls

        # DXI, DYI, DZI decls
        di_decls = [
            ConstAssignDecl(Identifier('auto'), Identifier(s), BinOpExpr(FloatLiteralExpr(1.0), BinOp.Div, VerbatimExpr(Verbatim(f'CCTK_DELTA_SPACE({n})'))))
            for n, s in enumerate(['DXI', 'DYI', 'DZI'])
        ]

        stencil_limit_checks = []
        stencil_limits = thorn_fn.eqn_list.stencil_limits()
        for i in range(3):
            if stencil_limits[i] != 0:
                stencil_limit_checks.append(VerbatimExpr(Verbatim(f'CCTK_ASSERT(cctk_nghostzones[{i}] >= {stencil_limits[i]});')))

        eqn_list = thorn_fn.eqn_list
        reassigned_lhses: Set[int] = set()

        def do_recycle_temporaries(lhs: sy.Symbol, rhs: sy.Expr, i: int) -> Tuple[sy.Symbol, sy.Expr]:
            active_replacements: List[TemporaryReplacement] = (
                sorted(filter(lambda r: r.begin_eqn <= i <= r.end_eqn, eqn_list.temporary_replacements),
                       key=lambda r: r.begin_eqn,
                       reverse=True)
            )

            current_line_replacement = typing.cast(Optional[TemporaryReplacement],
                                                   next(filter(lambda r: r.begin_eqn == i, active_replacements), None))

            for replacement in active_replacements:
                rhs = rhs.replace(replacement.old, replacement.new)  # type: ignore[no-untyped-call]

            if current_line_replacement:
                assert lhs == current_line_replacement.old, "Current line replacement target doesn't match LHS"
                lhs = current_line_replacement.new
                reassigned_lhses.add(i)

            return lhs, rhs

        # Sort the equations, perform temp-var replacements if needed, then convert each RHS to our tree type.
        eqns = [(lhs, SympyExpr(rhs)) for lhs, rhs in [do_recycle_temporaries(lhs, rhs, i) for i, (lhs, rhs) in
                                                       enumerate(sorted(eqn_list.eqns.items(), key=lambda kv: eqn_list.order.index(kv[0])))]]

        # Build the function decl and its body.
        nodes.append(
            ThornFunctionDecl(
                Identifier(fn_name),
                [DeclareCarpetXArgs(Identifier(fn_name)),
                 DeclareCarpetParams(),
                 UsingAlias(Identifier('vreal'), VerbatimExpr(Verbatim('Arith::simd<CCTK_REAL>'))),
                 ConstExprAssignDecl(Identifier('std::size_t'), Identifier('vsize'), VerbatimExpr(Verbatim('std::tuple_size_v<vreal>'))),
                 *layout_decls,
                 *di_decls,
                 *stencil_limit_checks,
                 CarpetXGridLoopCall(
                     output_centering,
                     output_region,
                     CarpetXGridLoopLambda(
                         preceding=xyz_decls,
                         equations=eqns,
                         succeeding=[],
                         temporaries=[str(lhs) for lhs in OrderedSet(eqn_list.eqns.keys()) if lhs in thorn_fn.eqn_list.temporaries],
                         reassigned_lhses=reassigned_lhses
                     ),
                 )]
            )
        )

        return CodeRoot(nodes)

    def generate_sync_batch_function_code(self, sync_batch: ExplicitSyncBatch) -> CodeRoot:
        return CodeRoot([
            Verbatim(self.boilerplate_setup),
            *[IncludeDirective(include) for include in self.boilerplate_includes],
            ThornFunctionDecl(
                Identifier(sync_batch.name),
                []
            )
        ])
