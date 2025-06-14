from EmitCactus.emit.ccl.param.param_tree import ParamRoot, Param, ParamAccess, ParamType, Steerability, \
    KeywordParamRange, IntParamRange, IntParamDescRange, IntParamClosedLowerBound, IntParamOpenUpperBound
from EmitCactus.emit.ccl.param.param_visitor import ParamVisitor
from EmitCactus.emit.tree import Identifier, String, Integer

if __name__ == '__main__':
    v = ParamVisitor()

    p = ParamRoot([
        Param(
            param_access=ParamAccess.Restricted,
            param_type=ParamType.Keyword,
            param_name=Identifier('Foo'),
            param_desc=String('recipes recipes'),
            steerability=Steerability.Always,
            range_descriptions=[
                KeywordParamRange([String(x) for x in ['foo', 'bar']], String('comment1')),
                KeywordParamRange([String(x) for x in ['baz', 'qux']], String('comment2'))
            ],
            default_value=String('foo')
        ),
        Param(
            param_access=ParamAccess.Shares,
            shares_with=Identifier('some_other_impl'),
            param_type=ParamType.Int,
            param_name=Identifier('Bar'),
            param_desc=String('recipes recipes recipes'),
            range_descriptions=[
                IntParamRange(IntParamDescRange(
                    IntParamClosedLowerBound(Integer(2)),
                    IntParamOpenUpperBound(Integer(5))
                ), String('comment3'))
            ],
            default_value=Integer(3)
        )
    ])

    print(v.visit(p))
