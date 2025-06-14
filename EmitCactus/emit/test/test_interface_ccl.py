from EmitCactus.emit.ccl.interface.interface_tree import InterfaceRoot, HeaderSection, IncludeSection, FunctionSection, \
    VariableSection, VariableGroup, Access, DataType, GroupType
from EmitCactus.emit.ccl.interface.interface_visitor import InterfaceVisitor
from EmitCactus.emit.tree import Identifier, Integer, String

if __name__ == '__main__':
    v = InterfaceVisitor()

    i = InterfaceRoot(
        HeaderSection(
            implements=Identifier('HeatEqn'),
            inherits=[Identifier('Grid')],
            friends=[]
        ),
        IncludeSection([]),
        FunctionSection([]),
        VariableSection([
            VariableGroup(
                access=Access.Public,
                group_name=Identifier('evol_group'),
                data_type=DataType.Real,
                group_type=GroupType.GF,
                time_levels=Integer(2),
                variable_names=[Identifier('U')],
                group_description=String("Heat equation fields")
            )
        ])
    )

    print(v.visit(i))
