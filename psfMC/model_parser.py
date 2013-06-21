from ast import *
from .ModelComponents import ComponentBase

_comps_name = 'components'


class ExprsToAssigns(NodeTransformer):
    """
    Walks the Abstract Syntax Tree from the parsed model file, and transforms
    bare expressions into augmented assignments that are tacked on to the end
    of the components list, ie:
    Sky(...)
    becomes:
    components += [Sky(...)]
    """
    def visit_Expr(self, node):
        return copy_location(AugAssign(
            target=Name(id=_comps_name, ctx=Store()),
            op=Add(),
            value=List(elts=[node.value], ctx=Load())
        ), node)


def component_list_from_file(filename):
    """
    Read in a file containing fit components and return them as a list
    """
    with open(filename) as f:
        model_tree = parse(f.read())

    # Inject distribution and component imports. Put them at the beginning
    # so user imports may override them. level=1 means relative import, e.g.:
    # from .ModelComponents import *
    increment_lineno(model_tree, n=3)
    comps = ImportFrom(module='ModelComponents',
                       names=[alias(name='*', asname=None)], level=1)
    dists = ImportFrom(module='distributions',
                       names=[alias(name='*', asname=None)], level=1)
    model_tree.body.insert(0, comps)
    model_tree.body.insert(1, dists)

    # Insert a statement creating an empty list called components
    comp_list_node = Assign(targets=[Name(id=_comps_name, ctx=Store())],
                            value=List(elts=[], ctx=Load()))
    model_tree.body.insert(2, comp_list_node)

    # Transform bare components expressions into list append statements
    model_tree = ExprsToAssigns().visit(model_tree)
    fix_missing_locations(model_tree)

    exec(compile(model_tree, filename, mode='exec'))

    return [comp for comp in locals()[_comps_name]
            if isinstance(comp, ComponentBase)]
