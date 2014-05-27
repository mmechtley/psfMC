import ast
from .ModelComponents import ComponentBase

_comps_name = 'components'


class ExprsToAssigns(ast.NodeTransformer):
    """
    Walks the Abstract Syntax Tree from the parsed model file, and transforms
    bare expressions into augmented assignments that are tacked on to the end
    of the components list, ie:
    Sky(...)
    becomes:
    components += [Sky(...)]
    """
    def visit_Expr(self, node):
        return ast.copy_location(ast.AugAssign(
            target=ast.Name(id=_comps_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.List(elts=[node.value], ctx=ast.Load())
        ), node)


def component_list_from_file(filename):
    """
    Read in a file containing fit components and return them as a list
    """
    with open(filename) as f:
        model_tree = ast.parse(f.read())

    # Inject distribution and component imports. Put them at the beginning
    # so user imports may override them. level=1 means relative import, e.g.:
    # from .ModelComponents import *
    ast.increment_lineno(model_tree, n=3)
    comps = ast.ImportFrom(module='ModelComponents',
                           names=[ast.alias(name='*', asname=None)], level=1)
    dists = ast.ImportFrom(module='distributions',
                           names=[ast.alias(name='*', asname=None)], level=1)
    model_tree.body.insert(0, comps)
    model_tree.body.insert(1, dists)

    # Insert a statement creating an empty list called components
    comps_node = ast.Assign(targets=[ast.Name(id=_comps_name, ctx=ast.Store())],
                            value=ast.List(elts=[], ctx=ast.Load()))
    model_tree.body.insert(2, comps_node)

    # Transform bare components expressions into list append statements
    model_tree = ExprsToAssigns().visit(model_tree)
    ast.fix_missing_locations(model_tree)

    exec(compile(model_tree, filename, mode='exec'))

    return [comp for comp in locals()[_comps_name]
            if isinstance(comp, ComponentBase.ComponentBase)]
