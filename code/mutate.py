import random
import copy
import generate_expression
import lib.expressions as Expressions


def mutate(expressions, operations_binary, operations_unary, symbols_used, probability=0.1):
    """Mutates expressions with more sophisticated strategies.

    Args:
        expressions: A list of Expressions.
        operations_binary: A list of binary operations.
        operations_unary: A list of unary operations.
        symbols_used: A list of symbolic variables.
        probability: The probability of mutating an expression.
    """
    for index in range(len(expressions)):
        if random.random() < probability:
            expr = expressions[index]
            nodes = list(expr.get_nodes())  # Get all nodes in the expression tree

            if nodes:  # Check to prevent errors if the expression is empty
                # Choose a node randomly (weighted by depth for more diverse mutations)
                weights = [n.depth for n in nodes]
                chosen_node = random.choices(nodes, weights=weights)[0]


                if isinstance(chosen_node, Expressions.Constant):
                    # Mutate constant:  Vary magnitude and potentially sign.
                    new_value = chosen_node.value * (1 + random.uniform(-0.5, 0.5))  # Add up to 50% variation
                    if random.random() < 0.2: #20% chance of flipping sign
                        new_value *= -1

                    if chosen_node.father:  # Check if this is the root
                        chosen_node.value = new_value
                    else:
                        expressions[index] = Expressions.Constant(new_value)
                        
                elif isinstance(chosen_node, tuple(operations_binary)):
                    # Mutate binary operation: Replace with another binary operation.
                    new_operation = random.choice(operations_binary)
                    if chosen_node.father:
                        new_node = new_operation(*chosen_node.sons)
                        chosen_node.father.replace_son(chosen_node, new_node)
                    else:
                        new_node = new_operation(*chosen_node.sons)
                        expressions[index] = new_node

                elif isinstance(chosen_node, tuple(operations_unary)):
                     #Mutate unary operation
                    new_operation = random.choice(operations_unary)
                    if chosen_node.father:
                        new_node = new_operation(chosen_node.sons[0])
                        chosen_node.father.replace_son(chosen_node, new_node)
                    else:
                        new_node = new_operation(chosen_node.sons[0])
                        expressions[index] = new_node
                        
                elif isinstance(chosen_node, Expressions.Variable):
                    # Mutate variable: Replace with another variable.
                    new_variable = random.choice(symbols_used)
                    if chosen_node.father:
                        chosen_node.father.replace_son(chosen_node, new_variable)
                    else:
                        expressions[index] = new_variable

                else:
                    print("Warning: unexpected node type encountered during mutation:", type(chosen_node))


# Helper function to replace a node in a tree
Expressions.Node.replace_son = lambda node, new_node: (setattr(node.father, f'sons[{list(node.father.sons).index(node)}]', new_node))

#Example Usage
if __name__ == '__main__':
    operations_binary = [Expressions.Add, Expressions.Multiply,
                         Expressions.Subtract, Expressions.Divide]
    operations_unary = [Expressions.Sin, Expressions.Cos, Expressions.Exp]
    x = Expressions.Variable("x")
    v_x = Expressions.Variable("v_x")
    a_x = Expressions.Variable("a_x")
    symbols_used = [x, v_x, a_x]
    expr1 = generate_expressions.get_a_single_expression(1)
    print("Original:", expr1)
    l = [expr1]
    mutate(l, operations_binary, operations_unary, symbols_used, 1) # probability of 1 means every expression will be mutated
    print("Mutated:", l[0])