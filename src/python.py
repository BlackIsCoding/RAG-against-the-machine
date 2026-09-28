# import ast

# string = """x = 10

# def hello():
#     pass
    
# def test():
#     if a == 5:
#         wwww

# class Person:
#     pass

# y = 20
# """



# tree = ast.parse(string)
# print(tree)
# print(ast.dump(tree, indent=4))

# for node in tree.body:
#     print(type(node).__name__)
#     print(node.lineno)
#     print(node.end_lineno)
#     print(node.col_offset)
#     print(node.end_col_offset)
#     print(ast.get_source_segment(string, node))

# for node in ast.walk(tree):
#     print(type(node).__name__)

# for node in tree.body:
#     for child in ast.iter_child_nodes(node):
#         print(type(child).__name__)

import ast

# exercice 1
source = """
import os
import sys

x = 10

def hello(name):
    print(name)

class Person:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(self.name)

def goodbye():
    print("bye")
"""

tree = ast.parse(source)

for node in tree.body:
    if isinstance(node , ast.FunctionDef) or isinstance(node , ast.ClassDef):
        name = node.name
        line = node.lineno
        print(f"{type(node).__name__} : {name}, line {line} --> {node.end_lineno}")
    else:
        print(f"{type(node).__name__} : line {node.lineno} -> {node.end_lineno}")

# exercice 2
print()
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        text = ast.get_source_segment(source, node)
        print(f"---{node.name}---")
        print(text)

# exercice 3
print()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        print("name :", node.name)
        print("start :", node.lineno)
        print("end :", node.end_lineno)
        print("source :", ast.get_source_segment(source, node), end='\n\n')