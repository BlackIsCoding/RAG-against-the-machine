# # import ast

# # string = """x = 10

# # def hello():
# #     pass
    
# # def test():
# #     if a == 5:
# #         wwww

# # class Person:
# #     pass

# # y = 20
# # """



# # tree = ast.parse(string)
# # print(tree)
# # print(ast.dump(tree, indent=4))

# # for node in tree.body:
# #     print(type(node).__name__)
# #     print(node.lineno)
# #     print(node.end_lineno)
# #     print(node.col_offset)
# #     print(node.end_col_offset)
# #     print(ast.get_source_segment(string, node))

# # for node in ast.walk(tree):
# #     print(type(node).__name__)

# # for node in tree.body:
# #     for child in ast.iter_child_nodes(node):
# #         print(type(child).__name__)

# import ast

# # exercice 1
# source = """
# import os
# import sys

# x = 10

# def hello(name):
#     print(name)

# class Person:
#     def __init__(self, name):
#         self.name = name

#     def say_hello(self):
#         print(self.name)

# def goodbye():
#     print("bye")
# """

# tree = ast.parse(source)

# for node in tree.body:
#     if isinstance(node , ast.FunctionDef) or isinstance(node , ast.ClassDef):
#         name = node.name
#         line = node.lineno
#         print(f"{type(node).__name__} : {name}, line {line} --> {node.end_lineno}")
#     else:
#         print(f"{type(node).__name__} : line {node.lineno} -> {node.end_lineno}")

# # exercice 2
# print()
# for node in tree.body:
#     if isinstance(node, ast.FunctionDef):
#         text = ast.get_source_segment(source, node)
#         print(f"---{node.name}---")
#         print(text)

# # exercice 3
# print()
# for node in ast.walk(tree):
#     if isinstance(node, ast.FunctionDef):
#         print("name :", node.name)
#         print("start :", node.lineno)
#         print("end :", node.end_lineno)
#         print("source :", ast.get_source_segment(source, node), end='\n\n')

# import ast

# source = """
# import os


# def read_file(path):
#     with open(path, "r") as f:
#         content = f.read()
#     return content


# def process_data(data):
#     result = []

#     for item in data:
#         if item > 10:
#             result.append(item * 2)
#         else:
#             result.append(item)

#     return result


# class DataProcessor:

#     def __init__(self, data):
#         self.data = data

#     def process(self):
#         return process_data(self.data)


# def main():
#     data = read_file("data.txt")
#     result = process_data(data)
#     print(result)
# """

# tree = ast.parse(source)


# def chunking(node, name):
#     chunk = {"type": type(node).__name__,
#             "name": name,
#             "start": node.lineno,
#             "end": node.end_lineno,
#             "text": ast.get_source_segment(source, node)}   
#     return chunk 

# chunks = []
# for node in tree.body:
#     if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#         name = node.name
#     else:
#         name = '-'
#     chunk = chunking(node, name)
#     if isinstance(node, ast.ClassDef):
#         chunk['children'] = []
#         for n in node.body:
#             name = n.name
#             chunk['children'].append(chunking(n, name))
#     chunks.append(chunk)

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i} :")
#     print(chunk)

import ast


def chunk_python(source, max_size):
    pass

source = """
import os
import sys


def read_file(path):
    with open(path, "r") as f:
        content = f.read()
    return content


def process_data(data):
    result = []

    for item in data:
        if item > 10:
            result.append(item * 2)
        else:
            result.append(item)

    return result


class DataProcessor:

    def __init__(self, data):
        self.data = data

    def process(self):
        return process_data(self.data)


def main():
    data = read_file("data.txt")
    result = process_data(data)
    print(result)
"""


tree = ast.parse(source)
chunks = []

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        name = node.name
    else:
        name = '-'
    chunk = {"type": type(node).__name__,
             "name": name,
             "size": node.end_col_offset - node.col_offset,
             "lines": (node.lineno, node.end_lineno),
             "text": ast.get_source_segment(source, node)}
    if isinstance(node, ast.ClassDef):
        chunk['children'] = []
        for child in node.body:
            child.parent = "class " + node.name
            c = {"type": type(child).__name__,
             "name": name,
             "size": child.end_col_offset - child.col_offset,
             "lines": (child.lineno, child.end_lineno),
             "text": ast.get_source_segment(source, child),
             "parent": child.parent}
            chunk['children'].append(c)
    chunks.append(chunk)

for chunk in chunks:
    if chunk['type'] == "ClassDef":
        print(chunk)