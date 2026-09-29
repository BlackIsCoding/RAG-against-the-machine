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