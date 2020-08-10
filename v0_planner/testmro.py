#!/usr/bin/env python3
class First(object):
    def __init__(self):
  #      super().__init__('First')
        super(First, self).__init__()
        print("first a la ros")

class Second(object):
    def __init__(self):
        super(Second, self).__init__()
        print("second")

class Third(First, Second):
    def __init__(self):
        super(Third, self).__init__()
        print("third")