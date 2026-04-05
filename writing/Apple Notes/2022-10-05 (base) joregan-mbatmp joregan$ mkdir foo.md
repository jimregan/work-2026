(base) joregan-mba:tmp joregan$ mkdir foo
(base) joregan-mba:tmp joregan$ cd foo
(base) joregan-mba:foo joregan$ mkdir a
(base) joregan-mba:foo joregan$ mkdir b
(base) joregan-mba:foo joregan$ touch c
(base) joregan-mba:foo joregan$ touch d
(base) joregan-mba:foo joregan$ mkdir e

>>> import glob
>>> import os
>>> for f in glob.glob("**"):
...     if os.path.isdir(f):
...             print(f, "is a directory")
... 
a is a directory
e is a directory
b is a directory