#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# 工具函数，创建文件夹
def make_dir(path):
  """ 如果没有这个文件夹，就创建 """
  try:
    os.mkdir(path)
  except OSError:
    pass

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

Py3 = sys.version_info[0] == 3
if Py3:
  from queue import Queue
else:
  from Queue import Queue