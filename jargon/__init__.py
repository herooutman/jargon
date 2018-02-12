#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

HOME_DIR = os.getenv("HOME")
LIB_DIR = os.path.join(HOME_DIR, "sbcode/lib/python")
sys.path.append(LIB_DIR)
