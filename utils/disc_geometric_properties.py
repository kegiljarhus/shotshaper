# -*- coding: utf-8 -*-
"""
Calculate moments of inertia for a disc from STL file.
"""

import numpy as np
import trimesh
import sys
import os 

path = os.path.dirname(os.path.realpath(__file__))
# attach to logger so trimesh messages will be printed to console
#trimesh.util.attach_to_log()

name = sys.argv[-1]

m = trimesh.load(os.path.join(path, 'discs', name + '.stl'))
trimesh.repair.fix_inversion(m)
trimesh.repair.fix_normals(m)
trimesh.repair.fix_winding(m)

if m.is_watertight and m.is_winding_consistent and m.is_volume:
    V = m.volume
    J = m.principal_inertia_components/V
    print('Volume: ', V)
    print('J_xy: %4.3e' % J[0])
    print('J_z: %4.3e' % J[2])

