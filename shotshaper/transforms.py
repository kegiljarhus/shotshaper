# -*- coding: utf-8 -*-
"""

The axes are denoted:
1: Earth
2: Body
3: Zero side slip - rotation around z, to apply roll
4: Wind axes - to replicate the CFD condition with wind coming straight onto the body

In Crowther & Potts, z-axis is defined pointing downwards. Here, it points upwards
leading to some different signs.
"""
import numpy as np
from numpy import cos,sin,matmul

def T_12(attitude):
    """
    Transform from Earth axes to Body axes
    """
    phi, theta, psi = attitude
                                                                    
    return np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)],
                     [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi)],
                     [-sin(theta),         sin(phi)*cos(theta),                              cos(phi)*cos(theta)                             ]])

def T_23(beta):
    """
    Transform from Body axes to Zero side slip axes,
    Rotation around z-axis by the side-slip angle
    """
    return np.array([[cos(beta), -sin(beta), 0],
                     [sin(beta),  cos(beta), 0],
                     [0,          0,         1]])


def T_34(alpha):
    """
    Transform from Zero side slip axes to Wind axes
    Rotation around y-axis by the angle of attack.
    """
    return np.array([[cos(alpha), 0, -sin(alpha)],
                     [0,          1,  0         ],
                     [sin(alpha), 0,  cos(alpha)]])

def T_14(vec, attitude, beta, alpha):
    return matmul(T_34(alpha), matmul(T_23(beta), matmul(T_12(attitude), vec)))

def T_21(attitude):
    """
    Transform from Body axes to Earth axes.
    Done by transposing the opposite 
    """
    return np.transpose(T_12(attitude))

def T_32(beta):
    return np.transpose(T_23(beta))

def T_43(alpha):
    return np.transpose(T_34(alpha))

def T_41(vec, attitude, beta, alpha):
     return matmul(T_21(attitude), matmul(T_32(beta), matmul(T_43(alpha), vec)))

def T_31(vec, attitude, beta):
    return matmul(T_21(attitude), matmul(T_32(beta), vec))


                   
