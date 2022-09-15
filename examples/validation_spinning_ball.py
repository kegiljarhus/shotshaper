# -*- coding: utf-8 -*-
"""
This program calculates the trajectory of a spinning soccer ball
that was experimentally investigated in

Mencke, J. E., Salewski, M., Trinhammer, O. L., & Adler, A. T. (2020). 
Flight and bounce of spinning sports balls. American Journal of Physics, 
88(11), 934-947.

"""

from shotshaper.projectile import SoccerBall
import matplotlib.pyplot as pl
import numpy as np
from scipy.linalg import norm

ball = SoccerBall()
u = np.array([20.0, 2.5, 4.9])
# Note that spin is here negative along the z-axis, as opposed
# to positive around the y-axis, since we define z as upwards,
# while Mencke et al. define y as upwards 
spin = (0,0,-46) 
speed = norm(u)
pitch = np.degrees(np.arctan2(u[2],u[0]))
yaw = -np.degrees(np.arctan2(u[1],u[0]))
    
s = ball.shoot(speed=speed,pitch=pitch,yaw=yaw,spin=spin)

x,y,z = s.position

f, (ax1, ax2) = pl.subplots(2, 1, figsize=(5,7))

ax1.plot(x,z,'C0-')
ax2.plot(x,y,'C0-')    
x,z = np.loadtxt('data/Spin_z_trajectory.dat', unpack=True, delimiter=';')
ax1.plot(x,z,'C1--')
x,y = np.loadtxt('data/Spin_y_trajectory.dat', unpack=True, delimiter=';')
ax2.plot(x,y,'C1--')    

ax1.legend(('Simulation','Experiment'))
ax2.legend(('Simulation','Experiment'))
    
ax1.set_xlabel('Length (m)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Drift (m)')

pl.show()
