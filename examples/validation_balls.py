# -*- coding: utf-8 -*-
"""
This program calculates trajectories that were experimentally conducted in

Mencke, J. E., Salewski, M., Trinhammer, O. L., & Adler, A. T. (2020). 
Flight and bounce of spinning sports balls. American Journal of Physics, 
88(11), 934-947.

Specifically, this includes data for:
    - a shot put throw, representing a heavy projectile where the 
      gravitational forces dominate over the aerodynamic forces
    - a soccer ball, with slightly more influence of air drag
    - a table tennis ball, significantly impacted by air drag due
      to low weight
"""

from shotshaper.projectile import SoccerBall, ShotPutBall, TableTennisBall
import matplotlib.pyplot as pl
import numpy as np

cases = ('Shot','Soccer ball','Table tennis ball')
projectiles = (ShotPutBall('M'), SoccerBall(), TableTennisBall())
ux = (9.0, 11.0, 11.8)
uy = (7.1, 10.2, 12.0)
z0 = (2.4, 0.0, 1.0)
spin = (0,0,0)

f, ax = pl.subplots(1, 1, figsize=(6,4))
for i,c in enumerate(cases):
    speed = np.sqrt(ux[i]**2 + uy[i]**2)
    pitch = np.degrees(np.arctan2(uy[i],ux[i]))
    position = (0, 0, z0[i])
    
    s = projectiles[i].shoot(speed=speed,pitch=pitch,position=position,spin=spin)

    x,y,z = s.position

    ax.plot(x,z,'C0-')
    ax.text(x[0]-0.5, z[0], c, fontsize=9, ha='right')
    
    x,z = np.loadtxt(f'data/{c}_trajectory.dat', unpack=True, delimiter=';')
    ax.plot(x,z,'C1--')
    
ax.legend(('Simulation','Experiment'))
ax.axis((-10,25,-1,6))
    
pl.xlabel('Length (m)')
pl.ylabel('Height (m)')

pl.show()
