# -*- coding: utf-8 -*-
"""
Example showing a single disc throw.
"""

from shotshaper.projectile import DiscGolfDisc
import matplotlib.pyplot as pl
import numpy as np

d = DiscGolfDisc('dd2')
U = 24.2
omega = 116.8
z0 = 1.3
pos = np.array((0,0,z0))
pitch = 15.5
nose = 0.0
roll = 14.7

shot = d.shoot(speed=U, omega=omega, pitch=pitch, 
               position=pos, nose_angle=nose, roll_angle=roll)

# Plot trajectory
pl.figure(1)
x,y,z = shot.position
pl.plot(x,y)

pl.xlabel('Distance (m)')
pl.ylabel('Drift (m)')
pl.axis('equal')

# Plot other parameters
arc,alphas,lifts,drags,moms,rolls = d.post_process(shot, omega)
fig, axes = pl.subplots(nrows=2, ncols=3, dpi=80,figsize=(13,5))

axes[0,0].plot(arc, lifts)
axes[0,0].set_xlabel('Distance (m)')
axes[0,0].set_ylabel('Lift force (N)')

axes[0,1].plot(arc, drags)
axes[0,1].set_xlabel('Distance (m)')
axes[0,1].set_ylabel('Drag force (N)')

axes[0,2].plot(arc, moms)
axes[0,2].set_xlabel('Distance (m)')
axes[0,2].set_ylabel('Moment (Nm)')

axes[1,0].plot(arc, alphas)
axes[1,0].set_xlabel('Distance (m)')
axes[1,0].set_ylabel('Angle of attack (deg)')

axes[1,1].plot(arc, shot.velocity[0,:])
axes[1,1].plot(arc, shot.velocity[1,:])
axes[1,1].plot(arc, shot.velocity[2,:])
axes[1,1].set_xlabel('Distance (m)')
axes[1,1].set_ylabel('Velocities (m/s)')
axes[1,1].legend(('u','v','w'))

axes[1,2].plot(arc, rolls)
axes[1,2].set_xlabel('Distance (m)')
axes[1,2].set_ylabel('Roll rate (rad/s)')
pl.tight_layout()

pl.show()
