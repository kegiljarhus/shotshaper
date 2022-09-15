# -*- coding: utf-8 -*-

import sys
from shotshaper.projectile import DiscGolfDisc
import matplotlib.pyplot as pl
import numpy as np
import shotshaper.environment as env
from shotshaper.transforms import T_12
from random import uniform


def rotz(pos,betad):
    beta = np.radians(betad)
    TZ =  np.array([[np.cos(beta), -np.sin(beta), 0],
                    [np.sin(beta),  np.cos(beta), 0],
                    [0,             0,            1]])

    return np.matmul(TZ,pos)

throws=[1,6,15]

nthrow = len(throws)
        # pitch, roll, nose,speed,spin,  yaw,  wind, length
params = [[15.5, 21.8, 0.0, 24.7, 138, -31.6, 4.8, 89.7],
          [12.3, 14.7, 0.8, 24.2, 128.5, -9.60, 4.8, 87.0],
          [5.20,-0.70, 0.0, 24.5, 147.7,  8.00, 4.8, 106.6]]
         

d = DiscGolfDisc('dd2')
fig1, ax1 = pl.subplots( )
    
fig1.set_figheight(4)
fig1.set_figwidth(6)

for i in range(nthrow):
    p = params[i]
    t = throws[i]
    U = p[3]
    omega = p[4]
    z0 = 1.5
    pos = np.array((0,0,z0))
    pitch = p[0]
    yaw = p[5]
    nose = p[2]
    roll = p[1]
    
    env.Uref = p[6]
    env.winddir = np.array((1,0,0))

    # Currently handle yaw by rotating the position after the throw, hence
    # also need to rotate the wind vector accordingly
    env.winddir = rotz(env.winddir, -yaw)

    s = d.shoot(speed=U, omega=omega, pitch=pitch, position=pos, nose_angle=nose, roll_angle=roll)

    pos = s.position
    for j in range(len(pos[0,:])):
        pos[:,j] = rotz(pos[:,j], yaw)
    x,y,z = pos
    arc,alphas,betas,lifts,drags,moms,rolls = d.post_process(s, omega)
    
    # Plot trajectory
    ax1.plot(x,y,f'C{i}-')
    
    # Experiment
    te,xe,ye,ve = np.loadtxt(f'data/throw{t}',skiprows=2,unpack=True)
    ax1.plot(xe,ye,f'C{i}--')

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Drift (m)')
    ax1.axis('equal')
    
    # Plot other parameters
    
    # axes[0,0].plot(arc, lifts)
    # axes[0,0].set_xlabel('Distance (m)')
    # axes[0,0].set_ylabel('Lift force (N)')
    
    # axes[0,1].plot(arc, drags)
    # axes[0,1].set_xlabel('Distance (m)')
    # axes[0,1].set_ylabel('Drag force (N)')
    
    # axes[0,2].plot(arc, moms)
    # axes[0,2].set_xlabel('Distance (m)')
    # axes[0,2].set_ylabel('Moment (Nm)')
    
    # axes[1,0].plot(arc, alphas)
    # axes[1,0].set_xlabel('Distance (m)')
    # axes[1,0].set_ylabel('Angle of attack (deg)')
    
    # axes[1,1].plot(arc, s.velocity[0,:])
    # axes[1,1].plot(arc, s.velocity[1,:])
    # axes[1,1].plot(arc, s.velocity[2,:])
    # axes[1,1].set_xlabel('Distance (m)')
    # axes[1,1].set_ylabel('Velocities (m/s)')
    
    # axes[1,2].plot(arc, rolls)
    # axes[1,2].set_xlabel('Distance (m)')
    # axes[1,2].set_ylabel('Roll rate (rad/s)')

pl.tight_layout()
pl.show()




