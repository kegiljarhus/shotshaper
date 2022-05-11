# -*- coding: utf-8 -*-
"""
Graphical user interface to explore the influence
of disc throw parameters on the trajectory
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider, TextBox
from shotshaper.projectile import DiscGolfDisc

name = 'dd2'
mass = 0.175

d = DiscGolfDisc(name, mass=mass)
speed = 24
omega = d.empirical_spin(speed)
z0 = 1.3
pos = np.array((0,0,z0))
pitch = 10
nose = 0.0
roll = 15.0  
yaw = 0
adjust_axes = False
     
s = d.shoot(speed=speed, omega=omega, pitch=pitch, position=pos, nose_angle=nose, roll_angle=roll,yaw=yaw)

x,y,z = s.position

# Creating figure
fig = pl.figure(1,figsize=(13, 6), dpi=80)
ax1 = pl.subplot(2,3,2)
ax2 = pl.subplot(2,3,5)
ax3 = pl.subplot(1,3,3)
fac = 1.6
ylim = fac*max(abs(min(y)),abs(max(y)))
ax1.axis((min(x),fac*max(x),-ylim,ylim))
ax2.axis((min(x),fac*max(x),min(z),fac*max(z)))
ax3.axis((-ylim,ylim,min(z),fac*max(z)))
        
ax3.invert_xaxis()

l1, = ax1.plot(x,y,lw=2)
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Drift (m)')
l2, = ax2.plot(x,z,lw=2)
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Height (m)')

l3, = ax3.plot(y,z,lw=2)
ax3.set_xlabel('Drift (m)')
ax3.set_ylabel('Height (m)')

xax = 0.07
ax4  = pl.axes([xax, 0.80, 0.25, 0.03], facecolor='lightgrey')
ax5  = pl.axes([xax, 0.75, 0.25, 0.03], facecolor='lightgrey')
ax6  = pl.axes([xax, 0.70, 0.25, 0.03], facecolor='lightgrey')
#ax7  = pl.axes([xax, 0.65, 0.25, 0.03], facecolor='lightgrey')
ax8  = pl.axes([xax, 0.65, 0.25, 0.03], facecolor='lightgrey')
ax9  = pl.axes([xax, 0.60, 0.25, 0.03], facecolor='lightgrey')
ax11 = pl.axes([xax, 0.55, 0.25, 0.03], facecolor='lightgrey')

s1 = Slider(ax=ax4, label='Speed (m/s)', valmin=15,  valmax=35, valinit=speed)
s2 = Slider(ax=ax5, label='Roll (deg)',   valmin=-110, valmax=110, valinit=roll)
s3 = Slider(ax=ax6, label='Pitch (deg)', valmin=-10,   valmax=50, valinit=pitch)
s5 = Slider(ax=ax8, label='Nose (deg)',   valmin=-5, valmax=5, valinit=nose)
s7 = Slider(ax=ax9, label='Mass (kg)',   valmin=0.140, valmax=0.200, valinit=mass)
s6 = Slider(ax=ax11, label='Spin (-)',   valmin=0, valmax=2, valinit=1.0)

def update(x):
    speed = s1.val
    roll = s2.val
    pitch = s3.val
    nose = s5.val
    spin = s6.val
    mass = s7.val 
    d = DiscGolfDisc(name,mass=mass)
    
    omega = spin*d.empirical_spin(speed)
    s = d.shoot(speed=speed, omega=omega, pitch=pitch, position=pos, nose_angle=nose, roll_angle=roll)
    x,y,z = s.position
    
    l1.set_xdata(x)
    l1.set_ydata(y)
    l2.set_xdata(x)
    l2.set_ydata(z)
    l3.set_xdata(y)
    l3.set_ydata(z)
    
    if adjust_axes:
        ax1.axis((min(x),max(x),min(y),max(y)))
        ax2.axis((min(x),max(x),min(z),max(z)))
        ax3.axis((min(y),max(y),min(z),max(z)))
        
    fig.canvas.draw_idle()

s1.on_changed(update)
s2.on_changed(update)
s3.on_changed(update)
s5.on_changed(update)
s6.on_changed(update)
s7.on_changed(update)

pl.show()
    
