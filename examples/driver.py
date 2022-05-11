# -*- coding: utf-8 -*-
"""
Simple example comparing different projectiles.
"""

from shotshaper.projectile import _Particle, ShotPutBall, SoccerBall
import matplotlib.pyplot as pl
import numpy as np

U = 10.0
angle = 20.0

p = _Particle()
shot = p.shoot(speed=U, pitch=angle)
pl.plot(shot.position[0,:],shot.position[2,:])

p = ShotPutBall('M')
shot = p.shoot(speed=U, pitch=angle)
pl.plot(shot.position[0,:],shot.position[2,:])

p = SoccerBall()

spin = np.array((0,0,0))
shot = p.shoot(speed=U, pitch=angle, spin=spin)
pl.plot(shot.position[0,:],shot.position[2,:])

spin = np.array((0,-10,0))
shot = p.shoot(speed=U, pitch=angle, spin=spin)
pl.plot(shot.position[0,:],shot.position[2,:])

pl.legend(('vacuum','air', 'no spin','spin'))

pl.show()


