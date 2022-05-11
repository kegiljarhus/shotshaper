# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:29:10 2021

@author: 2913452

TODO:
    - height of athlete, optimal angle shot put
        - note that biomech can influence
          how much force an athlete can emit for each angle
    - 
"""

from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .transforms import T_12, T_23, T_34, T_14, T_41, T_31
import matplotlib.pyplot as pl
from numpy import matmul,pi,sqrt,arctan2,radians,degrees,sin,cos,array,concatenate,linspace,zeros_like,cross,zeros,argmin
from numpy.linalg import norm
from . import environment
import os
import yaml

T_END = 60 # Maximum time for simulation
N_STEP = 100 # Number of simulation steps to use for post-processing. Note: Not actual simulation steps

def hit_ground(t, x, *args): 
    return x[2]

def stopped(t, x, *args):
    U = norm(x[3:6])
    return U - 1e-4

class Shot:
    def __init__(self,t,x,v,att=None):
        self.time = t
        self.position = x
        self.velocity = v
        if att is not None:
            self.attitude = att
        

class _Projectile(ABC):
    def __init__(self):
        pass
   
    def _shoot(self, advance_function, y0, *args):
        hit_ground.terminal = True
        hit_ground.direction = -1
        stopped.terminal = True
        stopped.direction = -1
        
        sol = solve_ivp(advance_function,[0,T_END],y0,
                        dense_output=True,args=args,
                        method='RK45',
                        events=(hit_ground,stopped))
        
        t = linspace(0,sol.t[-1],N_STEP)
        
        f = sol.sol(t)
        pos = array([f[0],f[1],f[2]])
        vel = array([f[3],f[4],f[5]])
        
        if len(f) <= 6:
            shot = Shot(t, pos, vel)
        else:    
            att = array([f[6],f[7],f[8]])
            shot = Shot(t, pos, vel, att)
        
        return shot
         
    @abstractmethod
    def advance(self,t,vec,*args):
        """
        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Right hand side of kinematic equations for a projectile
        :rtype: array
        """
        
class _Particle(_Projectile):
    def __init__(self):
        super().__init__()
        
        self.g = environment.g
        
    def initialize_shot(self, **kwargs):
        kwargs.setdefault('yaw', 0.0) 
        
        pitch = radians(kwargs["pitch"])
        yaw = radians(kwargs["yaw"])
        U = kwargs["speed"]
        xy = cos(pitch)
        u = U*xy*cos(yaw)
        w = U*sin(pitch)
        v = U*xy*sin(-yaw)
        if "position" in kwargs:
            x,y,z = kwargs["position"]
        else:
            x = 0.
            y = 0.
            z = 0.
        
        y0 = array((x,y,z,u,v,w))
        return y0
            
    def shoot(self, **kwargs):

        y0 = self.initialize_shot(**kwargs)
        shot = self._shoot(self.advance, y0)
        
        return shot
        
    def gravity_force(self, x=None):
        if x is None:
            return array((0,0,environment.g))
        else:
            # Messy way to return g array...
            f = zeros_like(x)
            f[0,:] = 0
            f[1,:] = 0
            f[2,:] = environment.g
            return f
        
    def advance(self, t, vec, *args):
        # x, y, z, u, v, w = vec
        x = vec[0:3]
        u = vec[3:6]
        
        f = self.gravity_force()
        
        return concatenate((u,f))
        
    
class _SphericalParticleAirResistance(_Particle):
    def __init__(self, mass, diameter):
        super().__init__()
        
        self.mass = mass
        self.diameter = diameter
        self.radius = 0.5*diameter
        self.area = 0.25*pi*diameter**2
        self.volume = 4./3.*pi*self.radius**3
        
    
    def air_resistance_force(self,U):
        Umag = norm(U)
        Cd = self.drag_coefficient(Umag)
        
        f = -0.5*environment.rho*self.area*Cd*abs(U)*U/self.mass
        
        return f
    
    def advance(self, t, vec, *args):
        x = vec[0:3]
        u = vec[3:6]
        
        f = self.air_resistance_force(u) \
          + self.gravity_force()
        
        return concatenate((u,f))
       
        
    def reynolds_number(self, velocity):
        """
        Reynolds number, non-dimensional number giving the 
        ratio of inertial forces to viscous forces. Used
        for calculating the drag coefficient.
        
        :param float velocity: Velocity seen by particle
        :return: Reynolds number
        :rtype: float
        
        """
        return environment.rho*velocity*self.diameter/environment.mu
    
    def drag_coefficient(self, velocity):
        """
        Drag coefficient for sphere, empirical curve fit
        taken from:
        
        F. A. Morrison, An Introduction to Fluid Mechanics, (Cambridge
        University Press, New York, 2013). This correlation appears in
        Figure 8.13 on page 625. 

        The full formula is:

        .. math::
            F = \\frac{2}{\\pi}\\cos^{-1}e^{-f} \\\\
            f = \\frac{B}{2}\\frac{R-r}{r\\sin\\phi}

        A hub loss is also caluclated in the same manner.

        :param float velocity: Velocity seen by particle
        :return: Drag coefficient
        :rtype: float
        """
        Re = self.reynolds_number(velocity)
        
        if Re <= 0:
            return 1e30
        
        tmp1 = Re/5.0
        tmp2 = Re/2.63e5
        tmp3 = Re/1e6
        
        Cd = 24.0/Re \
           + 2.6*tmp1/(1 + tmp1**1.52) \
           + 0.411*tmp2**-7.94/(1 + tmp2**-8) \
           + 0.25*tmp3/(1 + tmp3) 
           
        return Cd
    

class _SphericalParticleAirResistanceSpin(_SphericalParticleAirResistance):
    def __init__(self, mass, diameter):
        super().__init__(mass, diameter)
        
        
    def lift_coefficient(self, Umag, omega):
        # TODO - complex dependency on Re. For now,
        #        assume constant
        return 0.9
        
    def shoot(self, **kwargs):
        y0 = self.initialize_shot(**kwargs)
        spin = array((kwargs["spin"]))
        
        shot = self._shoot(self.advance, y0, spin)
        
        return shot        
    
    def spin_force(self,U,spin):
        
        Umag = norm(U)
        omega = norm(spin)
        
        Cl = self.lift_coefficient(Umag, omega)
        
        if U.ndim == 1:
            f = Cl*pi*self.radius**3*environment.rho*cross(spin, U)   
        else:
            f = zeros_like(U)
            for i in range(U.shape[1]):
                f[:,i] = Cl*pi*self.radius**3*environment.rho*cross(spin, U[:,i])   
        
        return f
    
    def advance(self, t, vec, spin):
        x = vec[0:3]
        u = vec[3:6]
        
        f = self.air_resistance_force(u) \
          + self.gravity_force() \
          + self.spin_force(u,spin)
        
        return concatenate((u,f))


class ShotPutBall(_SphericalParticleAirResistance):
    """
    Note that diameter can vary 110 mm to 130mm
    and 95 mm to 110 mm
    """
    def __init__(self, weight_class):
        
        if weight_class == 'M':
            mass = 7.26
            diameter = 0.11
        elif weight_class == 'F':
            mass = 4.0
            diameter = 0.095
            
        super().__init__(mass, diameter)
        
        
class SoccerBall(_SphericalParticleAirResistanceSpin):
    """
    Note that diameter can vary 110 mm to 130mm
    and 95 mm to 110 mm
    """
    def __init__(self):
        
        mass = 0.430    # 0.41-0.45
        diameter = 0.22 # 21.6-22.3
            
        super().__init__(mass, diameter)
    
    def drag_coefficient(self, velocity):
        # TODO - Texture and sewing pattern will alter
        #        the drag coefficient.
        return super().drag_coefficient(velocity)
    
    
class DiscGolfDisc(_Projectile):
    def __init__(self, name, mass=0.175):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, 'discs', name + '.yaml')
    
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            
            self.diameter = data['diameter']
            self.mass = mass
            self.weight = environment.g*mass
            self.area = pi*self.diameter**2/4.0
            self.I_xy = mass*data['J_xy']
            self.I_z = mass*data['J_z']
            
            a = array(data['alpha'])
            cl = array(data['Cl'])
            cd = array(data['Cd'])
            cm = array(data['Cm'])
        
        self._alpha,self._Cl,self._Cd,self._Cm = self._flip(a,cl,cd,cm)
        kind = 'linear'
        self.Cl_func = interp1d(self._alpha, self._Cl, kind=kind)
        self.Cd_func = interp1d(self._alpha, self._Cd, kind=kind)
        self.Cm_func = interp1d(self._alpha, self._Cm, kind=kind)
        
    def _flip(self,a,cl,cd,cm):
        """
        Data given from -90 deg to 90 deg.
        Expand to -180 to 180 using symmetry considerations.
        """
        n = len(a)

        idx = argmin(abs(a))
        a2 = zeros(2*n)
        cl2 = zeros(2*n)
        cd2 = zeros(2*n)
        cm2 = zeros(2*n)
        
        a2[idx:idx+n] = a[:]
        cl2[idx:idx+n] = cl[:]
        cd2[idx:idx+n] = cd[:]
        cm2[idx:idx+n] = cm[:]
        for i in range(idx):
            a2[i] = -(180 + a[idx-i])
            cl2[i] = -cl[idx-i]
            cd2[i] =  cd[idx-i]
            cm2[i] = -cm[idx-i]
        
        for i in range(idx+n,2*n):
            a2[i] = 180 - a[idx+n-i-2]
            cl2[i] = -cl[idx+n-i-2]
            cd2[i] =  cd[idx+n-i-2]
            cm2[i] = -cm[idx+n-i-2]
        
        return a2,cl2,cd2,cm2
        
    def _normalize_angle(self, alpha):
        """
        Ensure that the angle fulfils :math:`-\\pi < \\alpha < \\pi`

        :param float alpha: Angle in radians
        :return: Normalized angle
        :rtype: float
        """

        return arctan2(sin(alpha), cos(alpha))
    
    def Cd(self, alpha): 
        """
        Provide drag coefficent for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Drag coefficient
        :rtype: float
        """
        
        # NB! The stored data uses degrees for the angle
        return self.Cd_func(degrees(self._normalize_angle(alpha)))

    def Cl(self, alpha): 
        """
        Provide drag coefficent for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Drag coefficient
        :rtype: float
        """
        
        # NB! The stored data uses degrees for the angle
        return self.Cl_func(degrees(self._normalize_angle(alpha)))

    def Cm(self, alpha): 
        """
        Provide coefficent of moment for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Coefficient of moment
        :rtype: float
        """
    
        # NB! The stored data uses degrees for the angle
        return self.Cm_func(degrees(self._normalize_angle(alpha)))


    def plot_coeffs(self, color='k'):
        """
        Utility function to quickly explore disc coefficients.

        :param string color: Matplotlib color key. Default value is k, i.e. black.
        """
        pl.plot(self._alpha, self._Cl, 'C0-o',label='$C_L$')
        pl.plot(self._alpha, self._Cd, 'C1-o',label='$C_D$')
        pl.plot(self._alpha, 3*self._Cm, 'C2-o',label='$C_M$')
        
        
        pl.xlabel('Angle of attack ($^\circ$)')
        pl.ylabel('Aerodynamic coefficients (-)')
        pl.legend(loc='upper left')

        ax = pl.gca()
        ax2 = pl.gca().twinx()
        ax2.set_ylabel("Aerodynamic efficiency, $C_L/C_D$")
        pl.plot(self._alpha, self._Cl/self._Cd, 'C3-.',label='$C_L/C_D$')
        ax2.legend(loc='upper right')
        
        return ax,ax2
    
    
    def empirical_spin(self, speed):
        # Simple empirical formula for spin rate, based on curve-fitting
        # data from:
        # https://www.dgcoursereview.com/dgr/forums/viewtopic.php?f=2&t=7097

        omega = -0.257*speed**2 + 15.338*speed
        return omega
            
    
    def initialize_shot(self, **kwargs):
        U = kwargs["speed"]
        
        kwargs.setdefault('yaw', 0.0) 
        #kwargs.setdefault('omega', self.empirical_spin(U)) 
        
        pitch = radians(kwargs["pitch"])
        yaw = radians(kwargs["yaw"])
        omega = kwargs["omega"]
        
        # phi, theta
        roll_angle = radians(kwargs["roll_angle"]) # phi
        nose_angle = radians(kwargs["nose_angle"]) # theta
        # psi, rotation around z irrelevant for starting position
        #      since the disc is symmetric
        
        # Initialize position
        if "position" in kwargs:
            x,y,z = kwargs["position"]
        else:
            x = 0.
            y = 0.
            z = 0.
        
        # Initialize velocity
        xy = cos(pitch)
        u = U*xy*cos(yaw)
        v = U*xy*sin(-yaw)
        w = U*sin(pitch)
        
        # Initialize angles
        attitude = array([roll_angle, nose_angle, 0])
        # The initial orientation of the disc must also account for the
        # angle of the throw itself, i.e. the launch angle. 
        attitude += matmul(T_12(attitude), array((0, pitch, 0)))
        
        #attitude = matmul(T_23(yaw),attitude)
        #attitude += matmul(T_12(attitude), array((0, pitch, 0)))
        phi, theta, psi = attitude
        y0 = array((x,y,z,u,v,w,phi,theta,psi))
        return y0, omega
            
    def shoot(self, **kwargs):

        y0, omega = self.initialize_shot(**kwargs)
               
        shot = self._shoot(self.advance, y0, omega)
        
        return shot
    
    def post_process(self, s, omega):
        n = len(s.time)
        alphas = zeros(n)
        lifts = zeros(n)
        drags = zeros(n)
        moms = zeros(n)
        rolls = zeros(n)
        for i in range(n):
            x = s.position[:,i]
            u = s.velocity[:,i]
            a = s.attitude[:,i]
            
            alpha, beta, Fd, Fl, M, g4 = self.forces(x, u, a, omega)
            
            alphas[i] = alpha
            lifts[i] = Fl
            drags[i] = Fd
            moms[i] = M
            rolls[i] = -M/(omega*(self.I_xy - self.I_z))
        
        arc_length = norm(s.position, axis=0)
        return arc_length,degrees(alphas),lifts,drags,moms,degrees(rolls)
            
    def forces(self, x, u, a, omega):
        # Velocity in body axes
        urel = u - environment.wind_abl(x[2])
        u2 = matmul(T_12(a), urel)
        # Side slip angle is the angle between the x and y velocity
        beta = -arctan2(u2[1], u2[0])
        # Velocity in zero side slip axes
        u3 = matmul(T_23(beta), u2)
        # Angle of attack is the angle between 
        # vertical and horizontal velocity
        alpha = -arctan2(u3[2], u3[0])
        # Velocity in wind system, where forces are to be calculated
        u4 = matmul(T_34(alpha), u3)
        
        # Convert gravitational force from Earth to Wind axes
        g = array((0, 0, self.mass*environment.g))
        g4 = T_14(g, a, beta, alpha)
        
        # Aerodynamic forces
        q = 0.5*environment.rho*u4[0]**2
        S = self.area
        D = self.diameter
        
        Fd = q*S*self.Cd(alpha)
        Fl = q*S*self.Cl(alpha)
        M  = q*S*D*self.Cm(alpha)
        
        return alpha, beta, Fd, Fl, M, g4
        
    def advance(self, t, vec, omega):
        x = vec[0:3]
        u = vec[3:6]
        a = vec[6:9]
        
        alpha, beta, Fd, Fl, M, g4 = self.forces(x, u, a, omega)
        
        m = self.mass
        # Calculate accelerations
        dudt = (-Fd + g4[0])/m
        dvdt =        g4[1]/m
        dwdt = ( Fl + g4[2])/m
        acc4 = array((dudt,dvdt,dwdt))
        # Roll rate acts around x-axis (in axes 3: zero side slip axes)
        dphidt = -M/(omega*(self.I_xy - self.I_z))
        # Other angular rotations are ignored, assume zero wobble
        angvel3 = array((dphidt, 0, 0))
        
        acc1 = T_41(acc4, a, beta, alpha)
        angvel1 = T_31(angvel3, a, beta)
        
        return concatenate((u,acc1,angvel1)) 

    
    

    
