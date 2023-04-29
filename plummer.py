#
# Writing a general scheme for 4th order hermite integration of Plummer model
# with N particles, 3N positions, 3N velocities, N time factors,
# N would latter be 1024
# G = 1, total mass = 1
# softening of the potential with e = 4/N
# total energy of the system should be -1/4
# block time step algorithm, restricted to powers of 2, upper limit is 1/16
# IEEE 754 double precision, ie 64bit float
# No further can be achieved in python as conversion to 64bit is required mostly
# but we will set a provision
import numpy as np
import numpy.random as rnd
import scipy as sp
class Plummer():
    Time = 0.0  # global time variable
    PlummerRadius = 1.0
    Mass = 1.0  # Mass of one particle/star, = 1/N, as the total mass is 1
    G = 2.0      # gravitational constant
    Energy = 1.0
    def __init__(self, N=1024, seed=None, PlummerRadius=1, step=0.001, epsilon=None):
        print("hello")
        self.N = N
        self.Position = np.zeros([N,3])
        self.Velocity = np.zeros([N,3])
        self.Acceleration = np.zeros([N,3])
        self.Jerk = np.zeros([N,3])
        self.Snap = np.zeros([N,3])
        self.Crackle = np.zeros([N,3])
        self.PlummerRadius = PlummerRadius
        if epsilon is None:
            self.Epsilon = np.float64(4.0/N)
        else:
            self.Epsilon = epsilon
        self.Mass = np.float64(1.0/N)
        if seed is not None:
            self.Seed = seed
            rnd.seed(self.Seed)
        # comment or uncomment this

        # initializing r and v vectors

        r = 1/(self.PlummerRadius*np.sqrt(np.power(rnd.uniform(0, 1, N), -2.0/3) - 1.0))
        phi = rnd.uniform(0, 2*np.pi, N)
        theta = np.arccos(rnd.uniform(-1, 1, N))
        temp1 = []
        temp1.append(r*np.sin(theta)*np.cos(phi))   # x
        temp1.append(r*np.sin(theta)*np.sin(phi))   # y
        temp1.append(r*np.cos(theta))               # z
        self.Position = np.array(temp1).T

        #for i in range(N):
            #self.Velocity[i] =

        i = 0
        while(i<N):
            a = rnd.random()*2 -1
            b = rnd.random()*2 -1
            c = rnd.random()*2 -1
            if(a*a+b*b+c*c<1):
                t_ve =  self.escapeVelocity(i)
                self.Velocity[i][0] = a*t_ve
                self.Velocity[i][1] = b*t_ve
                self.Velocity[i][2] = c*t_ve
                i += 1
        #self.Velocity[0][0] = 0.1
        #self.Velocity[0][1] = -1/N
        print("constructor over")

        # Determining the timestep, either an array(for individual time step) or a number
        self.TimeStep = step
        self.acceleration()
        #self.jerk()
        #self.snap()
        #self.crackle()
        self.energy()
    def l(self):
        '''
        angular momentum
        '''
        temp1 = np.zeros(3)
        for i in range(self.N):
            temp1 += np.cross(self.Position[i], self.Velocity[i])
        self.L = self.Mass*temp1
        return self.Mass*temp1
    def energy(self):
        e = 0
        e = -self.G*self.Mass*self.Mass*np.sum(1.0/sp.spatial.distance.pdist(self.Position))
        e += self.Mass*(np.sum(self.Velocity**2))/2
        self.Energy = e
        return e

    def escapeVelocity(self,i):
        '''
        Escape velocity of ith particle
        '''
        return np.sqrt(2)*np.power(np.power(self.R(i), 2) + self.PlummerRadius, -1.0/4.0)
    def R(self,i):
        '''
        distance from origin of the ith particle
        '''
        return np.sqrt(np.sum(np.power(self.Position[i], 2)))
    def V(self,i):
        '''
        magnitude of velocity of the ith particle
        '''
        return np.sqrt(np.sum(np.power(self.Velocity[i], 2)))

    def accelerationi(self, i):
        '''
        return  the acceleration of the ith particle
        more precisely the softened acceleration
        '''
        tempa = np.zeros(3)
        for j in range(self.N):
            if( i!=j):
                tempr = self.Position[j]- self.Position[i]
                tempa += self.G*self.Mass*(tempr)/(tempr[0]**2 + tempr[1]**2 + tempr[2]**2 + self.Epsilon**2 )**(3.0/2.0)
        self.Acceleration[i] = tempa
        return tempa
    def acceleration(self):
        '''
        calculate the acceleration of the all the particles
        '''
        for i in range(self.N):
            self.accelerationi(i)

    def jerki(self, i):
        '''
        return  the jerk of the ith particle
        more precisely the softened jerk
        '''
        tempj = np.zeros(3)
        for j in range(self.N):
            if( i!=j):
                rij = self.Position[j] - self.Position[i]
                vij = self.Velocity[j] - self.Velocity[i]
                tempj += (vij)/(rij[0]**2 + rij[1]**2 + rij[2]**2 + self.Epsilon**2 )**(3.0/2.0)
                tempj -= 3*np.sum(vij*rij)*rij/(rij[0]**2 + rij[1]**2 + rij[2]**2 + self.Epsilon**2 )**(5.0/2.0)
        tempj *= self.G*self.Mass
        self.Jerk[i] = tempj
        return tempj
    def jerk(self):
        for i in range(self.N):
            self.jerki(i)

    def snapi(self, i):
        '''
        return  the snap of the ith particle
        more precisely the softened snap
        assuming that acceleration, jerk have been calculated
        '''
        temps = np.zeros(3)
        for j in range(self.N):
            if( i!=j):
                rij = self.Position[j] - self.Position[i]
                vij = self.Velocity[j] - self.Velocity[i]
                aij = self.Acceleration[j] - self.Acceleration[i]
                jij = self.Jerk[j] - self.Jerk[i]
                rd_ij = rij[0]**2 + rij[1]**2 + rij[2]**2 + self.Epsilon**2
                alpha = np.sum(rij*vij)/rd_ij
                beta = (np.sum(vij*vij) * np.sum(rij*aij))/rd_ij + alpha**2
                temps += aij/(rd_ij**(3.0/2))
                temps -= 6*alpha*jij
                temps -= 3*beta*aij
        temps *= self.G*self.Mass
        self.Snap[i] = temps
        return temps

    def snap(self):
        for i in range(self.N):
            self.snapi(i)

    def cracklei(self, i):
        '''
        return  the crackle of the ith particle
        more precisely the softened crackle
        assuming that acceleration, jerk, snap have been calculated
        '''
        temps = np.zeros(3)
        for j in range(self.N):
            if( i!=j):
                rij = self.Position[j] - self.Position[i]
                vij = self.Velocity[j] - self.Velocity[i]
                aij = self.Acceleration[j] - self.Acceleration[i]
                jij = self.Jerk[j] - self.Jerk[i]
                sij = self.Snap[j] - self.Snap[i]
                rd_ij = rij[0]**2 + rij[1]**2 + rij[2]**2 + self.Epsilon**2
                alpha = np.sum(rij*vij)/rd_ij
                beta = (np.sum(vij*vij) * np.sum(rij*aij))/rd_ij + alpha**2
                gamma = ( 3*np.sum(vij*aij) + np.sum(rij*jij))/rd_ij + alpha*(3*beta - 4 * (alpha**2))
                temps += jij/(rd_ij**(3.0/2))
                temps -= 9*alpha*sij
                temps -= 6*beta*jij
                temps -= 3*gamma*aij
        temps *= self.G*self.Mass
        self.Crackle[i] = temps
        return temps

    def crackle(self):
        for i in range(self.N):
            self.cracklei(i)



    ########################
    # integration schemes
    ########################
    def stepEuler(self):
        h = self.TimeStep
        self.acceleration()
        dv = self.Acceleration*h
        dx = dv*h
        self.Position += dx
        self.Velocity += dv
        self.Time += self.TimeStep

    def stepHermite4(self):
        r = np.copy(self.Position)
        v = np.copy(self.Velocity)
        a = np.copy(self.Acceleration)
        j = np.copy(self.Jerk)
        h = np.copy(self.TimeStep)
        # calculating the predicted values
        r_p = r + v*h + a*h*h/2 + j*h*h*h/6
        v_p = v + a*h + j*h*h/2
        self.Position = r_p
        self.Velocity = v_p

        # calculating the force
        self.acceleration()
        self.jerk()
        temp1 = a-self.Acceleration
        snap = -6*(temp1)/(h*h) - (4*j + 2*self.Jerk)/h
        crackle = -12*temp1/(h*h*h) - 6*(j+ self.Jerk)/(h*h)

        #correcting
        self.Position += snap*(h**4)/24 + crackle*(h**5)/120.0
        self.Velocity += snap*(h**3)/4 + crackle*(h**4)/24.0


        self.Time += self.TimeStep
        self.acceleration()
        self.jerk()
        self.Energy = self.energy()

    def stepHermite6(self):
        r = np.copy(self.Position)
        v = np.copy(self.Velocity)
        a = np.copy(self.Acceleration)
        j = np.copy(self.Jerk)
        s = np.copy(self.Snap)
        c = np.copy(self.Crackle)
        h = np.copy(self.TimeStep)

        # calculating the predicted values
        r_p = r + v*h + a*(h**2)/2 + j*(h**3)/6 + s*(h**4)/24 + c*(h**5)/120
        v_p = v + a*h + j*(h**2)/2 + s*(h**3)/6 + c*(h**4)/24
        a_p = a + j*h + s*(h**2)/2 + c*(h**3)/6
        self.Position = r_p
        self.Velocity = v_p
        self.Acceleration = a_p

        # calculating the force
        self.jerk()
        self.snap()
        self.crackle()

        # correcting
        self.Velocity = v + (a_p+a)*h/2 - (self.Jerk - j)*(h**2)/10 + (self.Snap + s)*(h**3)/120
        self.Position = r + (self.Velocity+v)*h/2 - (self.Acceleration - a)*(h**2)/10 + (self.Jerk + j)*(h**3)/120

        self.acceleration()
        self.jerk()
        self.snap()
        self.crackle()

        self.Time += self.TimeStep
        self.Energy = self.energy()
    def step(self):
        self.Time += self.TimeStep
        self.Energy = self.energy()
