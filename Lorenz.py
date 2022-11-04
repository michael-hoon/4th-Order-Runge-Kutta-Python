# importing required modules
import numpy as np
import matplotlib.pyplot as plt


# defining the system of lorenz equations 
def fx(x,y,z,t): return sigma*(y-x)     # time derivative of x
def fy(x,y,z,t): return x*(r-z)-y       # time derivative of y
def fz(x,y,z,t): return x*y-b*z         # time derivative of z


# defining constants in the lorenz equation
sigma = 10.0        # float sigma
b = 8.0 / 3.0       # float b
r = 350.0            # float r
t_in = 0.0          # float initial value of t
t_fin = 5.0        # float final value of t
h = 0.01            # float step size h
total_steps = int(np.floor((t_fin-t_in)/h))     # integer value of total number of steps in the Runge-Kutta method


# defining the classical Runge-Kutta 4th order method

def RungeKutta4(x,y,z,fx,fy,fz,t,h):
    """
    A function that returns the updated values of k1, k2, k3, k4 based on the 
    4th Order Runge-Kutta method for solving a nonlinear dynamical system. 
    Intermediate values for k1, k2, k3, k4 are computed at every sequential time step to obtain the subsequent values of x, y, and z.
    
    Parameters
    ----------
    x : array
        array values of x

    y : array
        array values of y

    z : array
        array values of z

    fx : float
        function definition for x time derivative 

    fy : float
        function definition for y time derivative

    fz : float
        function definition for z time derivative 

    t : array
        array values for t
    
    h : float   
        step size 
    
    """

    k1x, k1y, k1z = ( h*f(x,y,z,t) for f in (fx,fy,fz) )
    xs, ys, zs, ts = ( r+0.5*kr for r,kr in zip((x,y,z,t),(k1x,k1y,k1z,h)) )
    k2x, k2y, k2z = ( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )
    xs, ys, zs, ts = ( r+0.5*kr for r,kr in zip((x,y,z,t),(k2x,k2y,k2z,h)) )
    k3x, k3y, k3z = ( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )
    xs, ys, zs, ts = ( r+kr for r,kr in zip((x,y,z,t),(k3x,k3y,k3z,h)) )
    k4x, k4y, k4z  =( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )

    return (r+(k1r+2*k2r+2*k3r+k4r)/6 for r,k1r,k2r,k3r,k4r in 
            zip((x,y,z),(k1x,k1y,k1z),(k2x,k2y,k2z),(k3x,k3y,k3z),(k4x,k4y,k4z)))


t = total_steps * [0.0]
x = total_steps * [0.0]
y = total_steps * [0.0]
z = total_steps * [0.0]


x[0],y[0],z[0],t[0] = 0.0 , 1.0 , 0.0 , 0.0  # given initial conditions


# outputting array of values for x, y, and z
for i in range(1, total_steps):
    x[i],y[i],z[i] = RungeKutta4(x[i-1],y[i-1],z[i-1], fx,fy,fz, t[i-1], h)


# numpy array for timesteps of t to input as horizontal axis of the graph
t = np.arange(0, t_fin, h)


# plotting out y(t) as a function of time to show behaviour of y as t progresses
plt.plot(t, y)


# setting labels and title
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(f"sigma = {int(sigma)}, r = {int(r)}, b = {b}, \n initial conditions: [{x[0]}, {y[0]}, {z[0]}]")


# show the resulting plot
plt.show()


# # plotting solution for 3D view of Lorenz attractor (Just for fun).

# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')
# ax.grid()

# ax.plot3D(x, y, z)

# # Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)

# plt.show()