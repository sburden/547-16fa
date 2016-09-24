
# packages
import numpy as np
import scipy as sp
from scipy import linalg as la
import control as ctrl
import matplotlib as mpl
import pylab as plt

def D(g,y,d=1e-4):
  """
  approximate derivative via finite-central-differences 

  input:
    g - function - g : R^n -> R^m
    y - n array
    (optional)
    d - scalar - finite differences displacement parameter

  output:
    Dg(y) - m x n - approximation of Jacobian of g at y
  """
  # given $g:\mathbb{R}^n\rightarrow\mathbb{R}^m$:
  # $$D_y g(y)e_j \approx \frac{1}{2\delta}(g(y+\delta e_j) - g(y - \delta e_j)),\ \delta\ll 1$$
  e = np.identity(len(y))
  Dyg = []
  for j in range(len(y)):
      Dyg.append((.5/d)*(g(y+d*e[j]) - g(y-d*e[j])))
  return np.array(Dyg).T

def forward_euler(f,t,x,t0=0.,dt=1e-4,ut=None,ux=None,utx=None,return_u=False):
  """
  simulate x' = f(x,u) using forward Euler algorithm

  (pronounced "oiler")

  input:
    f : R x X x U --> X - vector field
      X - state space (must be vector space)
      U - control input set
    t - scalar - final simulation time
    x - initial condition; element of X

    (optional:)
    t0 - scalar - initial simulation time
    dt - scalar - stepsize parameter
    return_u - bool - whether to return u_

    (only one of:)
    ut : R --> U
    ux : X --> U
    utx : R x X --> U

  output:
    t_ - N array - time trajectory
    x_ - N x X array - state trajectory
    (if return_u:)
    u_ - N x U array - state trajectory
  """
  t_,x_,u_ = [t0],[x],[]
  
  inputs = sum([1 if u is not None else 0 for u in [ut,ux,utx]])
  assert inputs <= 1, "more than one of ut,ux,utx defined"

  if inputs == 0:
    assert not return_u, "no input supplied"
  else:
    if ut is not None:
      u = lambda t,x : ut(t)
    elif ux is not None:
      u = lambda t,x : ux(x)
    elif utx is not None:
      u = lambda t,x : utx(t,x)

  while t_[-1]+dt < t:
    if inputs == 0:
      _t,_x = t_[-1],x_[-1]
      dx = f(t_[-1],x_[-1]) * dt
    else:
      _t,_x,_u = t_[-1],x_[-1],u(t_[-1],x_[-1])
      dx = f(_t,_x,_u) * dt
      u_.append( _u )

    x_.append( _x + dx )
    t_.append( _t + dt )

  if return_u:
    return np.asarray(t_),np.asarray(x_),np.asarray(u_)
  else:
    return np.asarray(t_),np.asarray(x_)

def psi(f,t,x,t0=0.,dt=1e-4,ut=None,ux=None,utx=None):
  """
  simulate x' = f(x,u) using forward Euler algorithm, return final state

  input:
    f : R x X x U --> X - vector field
      X - state space (must be vector space)
      U - control input set
    t - scalar - final simulation time
    x - initial condition; element of X

    (optional:)
    t0 - scalar - initial simulation time
    dt - scalar - stepsize parameter

    (only one of:)
    ut : R --> U
    ux : X --> U
    utx : R x X --> U

  output:
    x(t) - X array - final state 
  """
  t_,x_ = forward_euler(f,t,x,t0=t0,dt=dt,ut=ut,ux=ux,utx=utx)
  return x_[-1]

def controllability(A,B):
  """
  controllability matrix of the pair (A,B)

  input:
    A - n x n 
    B - n x m

  output:
    C - n x (n*m) 
  """
  assert A.shape[0] == A.shape[1] # A is n x n
  assert A.shape[0] == B.shape[0] # B is n x m
  C = [B]
  for n in range(A.shape[0]):
    C.append( np.dot(A, C[-1]) )
  return np.hstack(C)

def controllable(A,B,eps=1e-3):
  """
  test controllability of the pair (A,B) for the LTI system  x' = A x + B u

  input:
    A - n x n 
    B - n x m
    (optional)
    eps - threshold on singular values of controllability matrix

  output:
    bool - controllable (with threshold eps)
  """
  C = controllability(A,B)
  _,s,_ = np.linalg.svd(C)
  return np.all( s > eps )

def observability(A,C):
  """
  observability matrix of the pair (A,C)

  input:
    A - n x n 
    C - m x n

  output:
    O - (n*m) x n
  """
  assert A.shape[0] == A.shape[1] # A is n x n
  assert A.shape[0] == C.shape[1] # C is m x n
  return controllability(A.T,C.T).T

def observable(A,C,eps=1e-3):
  """
  test observability of the pair (A,C) for the LTI system  x' = A x, y = C x

  input:
    A - n x n 
    C - m x n
    (optional)
    eps - threshold on singular values of observability matrix

  output:
    bool - observable (with threshold eps)
  """
  O = observability(A,C)
  _,s,_ = np.linalg.svd(O)
  return np.all( s > eps )

