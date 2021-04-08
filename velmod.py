from scipy import *
from numpy import *
import numpy as np
import numpy.fft as fft
import scipy.special as sp
# calculate velocity fields, decomposition of kinetic energies and spectrum
def gradient(psi,kx,ky):
    phi = fft.fft2(psi)
    cpx = fft.ifft2(1j*kx*phi)
    cpy = fft.ifft2(1j*(ky.T)*(phi))
    return cpx, cpy

def current(psi, kx, ky):
    cpx, cpy = gradient(psi,kx,ky)
    jx = imag(conj(psi)*cpx)
    jy = imag(conj(psi)*cpy)
    return jx, jy
    
def velocity(psi, x, y, kx, ky):
    rho = abs(psi)*abs(psi)
    cpx, cpy = gradient(psi,kx,ky)
    vx = ((imag(conj(psi)*cpx))/rho) 
    vx = np.nan_to_num(vx, nan=0, posinf=0, neginf=0)
    vy = ((imag(conj(psi)*cpy))/rho) 
    vy = np.nan_to_num(vy, nan=0, posinf=0, neginf=0)
    return vx, vy

def velocity_lab(psi, Omega, x, y, kx, ky):
    rho = abs(psi)*abs(psi)
    cpx, cpy = gradient(psi,kx,ky)
    # lab frame 
    vx = ((imag(conj(psi)*cpx))/rho) - Omega*y.T 
    vx = np.nan_to_num(vx, nan=0, posinf=0, neginf=0)
    # lab frame 
    vy = ((imag(conj(psi)*cpy))/rho) + Omega*x 
    vy = np.nan_to_num(vy, nan=0, posinf=0, neginf=0)
    return vx, vy

def velocity_rot(psi, Omega, x, y, kx, ky):
    rho = abs(psi)*abs(psi)
    cpx, cpy = gradient(psi,kx,ky)
    # Rot frame 
    vx = ((imag(conj(psi)*cpx))/rho) + Omega*y.T 
    vx = np.nan_to_num(vx, nan=0, posinf=0, neginf=0)
    # Rot frame 
    vy = ((imag(conj(psi)*cpy))/rho) - Omega*x   
    vy = np.nan_to_num(vy, nan=0, posinf=0, neginf=0)
    return vx, vy

def helmholtz(wx, wy, kx, ky, K2):
    wxk = fft.fft2(wx)
    wyk = fft.fft2(wy)
    kdotw = kx*wxk + (ky.T)*wyk
    wxkc = kdotw*kx/K2  #; wxkc[1] = 0
    wxkc = np.nan_to_num(wxkc, nan=0, posinf=0, neginf=0)
    wykc = kdotw*ky.T/K2 #; wykc[1] = 0
    wykc = np.nan_to_num(wykc, nan=0, posinf=0, neginf=0)
    wxki = wxk - wxkc
    wyki = wyk - wykc
    wxc = fft.ifft2(wxkc)
    wyc = fft.ifft2(wykc)
    wxi = fft.ifft2(wxki)
    wyi = fft.ifft2(wyki)
    Wi = (wxi, wyi)  
    Wc = (wxc, wyc)  
    return Wi, Wc

def energydecomp(psi, Omega, x, y, kx, ky, K2):
    rho = abs(psi)*abs(psi)
    vx, vy = velocity(psi, Omega, x, y, kx, ky)
    wx = np.sqrt(rho)*vx
    wy = np.sqrt(rho)*vy
    Wi, Wc = helmholtz(wx, wy, kx, ky, K2)
    wxi, wyi = Wi
    wxc, wyc = Wc
    et = (abs(wx)*abs(wx) + abs(wy)*abs(wy))
    et *= 0.5
    ei = (abs(wxi)*abs(wxi) + abs(wyi)*abs(wyi))
    ei *= 0.5
    ec = (abs(wxc)*abs(wxc) + abs(wyc)*abs(wyc))
    ec *= 0.5
    return et, ei, ec
      
def dfft(N,x,k):
    dx = (x[1,0]-x[0,0])/sqrt(2*pi)
    dy = dx
    dkx = len(k)*(k[1,0]-k[0,0])/sqrt(2*pi)
    dky = dkx
    return dx, dy, dkx, dky

def autocorrelate(psi, x, y, kx, ky):
    N = int(psi.shape[0])
    dx, dy, dkx, dky = dfft(N,x,kx)
    phi = np.pad(psi,int(N/2))
    chi = fft.fft2(phi)*dx*dy
    chi2 = abs(chi)**2
    xi = fft.ifft2(abs(chi)**2)*dkx*dky*(2*pi)**(2/2.)
    xis = fft.fftshift(xi)
    return xis

def bessel_reduce(kp, x, y, C):
    N = int(kp.shape[0])
    dx = dy = x[1,0]-x[0,0]
    Nx = 2*len(x)
    Lx = x[-1]- x[0]
    xp = np.linspace(-Lx, Lx, Nx).reshape(Nx,1)
    yp = xp
    rho = hypot(xp,yp.T)
    E = np.zeros(N)
    k = np.zeros(N)
    for i in range(len(k)):
                k = kp[i]
                E[i] = k*sum(real(sp.jv(0,k*rho)*C))*dx*dy/2/pi    
    return E

def incompressible(k, psi,x, y, kx, ky, K2):
    vx, vy = velocity(psi, x, y, kx, ky)
    wx = abs(psi)*vx
    wy = abs(psi)*vy
    Wi, Wc = helmholtz(wx, wy, kx, ky, K2)
    wxi, wyi = Wi
    cx = autocorrelate(wxi, x, y, kx, ky)
    cy = autocorrelate(wyi, x, y, kx, ky)
    C = 0.5*(cx+cy)
    Ek = bessel_reduce(k, x, y, C)
    return Ek
    
def compressible(k, psi,x, y, kx, ky, K2):
    vx, vy = velocity(psi, x, y, kx, ky)
    wx = abs(psi)*vx
    wy = abs(psi)*vy
    Wi, Wc = helmholtz(wx, wy, kx, ky, K2)
    wxc, wyc = Wc
    cx = autocorrelate(wxc, x, y, kx, ky)
    cy = autocorrelate(wyc, x, y, kx, ky)
    C = 0.5*(cx+cy)
    Ek = bessel_reduce(k, x, y, C)
    return Ek

#masking the data
def mask1(h, w, c=None, rad=None):  #c-center
    if c is None: 
        c = (int(w/2), int(h/2))
    if rad is None:
        rad = min(c[0], c[1], w-c[0], h-c[1])
    y, x = np.ogrid[:h, :w]
    dis_cen = np.sqrt((x - c[0])**2 + (y-c[1])**2)
    mask = dis_cen <= rad
    return mask
# log scale space   
def log10range(a,b,n):
    x = np.linspace(log10(a),log10(b),n)
    return  10**x