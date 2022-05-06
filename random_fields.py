# Main dependencies
import numpy
import scipy.fftpack
import numpy as np
import cv2

import torch
import math


def grf_idct_2d(Ln, alpha=2.0, tau=3.0):
    """
    Gaussian random field
    mean 0
    covariance operator C = (-Delta + tau^2)^(-alpha)
    Delta is the Laplacian with zero Neumann boundary condition
    """

    k = np.arange(Ln)
    K1,K2 = np.meshgrid(k,k)
    # Define the (square root of) eigenvalues of the covariance operator
    C = (np.pi**2)*(np.square(K1)+np.square(K2))+tau**2
    C = np.power(C,-alpha/2.0)
    C = (tau**(alpha-1))*C
    # # sample from normal discribution
    xr = np.random.standard_normal(size=(Ln,Ln))
    # coefficients in fourier domain
    L= C*xr
    L= Ln*L
    # apply boundary condition
    L[0,0] = 0.0
    # transform to real domain
    u = cv2.idct(L)
    print("max: ", u.max())
    print("min: ", u.min())

    return u



import numpy as np
import cv2

class GaussianRF_idct(object):
    """
    Gaussian random field Non-Periodic Boundary
    mean 0
    covariance operator C = (-Delta + tau^2)^(-alpha)
    Delta is the Laplacian with zero Neumann boundary condition
    """

    def __init__(self, dim, Ln, alpha=2.0, tau=3.0, device=None):
        self.dim = dim
        self.Ln = Ln
        self.device = device
        k = np.arange(Ln)
        K1,K2 = np.meshgrid(k,k)
        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2)*(np.square(K1)+np.square(K2))+tau**2
        C = np.power(C,-alpha/2.0)
        C = (tau**(alpha-1))*C
        # store coefficient
        self.coeff = C

    def _sample2d(self):
        """
        Single 2D Sample
        :return: GRF numpy.narray (Ln,Ln)
        """
        # # sample from normal discribution
        xr = np.random.standard_normal(size=(self.Ln,self.Ln))
        # coefficients in fourier domain
        L= self.coeff*xr
        L= self.Ln*L
        # apply boundary condition
        L[0,0] = 0.0
        # transform to real domain
        u = cv2.idct(L)
        return u

    def sample(self, N, mul=1):
        z_mat = np.zeros((N,self.Ln, self.Ln), dtype=np.float32)
        for ix in range(N):
            z_mat[ix,:,:] = self._sample2d()
        # convert to torch tensor
        z_mat = torch.from_numpy(z_mat)
        if self.device is not None:
            z_mat = z_mat.to(self.device)
        return z_mat



class GaussianRF_odd(object):

    def __init__(self, dim, size, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N, mul=1):

        coeff = torch.randn(N, *self.size, 2, device=self.device)*mul

        coeff[...,0] = self.sqrt_eig*coeff[...,0] #real
        coeff[...,1] = self.sqrt_eig*coeff[...,1] #imag

        ##########torch 1.7###############
        #u = torch.ifft(coeff, self.dim, normalized=False)
        #u = u[...,0]
        ##################################

        #########torch latest#############
        coeff_new = torch.complex(coeff[...,0],coeff[...,1])
        #print(coeff_new.size())
        u = torch.fft.ifft2(coeff_new, dim = (-2,-1), norm=None)

        u = u.real


        return u

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N, mul=1):

        coeff = torch.randn(N, *self.size, 2, device=self.device)*mul

        coeff[...,0] = self.sqrt_eig*coeff[...,0] #real
        coeff[...,1] = self.sqrt_eig*coeff[...,1] #imag

        ##########torch 1.7###############
        #u = torch.ifft(coeff, self.dim, normalized=False)
        #u = u[...,0]
        ##################################

        #########torch latest#############
        coeff_new = torch.complex(coeff[...,0],coeff[...,1])
        #print(coeff_new.size())
        u = torch.fft.ifft2(coeff_new, dim = (-2,-1), norm=None)

        u = u.real


        return u

