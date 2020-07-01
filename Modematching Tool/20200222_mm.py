# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:34:14 2020

@author: schaecl
"""

import numpy as np
import matplotlib.pyplot as plt

def get_Modematching(w01, w02, lam, z):
    """
    Calculatutes the mode matching coefficient between two Gaussian beam of the
    same wavelength. One beam is assumed to be stationary of waist w01. The
    other beam is displaced by z and has a waist of w02.

    Parameters
    ----------
    w01 : float
        The waist of the stationary beam.
    w02 : float
        The waist of the distance dependent beam.
    lam : float
        The wavelength.
    z : float
        The distance the beam with w02 travels.

    Returns
    -------
    Mode matching coefficient.

    """
    res = 4 * np.pi**2 * \
    np.abs((w02 * w01 * np.sqrt(1 + (z**2 * lam**2) / (np.pi**2 * w02**4))) / \
           (np.pi * w01**2 + np.pi*w02**2 * (1 + (z**2 * lam**2) / (np.pi**2 * w02**4))))**2
    return res

# Formula: 2 * f * NAeffEsq / 2
w0N12NIR = 2 * 20e-3 * 0.096 / 2
w0RTAPO = 2 * 18e-3 * 0.089 / 2
w0Olympus = 2 * 10.5e-3 * 0.089 / 2

lam = 630e-9
z = 1.6
w0LTAPO = 4.4e-3 / 2

print("w0N12NIR is {:g} mm".format(w0N12NIR * 1e3))
print("Mode matching for 12NIR is {:g}".format(get_Modematching(w0LTAPO, w0N12NIR, lam, z)))
print("Mode matching for RTAPO is {:g}".format(get_Modematching(w0LTAPO, w0RTAPO, lam, z)))
print("Mode matching for Olympus is {:g}".format(get_Modematching(w0LTAPO, w0Olympus, lam, z)))

def getMaxFocallength():
    """ Calculates the mode matching for the given beamdiameter w01 of the objective for 
        different focal lengths (1..100 mm) of the collimator.
        Creates a plot and returns a list of the mode matching factors.

    Parameters
    ----------
    w01 : float
        The beam radius of the LT-APO in mm.

    Returns
    -------
    res_list : list
        List of the resulting focal lengths.

    """
    w01_list = []
    max_f_list = []
    w01=1.7e-3

    f = 1e-3
    i = 0.05e-3
    while w01<2.2e-3:
        f = 1e-3
        f_list = []
        res_list = []
        w02_list = []
        w01_list.append(w01*1e3)
        while f<30e-3:
            w02=2*f*0.089/2
            res = get_Modematching(w01,w02,lam,z)
            w02_list.append(w02)
            f_list.append(f)
            res_list.append(res)
            f = f+i
        plt.plot(f_list, res_list)
        plt.xlabel('Focal length')
        plt.ylabel('Mode matching')
        max_f = f_list[res_list.index(max(res_list))]
        max_f_list.append(max_f*1e3)
        w0 = w02_list[res_list.index(max(res_list))]
        print("The perfect focal length would be {:g} mm".format(max_f*1e3))
        print("with a mode matching coefficient of {:g}".format(max(res_list)))
        print("and a beamdiameter of {:g} mm".format(w0*1e3))
        w01 = w01 + 0.1e-3
    plt.figure()
    plt.plot(max_f_list, w01_list)
    plt.xlabel('Ideal Focallength')
    plt.ylabel('Beam diamaeter Objective $w_{01}$ in mm')
    return res_list

getMaxFocallength()

def getMMatFocallength(f):
    """ Calculates the mode matching for the given beamdiameter w01 of the objective for 
        different focal lengths (1..100 mm) of the collimator.
        Creates a plot and returns a list of the mode matching factors.

    Parameters
    ----------
    w01 : float
        The beam radius of the LT-APO in mm.

    Returns
    -------
    res_list : list
        List of the resulting focal lengths.

    """
    w01=1e-3
    f = f*1e-3
    w02=2*f*0.096/2

    f_list = []
    res_list = []
    w01_list = []

    while w01<2.5e-3:
        w01_list.append(w01*1e3)
        res = get_Modematching(w01,w02,lam,z)
        res_list.append(res)
        w01 = w01+0.1e-3
    plt.plot(w01_list, res_list)
    plt.xlabel('Beam diamaeter Objective $w_{01}$ in mm')
    plt.ylabel('Mode matching coeff')
    plt.plot(w01_list, 0.64)

    w01 = w01 + 0.1e-3
    return res_list
