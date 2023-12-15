"""
This is the age_model.py module, to invert isochronal layers along a radar profile.
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import os
import sys
import yaml
from scipy.optimize import least_squares

from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages


class RadarLine(object):

    def __init__(self,label):

        self.label = label
        self.line_name = label.split("/")[-2]

    # define default parameters
    def default_params(self):
                self.is_bedelev = False
                self.is_trace = False
                self.is_basal = False
                self.nbiso = 0
                self.nbdsz = 0
                self.nbhor = 0
                self.calc_sigma = True
                self.settick = 'auto'
                self.interp_method = 'lin_aver'
                self.distance_unit = 'km'
                self.firn_correction = 14.6
                self.resolution = 1.
                self.is_EDC = False
                self.is_BELDC = False
                self.calc_isoage = False
                self.distance_EDC = 0.
                self.distance_BELDC = 0.
                self.distance_start = 'auto'
                self.distance_end = 'auto'
                self.max_depth = 'auto'
                self.iso_accu_sigma = 0.
                self.age_surf = -50.
                self.dzeta = 0.01
                self.grid_type = 'regular'
                self.ratio = 0.1
                self.p_prior = 2
                self.p_sigma = 5
                self.EDC_line_dashed = False
                self.is_NESW = False
                self.reverse_distance = False
                self.aspect = 0.028
                self.is_legend = True
                self.min_tick = 0.
                self.max_tick = 100.
                self.delta_tick = 10.
                self.invert_thk = False
                self.invert_s = False
                self.s = 0.
                self.opt_method = 'leastsq1D'
                self.accu_step = 0.001
                self.p_step = 0.5
                self.thick_step = 10.
                self.bad_fit = False
                self.init_bed = 4000.
                self.nans = True
                self.bed_nans=True      # include nans
                self.max_age = 1500     # in kyr
                self.place = 'Dome C'
                self.res_max = 20000    # in kyr m-1, max resultion for bottom age
                self.is_core = False

    # get parameters from yaml file
    def load_parameters(self):

        # parameters for all radar lines (file is mandatory - will overwrite default params)
        data = yaml.load(open(self.label+'../parameters_all_radar_lines.yml').read(),
                         Loader=yaml.FullLoader)
        if data != None:
            self.__dict__.update(data)

        # parameters for individual radar lines (optional - will overwrite general params)
        filename = self.label+'parameters.yml'
        if os.path.isfile(filename):
            data = yaml.load(open(filename).read(), Loader=yaml.FullLoader)
            if data != None:
                self.__dict__.update(data)

    # load data from .txt or .dat file and make class varibles
    def load_radar_data(self):

        #Reading the radar horizon dataset
        nbcolumns = 6+self.nbiso+self.is_bedelev+self.is_trace+self.is_basal
        filename = self.label+'radar-data.txt'
        if os.path.isfile(filename):
            readarray = np.loadtxt(filename, usecols=range(nbcolumns),
                                   skiprows=1)
        else:
            readarray = np.loadtxt(self.label+'radar-data.dat', usecols=range(nbcolumns),
                                   skiprows=1)

        # basic parameters from radar data file
        if readarray[0, 4] > readarray[-1, 4]:
            readarray = readarray[::-1, :]
        self.LON_raw = readarray[:, 0]
        self.LAT_raw = readarray[:, 1]
        self.x_raw = readarray[:, 2]
        self.y_raw = readarray[:, 3]
        self.distance_raw = readarray[:, 4]
        if self.distance_unit == 'm':
            self.distance_raw = self.distance_raw/1000.
        self.thk_raw = readarray[:, 5]

        # parameters for analysis
        index = 6
        if self.is_bedelev:
            self.bedelev = readarray[:, index]
            index = index+1
        if self.is_trace:
            self.trace = readarray[:, index]
            index = index+1
        if self.is_basal:
            self.basal_raw = readarray[:, index]
            index = index+1
        self.iso_raw = np.transpose(readarray[:, index:index+self.nbiso])
        index = index+self.nbiso

        # set start and end points where there are at least 2 non nan isochrones
        non_nans = np.array([np.count_nonzero(~np.isnan((self.iso_raw[:,i]).flatten())) for i in range(len(self.distance_raw))])
        if self.distance_start == 'auto':
            self.distance_start = self.distance_raw[np.argmax(non_nans > 2)]
        if self.distance_end == 'auto':
            self.distance_end = self.distance_raw[len(self.distance_raw) - np.argmax(non_nans[::-1] > 2)-1]-self.resolution

        self.distance = np.arange(self.distance_start, self.distance_end+self.resolution,
                                  self.resolution)

    # Linear interpolation fuction
    def interp1d_lin_aver(self,x, y, left=np.nan, right=np.nan, nans=True):
        """
        Interpolation of a linear by parts function using averaging.
        This function returns nan when there are all nans in one interpolation interval.
        FIXME: there is a problem in this routine when the x are in decreasing order.
        """
        def f(xp):
            yp = np.empty(np.size(xp)-1)
            for i in range(np.size(xp)-1):
                xmod = x[~(np.isnan(x)+np.isnan(y))]
                ymod = y[~(np.isnan(x)+np.isnan(y))]
                xmod2 = xmod[np.where((xmod > xp[i])*(xmod < xp[i+1]))]
                ymod2 = ymod[np.where((xmod > xp[i])*(xmod < xp[i+1]))]
                xmod3 = np.concatenate((np.array([xp[i]]), xmod2, np.array([xp[i+1]])))
                if nans:
                    y1 = np.interp(xp[i:i+2], x, y, right=right, left=left)
                else:
                    y1 = np.interp(xp[i:i+2], xmod, ymod, right=right, left=left)  #left=right
                ymod3 = np.concatenate((np.array([y1[0]]), ymod2, np.array([(y1[1])])))
                if np.isnan(ymod3).all():
                    yp[i] = np.nan
                else:
                    xmod4 = xmod3[np.where(~(np.isnan(ymod3)+np.isnan(xmod3)))]
                    ymod4 = ymod3[np.where(~(np.isnan(ymod3)+np.isnan(xmod3)))]
                    yp[i] = np.sum((ymod4[1:]+ymod4[:-1])/2*(xmod4[1:]-xmod4[:-1]))
                    yp[i] = yp[i]/(xmod4[-1]-xmod4[0])
            return yp
        return f

    # spacial interpolation of ice thicknesses and isochrone depths
    def interp_radar_data(self):

        # interpolate ice thickness
        f = self.interp1d_lin_aver(self.distance_raw, self.thk_raw, nans=self.bed_nans)
        self.thkreal = f(np.concatenate((self.distance-self.resolution/2,
                                         np.array([self.distance[-1]+self.resolution/2]))))
        self.thk = np.where(np.isnan(self.thkreal),self.init_bed,self.thkreal)

        # array to store interpolated data
        self.iso = np.zeros((self.nbiso, np.size(self.distance)))
        self.iso_modage = np.empty_like(self.iso)
        self.iso_modage_sigma = np.empty_like(self.iso)
        self.iso_EDC = np.zeros(self.nbiso)

        # interp basal unit if present
        if self.is_basal:
            f = self.interp1d_lin_aver(self.distance_raw, self.basal_raw, nans=self.nans)
            self.basal = f(np.concatenate((self.distance-self.resolution/2,
                                             np.array([self.distance[-1]+self.resolution/2]))))
        # interp isochrone data
        for i in range(self.nbiso):
            f = self.interp1d_lin_aver(self.distance_raw, self.iso_raw[i, :], nans=self.nans)
            self.iso[i, :] = f(np.concatenate((self.distance-self.resolution/2,
                                               np.array([self.distance[-1]+self.resolution/2]))))
        # get coords of distance nodes
        self.LON = np.interp(self.distance,self.distance_raw, self.LON_raw)
        self.LAT = np.interp(self.distance,self.distance_raw, self.LAT_raw)

        # INCOMPLETE: for use when isochrones are given in twtt
        self.LON_twtt = np.empty_like(self.distance)
        self.LAT_twtt = np.empty_like(self.distance)
        for j in range(np.size(self.distance)):
            self.LON_twtt[j] = self.LON_raw[np.argmin(np.absolute(self.LON_raw-self.LON[j]) +\
                               np.absolute(self.LAT_raw-self.LAT[j]))]
            self.LAT_twtt[j] = self.LAT_raw[np.argmin(np.absolute(self.LON_raw-self.LON[j]) +\
                               np.absolute(self.LAT_raw-self.LAT[j]))]

    # Reading the AICC2012 dataset, calculation of steady age and interpolation
    def iso_data(self):

        # load AICC2012 data
        readarray = np.loadtxt(self.label+'../AICC2012.txt')
        self.AICC2012_depth = readarray[:, 0]
        self.AICC2012_iedepth = readarray[:, 1]
        self.AICC2012_accu = readarray[:, 2]
        self.AICC2012_age = readarray[:, 3]
        self.AICC2012_sigma = readarray[:, 4]

        # calc AICC2012 accu variation ratio
        self.AICC2012_averageaccu = np.sum((self.AICC2012_age[1:] -\
                                    self.AICC2012_age[:-1])*self.AICC2012_accu[:-1])/\
                                    (self.AICC2012_age[-1]-self.AICC2012_age[0])
        print('average accu: ', self.AICC2012_averageaccu)
        self.AICC2012_steadyage = np.cumsum(np.concatenate((np.array([self.AICC2012_age[0]]),\
                                  (self.AICC2012_age[1:]-self.AICC2012_age[:-1])*\
                                  self.AICC2012_accu[:-1]/self.AICC2012_averageaccu)))
        print('steady/unsteady ratio: ', self.AICC2012_steadyage[-1]/self.AICC2012_age[-1])

        # if isochrone ages are not given they can be caluclated by linking to EDC
        if (self.is_EDC and self.calc_isoage):

            # for each isochrone get EDC depth
            for i in range(self.nbiso):
                self.iso_EDC[i] = np.interp(self.distance_EDC, self.distance_raw, self.iso_raw[i, :])

            # depth error
            self.z_err = np.loadtxt(self.label+'z-err.txt')
            # isochrone age
            self.iso_age = np.interp(self.iso_EDC, self.AICC2012_depth, self.AICC2012_age)
            self.iso_age = np.transpose([self.iso_age])
            # age uncertainty
            self.iso_sigma1 = (np.interp(self.iso_EDC+self.z_err, self.AICC2012_depth, self.AICC2012_age) - \
                                np.interp(self.iso_EDC-self.z_err, self.AICC2012_depth, self.AICC2012_age)) / 2.
            # depth uncertainty
            self.iso_sigma2 = np.interp(self.iso_EDC, self.AICC2012_depth, self.AICC2012_sigma)
            # combined uncertainty
            self.iso_sigma = np.sqrt(self.iso_sigma1**2+self.iso_sigma2**2)
            self.iso_sigma = np.transpose([self.iso_sigma])

            # accumulation uncertainty
            self.iso_accu_sigma = np.zeros((self.nbiso, 1))
            self.iso_accu_sigma[0] = self.iso_sigma[0]/(self.iso_age[0]-self.age_surf)
            self.iso_accu_sigma[1:] = np.sqrt(self.iso_sigma[1:]**2+self.iso_sigma[:-1]**2)/\
                                      (self.iso_age[1:]-self.iso_age[:-1])

            # save isochrone ages and uncertainties
            output = np.hstack((self.iso_age, self.iso_sigma, self.iso_accu_sigma))
            with open(self.label+'ages.txt', 'w') as f:
                f.write('#age (yr BP)\tsigma_age (yr BP)\tsigma_accu\n')
                np.savetxt(f, output, delimiter="\t")


        #Reading ages of isochrones and their sigmas
        if os.path.isfile(self.label+'../ages.txt'):                # general age file
            readarray = np.loadtxt(self.label+'../ages.txt')
        if os.path.isfile(self.label+'ages.txt'):                   # specific to individual radar line
            readarray = np.loadtxt(self.label+'ages.txt')
        self.iso_age = np.transpose([readarray[:, 0]])
        self.iso_age = self.iso_age[0:self.nbiso]
        self.iso_sigma = np.transpose([readarray[:, 1]])
        self.iso_sigma = self.iso_sigma[0:self.nbiso]

        # interpolated observed isochrone steady ages and sigmas
        self.iso_steadyage = np.interp(self.iso_age, self.AICC2012_age, self.AICC2012_steadyage)
        self.iso_steadysigma = np.sqrt((np.interp(self.iso_age+self.iso_sigma, self.AICC2012_age, self.AICC2012_steadyage) - self.iso_steadyage)*\
                            (self.iso_steadyage - np.interp(self.iso_age-self.iso_sigma, self.AICC2012_age, self.AICC2012_steadyage)))

    # arrays to store parameters
    def init_arrays(self):

        # arrys used while running model
        self.a = self.a*np.ones(np.size(self.distance))         # accumulation
        self.p_prime = m.log(self.p_prior+1)*np.ones(np.size(self.distance))         # omega_D parameter
        self.p = self.p_prior*np.ones(np.size(self.distance))         # omega_D parameter
        self.s = self.s*np.ones(np.size(self.distance))         # omega parameter
        self.thkie = np.empty_like(self.distance)               # thickness ice extrapolated
        self.zetanodes = self.grid()                            # normalised elevation nodes
        self.zeta = np.ones((np.size(self.zetanodes), np.size(self.distance)))*self.zetanodes
        self.zetaie = np.empty_like(self.zeta)
        self.depth = np.empty_like(self.zeta)
        self.depthie = np.empty_like(self.zeta)
        self.D = np.empty_like(self.zeta[:-1, :])               # ratio of total ice extrapolated depth : actual depth
        self.agesteady = np.zeros((np.size(self.zetanodes), np.size(self.distance)))
        self.age = np.zeros((np.size(self.zetanodes), np.size(self.distance)))
        self.age_density = np.zeros((np.size(self.zetanodes)-1, np.size(self.distance)))
        self.dist = np.ones((np.size(self.zetanodes), np.size(self.distance)))*self.distance
        self.omega_D = np.empty_like(self.age)
        self.omega = np.empty_like(self.age)
        self.tau = np.empty_like(self.age)
        self.m = np.empty_like(self.distance)
        self.resi_sd = np.empty_like(self.distance)
        self.bic = np.empty_like(self.distance)
        self.niso = np.empty_like(self.distance)
        self.sigma_a = np.zeros_like(self.distance)
        self.sigma_h = np.zeros_like(self.distance)
        self.sigma_p = np.zeros_like(self.distance)
        self.sigma_m = np.zeros_like(self.distance)
        self.sigma_age = np.zeros_like(self.age)
        self.sigma_logage = np.zeros_like(self.age)
        # arrays to store results
        self.agebot = np.empty_like(self.distance)
        self.realagebot = np.empty_like(self.distance)
        self.agebot10kyrm = np.empty_like(self.distance)
        self.agebot15kyrm = np.empty_like(self.distance)
        self.age100m = np.empty_like(self.distance)
        self.age150m = np.empty_like(self.distance)
        self.age200m = np.empty_like(self.distance)
        self.age250m = np.empty_like(self.distance)
        self.height0dot6Myr = np.nan*np.ones_like(self.distance)
        self.height0dot8Myr = np.nan*np.ones_like(self.distance)
        self.height1Myr = np.nan*np.ones_like(self.distance)
        self.height1dot2Myr = np.nan*np.ones_like(self.distance)
        self.height1dot5Myr = np.nan*np.ones_like(self.distance)
        self.twtt0dot6Myr = np.nan*np.ones_like(self.distance)
        self.twtt0dot8Myr = np.nan*np.ones_like(self.distance)
        self.twtt1Myr = np.nan*np.ones_like(self.distance)
        self.twtt1dot2Myr = np.nan*np.ones_like(self.distance)
        self.twtt1dot5Myr = np.nan*np.ones_like(self.distance)
        self.sigmabotage = np.empty_like(self.distance)
        self.age_density1Myr = np.nan*np.ones_like(self.distance)
        self.age_density1dot2Myr = np.nan*np.ones_like(self.distance)
        self.age_density1dot5Myr = np.nan*np.ones_like(self.distance)
        self.res_index = np.empty_like(self.distance)
        self.depth_res_max = np.empty_like(self.distance)
        self.depth_max = np.empty_like(self.distance)
        self.age_res_max = np.empty_like(self.distance)
        self.twttBed = np.nan*np.ones_like(self.distance)
        self.agebotmin = np.empty_like(self.distance)
        self.agebotmax = np.empty_like(self.distance)
        self.height1dot5Myrmin = np.empty_like(self.distance)
        self.height1dot5Myrmax = np.empty_like(self.distance)
        if not self.is_basal:
            self.basal = np.empty(len(self.distance))
            self.basal[:] = np.nan

    # generate zeta grid (linear or regular)
    def grid(self):
        start = 0                   # normalised z coord, start is the bottom, end is surface
        end = 1
        try:
            nb_steps = int(np.floor(1/self.dzeta))                  # grid seperated into this number of steps
        except KeyError:
            nb_steps = m.floor((end-start)/self.resolution)
            end = start + self.resolution * nb_steps
        if self.grid_type == 'regular':                             # grid lines have equal vertical seperation dz
            eps = (end-start)/nb_steps/2
            grid = np.arange(start, end+eps, (end-start)/nb_steps)
        elif self.grid_type == 'linear':                            # grid lines have vertical seperation with decreases with depth
            eps = (1.-self.ratio)/nb_steps
            grid = np.arange(self.ratio, 2.-self.ratio+eps, (2.-2*self.ratio)/(nb_steps-1))
            grid = grid * (end-start)/nb_steps
            grid = np.cumsum(np.concatenate((np.array([start]), grid)))
        else:
            print('Type of grid not recognized.')

        grid = grid[::-1]
        grid = np.transpose([grid])
        return grid

    # find nearest node below selected depths
    def find_nearest(self,array,searchvals):
        """
        takes search array and array of values to find nearest indices below searchvals
        """
        indices = np.array([], dtype=int)
        for val in searchvals:
            diff = array-val
            diff[diff>0] = -np.inf
            index = diff.argmax()
            if m.isnan(val):
                index = np.nan
            indices = np.append(indices,index)
        return indices

    # forward model
    def model1D_1order(self, j):

        # ie means ice extrapolation so if firn was also ice with density 1
        # thk is total height from bedrock to surface
        self.thkie[j] = np.interp(self.thk[j], np.concatenate((self.AICC2012_depth,\
                        np.array([self.AICC2012_depth[-1]+1e10]))), np.concatenate((\
                        self.AICC2012_iedepth, np.array([self.AICC2012_iedepth[-1]+1e10]))))

        # depth is measured from surface (depth=0) down
        self.depth[:, j] = self.thk[j]*(1-self.zeta[:, j])
        self.depthie[:, j] = np.interp(self.depth[:, j], np.concatenate((self.AICC2012_depth,\
                             np.array([self.AICC2012_depth[-1]+1e10]))), np.concatenate((\
                             self.AICC2012_iedepth, np.array([self.AICC2012_iedepth[-1]+1e10]))))

        # zeta is measured from bedrock (zeta=0) up
        self.zetaie[:, j] = (self.thkie[j]-self.depthie[:, j])/self.thkie[j]

        # ratio of total ice extrapolated depth : actual depth
        self.D[:, j] = (self.depthie[1:, j]-self.depthie[:-1, j])/(self.depth[1:, j]-\
                       self.depth[:-1, j])

        # p affects linearity of omega - greater p, omega more linear
        self.p[j] = m.exp(self.p_prime[j])-1

        # calc omega which affects thinning function
        self.omega_D[:, j] = 1-(self.p[j]+2)/(self.p[j]+1)*(1-self.zetaie[:, j])+1/\
                             (self.p[j]+1)*(1-self.zetaie[:, j])**(2+self.p[j])

        #Parrenin et al. (CP, 2007a) 2.2 (2)
        self.omega[:, j] = self.s[j]*self.zetaie[:, j]+(1-self.s[j])*self.omega_D[:, j]
        self.tau[:, j] = self.omega[:, j]   # since no melting in this model, thining = omega

        # new non linear tau
        self.age_density[:-1, j] = np.where( self.tau[1:-1, j]>0, 1/self.a[j]*(1/ self.tau[1:-1, j] + 1/self.tau[:-2, j])/2, np.nan)
        self.age_density[-1, j] = np.nan

        # cumulative sum over all zeta, age steady = age density* depth
        self.agesteady[:, j] = np.cumsum(np.concatenate((np.array([self.age_surf]),\
                               (self.depthie[1:, j]-self.depthie[:-1, j])*\
                               self.age_density[:, j])), axis=0)

        # age interpolated from descrete lines of age steady
        self.age[:, j] = np.interp(self.agesteady[:, j], np.concatenate((np.array([-1000000000]),\
                         self.AICC2012_steadyage, np.array([1e9*self.AICC2012_steadyage[-1]]))),\
                         np.concatenate((np.array([self.AICC2012_age[0]]), self.AICC2012_age,\
                         np.array([1e9*self.AICC2012_age[-1]]))))

        # get args of depth nodes closest to and smaller than isochrone depths
        closest_i = self.find_nearest(self.depth[:, j],self.iso[:, j])
        # interpolate 1/tau for isochrone depths
        tau_inv_interp = np.interp(self.iso[:, j], self.depth[:-1, j], 1/self.tau[:-1,j])
        # calc steady using 1/tau linear interpolation
        self.iso_modsteadyage=np.copy(self.iso_modage)
        mask = (~np.isnan(closest_i))
        self.iso_modsteadyage[mask,j] = self.agesteady[np.intc(closest_i[mask]),j] + \
                       ( tau_inv_interp[mask] + 1/self.tau[np.intc(closest_i[mask]), j] )/2 * \
                       ( self.iso[mask, j] - self.depth[np.intc(closest_i[mask]),j] ) *1/self.a[j]
        self.iso_modsteadyage[~mask,j] = closest_i[~mask]

        # interpolate real age
        self.iso_modage[:, j] = np.interp(self.iso_modsteadyage[:, j], np.concatenate((np.array([-1000000000]),\
                   self.AICC2012_steadyage, np.array([1e9*self.AICC2012_steadyage[-1]]))),\
                   np.concatenate((np.array([self.AICC2012_age[0]]), self.AICC2012_age,\
                   np.array([1e9*self.AICC2012_age[-1]]))))

        # melt rate
        self.m[j] = self.a[j]* np.interp(self.thkreal[j], self.depth[:,j], self.omega[:,j], right=0)

        # return accumulation, melting, p, age and geothermal flux
        return np.concatenate((np.array([self.a[j]]), np.array([self.thk[j]]),\
               np.array([self.p[j]]), self.age[:, j], np.log(self.age[1:, j]-\
               self.age_surf),np.array([self.m[j]])))

    # interpolate model outputs useful for graphs
    def model1D_finish(self, j):

        # resolution threshold for bottom age
        if np.nanmax(self.age_density[:,j]) > self.res_max:
            self.res_index[j] = next(x[0] for x in enumerate(self.age_density[:,j]) if x[1] > self.res_max)            # find index of first element above threshold
        else:
            self.res_index[j] = len(self.age_density[:,j])
        self.depth_res_max[j] = np.interp(self.res_max, self.age_density[:int(self.res_index[j]),j], self.depth[:int(self.res_index[j]),j])
        self.age_res_max[j] = np.interp(self.res_max, self.age_density[:int(self.res_index[j]),j], self.age[:int(self.res_index[j]),j])
        self.depth_max[j] = min(self.depth_res_max[j], self.thkreal[j])
        # bottom age either at bedrock or at max resolution depth
        self.agebot[j] = np.interp(self.depth_max[j], self.depth[:, j], self.age[:, j])
        self.realagebot[j] = np.interp(min(self.thk[j], self.thkreal[j]), self.depth[:, j], self.age[:, j])
        self.age100m[j] = np.interp(min(self.thk[j], self.thkreal[j])-100, self.depth[:, j], self.age[:, j])
        self.age150m[j] = np.interp(min(self.thk[j], self.thkreal[j])-150, self.depth[:, j], self.age[:, j])
        self.age200m[j] = np.interp(min(self.thk[j], self.thkreal[j])-200, self.depth[:, j], self.age[:, j])
        self.age250m[j] = np.interp(min(self.thk[j], self.thkreal[j])-250, self.depth[:, j], self.age[:, j])

        self.agebot10kyrm[j] = np.interp(10000., self.age_density[:, j],
                         (self.age[:-1, j]+self.age[1:,j])/2)
        self.agebot15kyrm[j] = np.interp(15000., self.age_density[:, j],
                         (self.age[:-1, j]+self.age[1:,j])/2)
        if self.agebot10kyrm[j] > self.realagebot[j]:
            self.agebot10kyrm[j] = np.nan
        if self.agebot15kyrm[j] > self.realagebot[j]:
            self.agebot15kyrm[j] = np.nan


        if self.agebot[j] >= 1000000.:
            self.age_density1Myr[j] = np.interp(1000000., (self.age[:-1,j]+self.age[1:,j])/2,
                                self.age_density[:,j])
        else:
            self.age_density1Myr[j] = np.nan
        if self.agebot[j] >= 1200000.:
            self.age_density1dot2Myr[j] = np.interp(1200000., (self.age[:-1,j]+self.age[1:,j])/2,
                                self.age_density[:,j])
        else:
            self.age_density1dot2Myr[j] = np.nan
        if self.agebot[j] >= 1500000.:
            self.age_density1dot5Myr[j] = np.interp(1500000., (self.age[:-1,j]+self.age[1:,j])/2,
                                self.age_density[:,j])
        else:
            self.age_density1dot5Myr[j] = np.nan

        if max(self.age[:, j]) >= 600000:
            self.height0dot6Myr[j] = self.thk[j]-np.interp(600000, self.age[:, j], self.depth[:, j])
            self.twtt0dot6Myr[j] = (np.interp(600000, self.age[:, j], self.depth[:, j])-self.firn_correction)*100/84.248+250.
        else:
            self.height0dot6Myr[j] = np.nan
            self.twtt0dot6Myr[j] = -98765.0
        if max(self.age[:, j]) >= 800000:
            self.height0dot8Myr[j] = self.thk[j]-np.interp(800000, self.age[:, j], self.depth[:, j])
            self.twtt0dot8Myr[j] = (np.interp(800000, self.age[:, j], self.depth[:, j])-self.firn_correction)*100/84.248+250.
        else:
            self.height0dot8Myr[j] = np.nan
            self.twtt0dot8Myr[j] = -98765.0
        if max(self.age[:, j]) >= 1000000:
            self.height1Myr[j] = self.thk[j]-np.interp(1000000, self.age[:, j], self.depth[:, j])
            self.twtt1Myr[j] = (np.interp(1000000, self.age[:, j], self.depth[:, j])-self.firn_correction)*100/84.248+250.
        else:
            self.height1Myr[j] = np.nan
            self.twtt1Myr[j] = -98765.0
        if max(self.age[:, j]) >= 1200000:
            self.height1dot2Myr[j] = self.thk[j]-np.interp(1200000, self.age[:, j], self.depth[:, j])
            self.twtt1dot2Myr[j] = (np.interp(1200000, self.age[:, j], self.depth[:, j])-self.firn_correction)*100/84.248+250.
        else:
            self.height1dot2Myr[j] = np.nan
            self.twtt1dot2Myr[j] = -98765.0
        if max(self.age[:, j]) >= 1500000:
            self.height1dot5Myr[j] = self.thk[j]-np.interp(1500000, self.age[:, j], self.depth[:, j])
            self.twtt1dot5Myr[j] = (np.interp(1500000, self.age[:, j], self.depth[:, j])-self.firn_correction)*100/84.248+250.
        else:
            self.height1dot5Myr[j] = np.nan
            self.twtt1dot5Myr[j] = -98765.0
        #TODO: make a function to convert to twtt, and make an array for the different isochrones.
        self.twttBed[j] = (self.thk[j]-self.firn_correction)*100/84.248+250.

    # Residuals function
    def residuals1D(self, variables1D, j):

        var = variables1D+0.0
        # seperate variables to appropriate array
        self.a[j] = var[0]
        var = np.delete(var, [0])
        self.p_prime[j] = var[0]
        var = np.delete(var, [0])
        if self.invert_thk:
            self.thk[j] = m.exp(var[0])
            if self.thk[j] < self.iso[-1,j]:       # check bedrock is deeper than isochrones
                self.thk[j] = self.iso[-1,j]
            var = np.delete(var, [0])
        if self.invert_s:
            self.s[j] = var[0]
            var = np.delete(var, [0])
        # run forward model
        self.model1D_1order(j)
        # calculate residuals
        resi = (np.log(self.iso_steadyage.flatten())-np.log(self.iso_modsteadyage[:, j]))/self.iso_steadysigma.flatten()*self.iso_steadyage.flatten()
        self.age_resi = resi[np.where(~np.isnan(resi))]
        resi = np.concatenate((self.age_resi, np.array([(self.p_prime[j]-m.log(self.p_prior+1))/\
               self.p_prime_sigma])))

        return resi

    # get jacobian for given index
    def jacobian1D(self, j):
        epsilon = np.sqrt(np.diag(self.hess1D))/10000000000.
        model0 = self.model1D_1order(j)
        jacob = np.empty((np.size(self.variables1D), np.size(model0)))
        for i in np.arange(np.size(self.variables1D)):
            self.variables1D[i] = self.variables1D[i]+epsilon[i]
            self.residuals1D(self.variables1D, j)
            model1 = self.model1D_1order(j)
            self.variables1D[i] = self.variables1D[i]-2*epsilon[i]  # try 2 epsilon
            self.residuals1D(self.variables1D, j)
            model2 = self.model1D_1order(j)
            jacob[i] = (model1-model2)/2./epsilon[i]
            self.variables1D[i] = self.variables1D[i]+epsilon[i]
        self.residuals1D(self.variables1D, j)

        return jacob

    # get sigmas for various parameters
    def sigma1D(self, j):

        jacob = self.jacobian1D(j)

        index = 0
        c_model = np.dot(np.transpose(jacob[:, index:index+1]),
                         np.dot(self.hess1D, jacob[:, index:index+1]))
        self.sigma_a[j] = np.sqrt(np.diag(c_model))[0]
        index = index+1
        c_model = np.dot(np.transpose(jacob[:, index:index+1]),
                         np.dot(self.hess1D, jacob[:, index:index+1]))
        self.sigma_h[j] = np.sqrt(np.diag(c_model))[0]
        index = index+1
        c_model = np.dot(np.transpose(jacob[:, index:index+1]),
                         np.dot(self.hess1D, jacob[:, index:index+1]))
        self.sigma_p[j] = np.sqrt(np.diag(c_model))[0]
        index = index+1
        c_model = np.dot(np.transpose(jacob[:, index:index+np.size(self.age[:, j])]),
                         np.dot(self.hess1D, jacob[:, index:index+np.size(self.age[:, j])]))
        self.sigma_age[:, j] = np.sqrt(np.diag(c_model))
        index = index+np.size(self.age[:, j])
        c_model = np.dot(np.transpose(jacob[:, index:index+np.size(self.age[1:, j])]),
                         np.dot(self.hess1D, jacob[:, index:index+np.size(self.age[1:, j])]))
        self.sigma_logage[1:, j] = np.sqrt(np.diag(c_model))
        self.sigma_logage[0, j] = np.nan
        index = index+np.size(self.age[1:, j])
        c_model = np.dot(np.transpose(jacob[:, index:index+1]),
                         np.dot(self.hess1D, jacob[:, index:index+1]))
        self.sigma_m[j] = np.sqrt(np.diag(c_model))[0]
        index = index+1

        self.sigmabotage[j] = np.interp(self.depth_max[j], self.depth[:, j], self.sigma_age[:, j])
        self.iso_modage_sigma[:, j] = np.interp(self.iso[:, j], self.depth[:, j], self.sigma_age[:, j])


        return

    #Plotting the raw and interpolated radar datasets
    def data_display(self):

        fig = plt.figure('Data')
        plt.plot(self.distance_raw, self.thk_raw, label='raw bedrock', color='0.5', linewidth=2)
        plt.plot(self.distance, self.thkreal, label='interpolated bedrock', color='k', linewidth=2)
        for i in range(self.nbiso):
            if i == 0:
                plt.plot(self.distance_raw, self.iso_raw[i, :], color='c', label='raw isochrones')
                plt.plot(self.distance, self.iso[i, :], color='b', label='interpolated isochrones')
            else:
                plt.plot(self.distance_raw, self.iso_raw[i, :], color='c')
                plt.plot(self.distance, self.iso[i, :], color='b')
        for i in range(self.nbhor):
            if i == 0:
                plt.plot(self.distance_raw, self.hor_raw[i, :], color='y', label='raw horizons')
                plt.plot(self.distance, self.hor[i, :], color='g', label='interpolated horizons')
            elif i > 0 and i < self.nbhor-self.nbdsz:
                plt.plot(self.distance_raw, self.hor_raw[i, :], color='y')
                plt.plot(self.distance, self.hor[i, :], color='g')
            elif i == self.nbhor-self.nbdsz:
                plt.plot(self.distance_raw, self.hor_raw[i, :], color='orange', label='raw DSZ')
                plt.plot(self.distance, self.hor[i, :], color='r', label='interpolated DSZ')
            else:
                plt.plot(self.distance_raw, self.hor_raw[i, :], color='orange')
                plt.plot(self.distance, self.hor[i, :], color='r')
        if self.is_basal:
            plt.plot(self.distance, self.basal, color='black',alpha=0.7,label='Basal layer', linewidth=1)
        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            if self.EDC_line_dashed == True:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2, linestyle='--')
            else:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_BELDC:
            BELDC_x = np.array([self.distance_BELDC, self.distance_BELDC])
            BELDC_y = np.array([0., 3200.])
            plt.plot(BELDC_x, BELDC_y, label='BELDC ice core', color='r', linewidth=2, linestyle='--')

        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('distance (km)')
        plt.ylabel('depth (m)')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y2, 0))
        if self.reverse_distance:
            plt.gca().invert_xaxis()
        pp = PdfPages(self.label+'Data.pdf')
        pp.savefig(plt.figure('Data'))
        pp.close()
        plt.close(fig)

    # find accumulation for each layer
    def accu_layers(self):

        self.accusteady_layer = np.zeros((self.nbiso, np.size(self.distance)))
        self.accu_layer = np.zeros((self.nbiso, np.size(self.distance)))
        for j in range(np.size(self.distance)):
            self.iso_modage[:, j] = np.interp(self.iso[:, j], self.depth[:, j], self.age[:, j])
            self.accusteady_layer[0, j] = self.a[j]*(self.iso_modage[0, j]-self.age_surf)/\
                                          (self.iso_age[0]-self.age_surf)
            self.accusteady_layer[1:, j] = self.a[j]*(self.iso_modage[1:, j]-\
                                           self.iso_modage[:-1, j])/(self.iso_age[1:]-\
                                           self.iso_age[:-1]).flatten()
        self.accu_layer[0, ] = self.accusteady_layer[0, :]*(self.iso_steadyage[0]-\
                               self.age_surf)/(self.iso_age[0]-self.age_surf)
        self.accu_layer[1:, ] = self.accusteady_layer[1:, ]*(self.iso_steadyage[1:]-\
                                self.iso_steadyage[:-1])/(self.iso_age[1:]-self.iso_age[:-1])

        return

    # save data for maps
    def bot_age_save(self):

        output = np.vstack((self.LON, self.LAT, self.distance, self.thk, self.agebot,
                            self.agebotmin, self.agebotmax, self.age100m, self.age150m, self.age200m,
                            self.age250m, self.age_density1Myr, self.age_density1dot2Myr,
                            self.age_density1dot5Myr, self.height0dot6Myr, self.height0dot8Myr,
                            self.height1Myr, self.height1dot2Myr, self.height1dot5Myr,
                            self.agebot10kyrm, self.agebot15kyrm, self.thkreal))

        with open(self.label+'agebottom.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tinverted_thickness(m)\tbottom_age(yr-b1950)'
                    '\tage-min(yr-b1950)\tage-max(yr-b1950)'
                    '\tage100m\tage150m\tage200m\tage250\tage_density1Myr\tage_density1.2Myr\t'
                    'age_density1.5Myr\theight0.6Myr\theight0.8Myr\theight1Myr\theight1.2Myr\t'
                    'height1.5Myr'
                    '\tage-10kyrm\tage-15kyrm\treal_thickness'
                    '\n')

            np.savetxt(f, np.transpose(output), delimiter="\t")

    # save isochrone ages
    def iso_age_save(self):
        output = np.vstack((self.LON, self.LAT, self.distance, self.iso_modage,
                            self.iso_modage_sigma))
        header = '#LON\tLAT\tdistance(km)'
        for i in range(self.nbiso):
            header = header+'\tiso_no_'+str(i+1)
        for i in range(self.nbiso):
            header = header+'\tsigma_iso_no_'+str(i+1)
        header = header+'\n'
        with open(self.label+'ageisochrones.txt', 'w') as f:
            f.write(header)
            np.savetxt(f, np.transpose(output), delimiter="\t")
        for i in range(self.nbiso):
            print('isochrone no:', i+1, ', average age: ', np.nanmean(self.iso_modage[i, :]),
                  ', stdev age: ', np.nanstd(self.iso_modage[i, :]))

    # save accumulation parameters
    def parameters_save(self):
        output = np.vstack((self.LON, self.LAT, self.distance, self.a, self.sigma_a,
                            self.accu_layer))
        header = '#LON\tLAT\tdistance(km)\taccu(ice-m/yr)\tsigma_accu'
        header = header + '\tlayer ' + str(int(self.age_surf/1000.)) + '-' +\
                 str(int(self.iso_age[0]/1000.)) + 'kyr'
        for i in range(self.nbiso-1):
            header = header + '\tlayer ' + str(int(self.iso_age[i]/1000.)) + '-' +\
                     str(int(self.iso_age[i+1]/1000.)) + 'kyr'
        header = header + '\n'
        with open(self.label+'a.txt', 'w') as f:
            f.write(header)
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.m, self.sigma_m))
        with open(self.label+'m.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tmelting(ice-m/yr)\tsigma_melting\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.p, self.sigma_p))
        with open(self.label+'p.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tp\tsigma_p\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.p_prime))
        with open(self.label+'p_prime.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tp\tp_prime\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.resi_sd, self.bic, self.niso))
        with open(self.label+'resi_sd.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tresi_sd\tBIC\tN_iso\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        diff = self.thk-self.basal
        output = np.vstack((self.LON, self.LAT, self.distance, self.stagnant, self.thk,  self.basal, diff))
        with open(self.label+'stagnant.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tstagnant_ice (m)\tinverted_thickness (m)\tbasal_unit (m)\tdifference (m)\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.depth_max,  self.agebot, self.sigmabotage))
        with open(self.label+'res_max.txt', 'w') as f:
            f.write('#LON\tLAT\tdistance(km)\tdepth (m)\tage (yrs)\tage sigma(kyr)\n')
            np.savetxt(f, np.transpose(output), delimiter="\t")
        output = np.vstack((self.LON, self.LAT, self.distance, self.m, self.stagnant, self.agebot, self.age_density1dot2Myr, self.a, self.p, self.resi_sd))

        # matrices which can be optionally saved in order to replot model results
        np.savetxt(self.label+'sigma_thickness.txt', self.sigma_h, delimiter='\t')
        np.savetxt(self.label+'agesteady.txt', self.agesteady/1000., delimiter='\t')
        np.savetxt(self.label+'agematrix.txt', self.age/1000., delimiter='\t')
        np.savetxt(self.label+'x_dist.txt', self.dist, delimiter='\t')
        np.savetxt(self.label+'y_depth.txt', self.depth, delimiter='\t')
        np.savetxt(self.label+'sigma_age.txt', self.sigma_age/1000., delimiter='\t')

    # plot modelled parameters
    def model_display(self):

        # calculate stagnant ice thickness
        self.stagnant = self.thkreal[:] - self.thk[:]
        inverted_depth = np.where(self.stagnant>0, np.nan, self.thk)
        # model steady
        fig, plotmodel = plt.subplots()
        plt.plot(self.distance, self.thkreal, label='obs. bedrock', color='k',
            linewidth=2)
        for i in range(self.nbiso):
            if i == 0:
                plt.plot(self.distance, self.iso[i, :], color='w', linewidth=1,
                         label='obs. isochrones')
            else:
                plt.plot(self.distance, self.iso[i, :], color='w', linewidth=1)
        # levels for age colour gradient
        levels = np.arange(0, self.max_age, self.max_age/10)
        levels_color = np.arange(0, self.max_age, self.max_age/100)
        plt.contourf(self.dist, self.depth, self.agesteady/1000., levels_color, cmap='jet')
        # plot bedrock, stagnant ice, basal unit
        plt.fill_between(self.distance, self.thkreal, self.thk,
            where=self.thk<self.thkreal, color='0.7', label='stagnant ice')
        plt.fill_between(self.distance, self.thkreal, self.thk,
            where=self.thk>self.thkreal, color='white', label='bedrock')
        plt.plot(self.distance, inverted_depth, color='darkviolet',
            label='inverted depth', linewidth=1)
        if self.is_basal:
            plt.plot(self.distance, self.basal, color='black',
                label='Basal layer', linewidth=1)
        # show EDC
        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('Distance (km)')
        plt.ylabel('Depth (m)')
        cb = plt.colorbar()
        cb.set_ticks(levels)
        cb.set_ticklabels(levels)
        cb.set_label('Modeled steady age (kyr)')
        x1, x2, y1, y2 = plt.axis()
        if not self.invert_thk:
            y2 =y2*1.05
        # show reliablity index
        plt.scatter(self.distance, np.ones(len(self.distance))*(y2), c=self.resi_sd, norm=Normalize(vmin=0., vmax=self.sigma_r), cmap='PiYG_r', marker='s', s=4, lw=0. )
        cb = plt.colorbar(orientation='horizontal', shrink =0.7, pad=0.16)
        cb.set_label('Reliability index')
        cb.set_ticks([0,2])
        cb.set_ticklabels(["Reliable","Unreliable"])
        if self.place == 'Ridge B':
            y2 = np.max(self.thkreal)*1.05
        plt.axis((x1,x2,y1,y2*1.05))
        x1, x2, y1, y2 = plt.axis()
        if self.max_depth == 'auto':
            self.max_depth = y2
        plt.axis((min(self.distance), max(self.distance), self.max_depth, 0))

        if self.reverse_distance:
            plt.gca().invert_xaxis()
        pp = PdfPages(self.label+'Model-steady.pdf')
        pp.savefig(fig)
        # plt.show()
        pp.close()
        plt.close(fig)

        # model
        fig, plotmodel = plt.subplots()
        plt.plot(self.distance, self.thkreal, color='k', linewidth=2, label='bed')
        for i in range(self.nbiso):
            if i == 0:
                plt.plot(self.distance, self.iso[i, :], color='w', linewidth=1,
                         label='obs. isochrones')
            else:
                plt.plot(self.distance, self.iso[i, :], color='w', linewidth=1)

        plt.contourf(self.dist, self.depth, self.age/1000., levels_color, cmap='jet')


        plt.fill_between(self.distance, self.thkreal, self.thk,
            where=self.thk<self.thkreal, color='0.7',label='stagnant ice',interpolate=True)
        plt.fill_between(self.distance, self.thkreal, self.thk,
            where=self.thk>self.thkreal, color='white', label='bedrock',interpolate=True)
        plt.plot(self.distance, inverted_depth, color='darkviolet',
            label='inverted depth', linewidth=1)
        if self.is_basal:
            plt.plot(self.distance, self.basal, color='black',
                label='Basal layer', linewidth=1)

        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            if self.EDC_line_dashed == True:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2, linestyle='--')
            else:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_BELDC:
            BELDC_x = np.array([self.distance_BELDC, self.distance_BELDC])
            BELDC_y = np.array([0., 3200.])
            plt.plot(BELDC_x, BELDC_y, label='BELDC ice core', color='r', linewidth=2, linestyle='--')
        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('Distance (km)')
        plt.ylabel('Depth (m)')
        if self.is_legend:
            leg = plt.legend(loc=1)
            frame = leg.get_frame()
            frame.set_facecolor('0.75')

        # cb = plt.colorbar(location='bottom', pad=0.16)
        cb = plt.colorbar()
        cb.set_ticks(levels)
        cb.set_ticklabels(np.asarray(levels, dtype = 'int'))
        cb.set_label('Modelled age (kyr)')
        x1, x2, y1, y2 = plt.axis()
        # show reliability index
        if not self.invert_thk:
            y2 =y2*1.05
        plt.scatter(self.distance, np.ones(len(self.distance))*(y2), c=self.resi_sd, norm=Normalize(vmin=0., vmax=self.sigma_r), cmap='PiYG_r', marker='s', s=4, lw=0. )
        cb = plt.colorbar(orientation='horizontal', shrink =0.7, pad=0.16)
        cb.set_label('Reliability index')
        cb.set_ticks([0,2])
        cb.set_ticklabels(["Reliable","Unreliable"])
        x1, x2, y1, y2 = plt.axis()
        if self.max_depth == 'auto':
            self.max_depth = y2
        plt.axis((min(self.distance), max(self.distance), self.max_depth, 0))

        if self.reverse_distance:
            plt.gca().invert_xaxis()
        if self.settick == 'manual':
            plotmodel.set_xticks(np.arange(self.min_tick, self.max_tick+1., self.delta_tick))
        pp = PdfPages(self.label+'Model.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close(fig)
        # self.dist = np.max(self.dist) - self.dist
        # self.distance = np.max(self.distance) - self.distance

        # AgeMisfit
        fig, plotmodel = plt.subplots()
        plt.plot(self.distance, self.thkreal, color='k', linewidth=2)
        plt.fill_between(self.distance, self.thkreal, self.thk, where=self.thk<self.thkreal, color='0.5', label='stagnant ice')
        norm = Normalize(vmin=-5000, vmax=5000)
        for i in range(self.nbiso):
            colorscatter = self.iso_modage[i, :]-self.iso_age[i]
            if i == 0:
                sc = plt.scatter(self.distance, self.iso[i, :], c=colorscatter,
                                 label='obs. isochrones', s=3, edgecolor=None, norm=norm,cmap='coolwarm')
            else:
                plt.scatter(self.distance, self.iso[i, :], c=colorscatter, s=3, edgecolor=None,
                            norm=norm,cmap='coolwarm')

        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            if self.EDC_line_dashed == True:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2,
                         linestyle='--')
            else:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_BELDC:
            BELDC_x = np.array([self.distance_BELDC, self.distance_BELDC])
            BELDC_y = np.array([0., 3200.])
            plt.plot(BELDC_x, BELDC_y, label='BELDC ice core', color='r', linewidth=2, linestyle='--')

        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('distance (km)')
        plt.ylabel('depth (m)')
        if self.is_legend:
            print('test')
            leg = plt.legend(loc=1)
            frame = leg.get_frame()
            frame.set_facecolor('0.75')
        cb = plt.colorbar(sc)
        plt.plot(self.distance, inverted_depth, color='darkviolet', label='inverted depth', linewidth=1)
        cb.set_label('Age misfit (yr)')
        x1, x2, y1, y2 = plt.axis()

        plt.axis((min(self.distance), max(self.distance), self.max_depth, 0))
        if self.reverse_distance:
            plt.gca().invert_xaxis()
        if self.settick == 'manual':
            plotmodel.set_xticks(np.arange(self.min_tick, self.max_tick+1., self.delta_tick))
        pp = PdfPages(self.label+'AgeMisfit.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close(fig)

        # model confidence intervals
        fig, plotmodelci = plt.subplots()
        plt.plot(self.distance, self.thkreal, color='k', linewidth=2)
        plt.fill_between(self.distance, self.thkreal, self.thk, where=self.thk<self.thkreal, color='0.5', label='stagnant ice')
        plt.plot(self.distance, inverted_depth, color='r', label='inverted depth', linewidth=1)
        for i in range(self.nbiso):
            if i == 0:
                plt.plot(self.distance, self.iso[i, :], color='w', label='obs. isochrones')
            else:
                plt.plot(self.distance, self.iso[i, :], color='w')
        levels_log = np.arange(2, 6, 0.1)
        levels = np.power(10, levels_log)
        plt.contourf(self.dist[1:-1,:], self.depth[1:-1,:], self.sigma_age[1:-1,:], levels, norm=LogNorm())
        cb = plt.colorbar()
        cb.set_label('Modeled age confidence interval (yr)')
        levels_labels = np.array([])
        for i in np.arange(2, 6, 1):
            levels_labels = np.concatenate((levels_labels,
                                            np.array([10**i, '', '', '', '', '', '', '', '', ''])))

        cb.set_ticklabels(levels_labels)
        levels_ticks = np.concatenate((np.arange(100, 1000, 100),
                                       np.arange(1000, 10000, 1000),
                                       np.arange(10000, 100000, 10000),
                                       np.arange(100000, 600000, 100000)))
        cb.set_ticks(levels_ticks)
        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            if self.EDC_line_dashed == True:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2, linestyle='--')
            else:
                plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_BELDC:
            BELDC_x = np.array([self.distance_BELDC, self.distance_BELDC])
            BELDC_y = np.array([0., 3200.])
            plt.plot(BELDC_x, BELDC_y, label='BELDC ice core', color='r', linewidth=2, linestyle='--')

        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('distance (km)')
        plt.ylabel('depth (m)')
        if self.is_legend:
            leg = plt.legend(loc=1)
            frame = leg.get_frame()
            frame.set_facecolor('0.75')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((min(self.distance), max(self.distance), self.max_depth, 0))
        if self.reverse_distance:
            plt.gca().invert_xaxis()
        if self.settick == 'manual':
            plotmodelci.set_xticks(np.arange(self.min_tick, self.max_tick+1., self.delta_tick))
        pp = PdfPages(self.label+'Model-confidence-interval.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close(fig)

        # thinning
        plt.figure('Thinning')
        plt.plot(self.distance, self.thkreal, label='obs. bedrock', color='k', linewidth=2)
        plt.fill_between(self.distance, self.thkreal, self.thk, where=self.thk<self.thkreal, color='0.5', label='stagnant ice')
        plt.plot(self.distance, inverted_depth, color='r', label='inverted depth', linewidth=1)
        plt.contourf(self.dist, self.depth, self.tau)
        if self.is_EDC:
            EDC_x = np.array([self.distance_EDC, self.distance_EDC])
            EDC_y = np.array([0., 3200.])
            plt.plot(EDC_x, EDC_y, label='EDC ice core', color='r', linewidth=2)
        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('distance (km)')
        plt.ylabel('depth (m)')
        plt.legend(loc=2)
        cb = plt.colorbar()
        cb.set_label('Modeled thinning')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((min(self.distance), max(self.distance), self.max_depth, 0))
        if self.reverse_distance:
            plt.gca().invert_xaxis()
        pp = PdfPages(self.label+'Thinning.pdf')
        pp.savefig(plt.figure('Thinning'))
        pp.close()
        plt.close(fig)

        # accumulation hisotry
        lines = [list(zip(self.distance, 917*self.accu_layer[i, :])) for i in range(self.nbiso)]
        z = (self.iso_age.flatten()[1:]+self.iso_age.flatten()[:-1])/2
        z = np.concatenate((np.array([(self.age_surf+self.iso_age.flatten()[0])/2]), z))
        fig, ax = plt.subplots()
        lines = LineCollection(lines, array=z, cmap=plt.cm.rainbow, linewidths=2)
        ax.add_collection(lines)
        ax.autoscale()
        cb = fig.colorbar(lines)
        cb.set_label('Average layer age (yr)')
        if self.is_NESW:
            plt.xlabel('<NE - distance (km) - SW>')
        else:
            plt.xlabel('distance (km)')
        plt.ylabel('accumulation (mm-we/yr)')
        if self.reverse_distance:
            plt.gca().invert_xaxis()

        pp = PdfPages(self.label+'AccumulationHistory.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close(fig)
        print('model displayed')

    # model parameter graphs
    def parameters_display(self):

        plt.figure()
        plt.plot(self.distance, self.a*100, label='accumulation', color='k')
        plt.ylabel('accu. (cm/yr)', fontsize=10)
        plt.xlabel('distance (km)')
        plt.savefig(self.label+'accumulation.pdf')

        plt.figure()
        plt.plot(self.distance, self.p, label='p', color='k')
        plt.ylabel('p parameter', fontsize=10)
        plt.xlabel('distance (km)')
        plt.savefig(self.label+'p.pdf')

        plt.figure()
        plt.plot(self.distance, self.p_prime, label='p_prime', color='k')
        plt.ylabel('p\' parameter', fontsize=10)
        plt.xlabel('distance (km)')
        plt.savefig(self.label+'p_prime.pdf')

        plt.figure()
        plt.plot(self.distance, self.m, label='melting', color='k')
        plt.ylabel('melting', fontsize=10)
        plt.xlabel('distance (km)')
        plt.savefig(self.label+'m.pdf')

        plt.figure()
        plt.plot(self.distance, self.resi_sd, label='residual sd', color='k')
        plt.ylabel('residual standard deviation', fontsize=10)
        plt.xlabel('distance (km)')
        plt.savefig(self.label+'resi_sd.pdf')

    # age profile at specified site
    def drill(self, name, distance_drill):

        # model result for drill ice column
        age_drill = np.array([ np.interp(distance_drill, self.distance, self.age[dep]) for dep in range(len(self.age)) ])
        sigmaage_drill = np.array([ np.interp(distance_drill, self.distance, self.sigma_age[dep]) for dep in range(len(self.sigma_age)) ])
        depth_drill = np.array([ np.interp(distance_drill, self.distance, self.depth[dep]) for dep in range(len(self.depth)) ])
        iso_drill = np.array([ np.interp(distance_drill, self.distance, self.iso[dep]) for dep in range(len(self.iso)) ]).reshape(self.iso_age.shape)
        omega_drill = np.array([ np.interp(distance_drill, self.distance, self.omega[dep]) for dep in range(len(self.omega)) ])
        age_density_drill = np.array([ np.interp(distance_drill, self.distance, self.age_density[dep]) for dep in range(len(self.age_density)) ])
        # single values at drill location
        R_AICC2012 = self.AICC2012_accu/self.AICC2012_averageaccu
        f = self.interp1d_lin_aver(self.AICC2012_age, R_AICC2012, right=1., left=R_AICC2012[0])
        R_drill = f(age_drill)
        a_drill = np.interp(distance_drill,self.distance, self.a)
        accu_drill = a_drill*R_drill
        accu_drill = np.concatenate((accu_drill,np.array([np.nan])))
        bed_drill = np.interp(distance_drill, self.distance, self.thkreal)
        stag_drill = np.interp(distance_drill, self.distance, self.stagnant)
        stigmastag_drill = np.interp(distance_drill, self.distance, self.sigma_h)
        age_drill_bot = np.interp(distance_drill, self.distance, self.agebot)
        depth_drill_bot = np.interp(distance_drill, self.distance, self.depth_max)
        agefromdepth = np.interp(depth_drill_bot, depth_drill, age_drill)
        sigmaage_drill_bot = np.interp(min(bed_drill, depth_drill_bot), depth_drill, sigmaage_drill)
        p_drill = np.interp(distance_drill, self.distance, self.p)
        depth1dot2_drill = bed_drill - stag_drill- np.interp(distance_drill, self.distance, self.height1dot2Myr)
        depth1dot5_drill = bed_drill -  stag_drill- np.interp(distance_drill, self.distance, self.height1dot5Myr)
        agedens1dot2_drill = np.interp(distance_drill, self.distance, self.age_density1dot2Myr)
        agedens1dot5_drill = np.interp(distance_drill, self.distance, self.age_density1dot5Myr)
        # print to terminal
        print('age from z interpolation', agefromdepth, '\nage from x interpolation', age_drill_bot)
        print('accumulation at', name, 'is: ', a_drill)
        print('Age at drill 60 m above stagnant ice is: ', age_drill_bot, '+-', sigmaage_drill_bot, 'yrs')
        print('p parameter at drill is: ', p_drill)
        print('Stagnant ice thickness is: ', stag_drill, '+-', stigmastag_drill, 'm')
        print('Total ice thickness  is: ', bed_drill, 'm')
        if stag_drill > 0:
            print('Ice thickness above stagnant ice is: ', depth_drill[-1])
        # age and thinning for profile (old threshold = bed-60m)
        age_plot, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(age_drill[depth_drill<max(depth_drill)-60]/1e3, omega_drill[depth_drill<max(depth_drill)-60], label='thinning', color='orange')
        ax1.plot(age_drill[depth_drill<max(depth_drill)-60]/1e3, depth_drill[depth_drill<max(depth_drill)-60], label='depth', color = 'b')
        ax1.errorbar(self.iso_age.flatten()/1e3, iso_drill.flatten(), xerr=self.iso_sigma.flatten()/1e3,  fmt=".",label='depth isos', color = 'black')
        ax1.invert_yaxis()
        ax1.set_ylabel('depth (m)', color='b')
        ax2.set_ylabel('thinning', color='orange')
        ax1.set_xlabel('Age (kyr)')
        ax1.legend(loc=1)
        pp = PdfPages(self.label+'Age_'+ name + '.pdf')
        pp.savefig(age_plot)
        pp.close()
        # vertical velocity profile
        accu_now = accu_drill[0]
        vv_plot, ax1 = plt.subplots()
        vv = -accu_now*omega_drill      #vertical velocity
        ax1.plot(vv, depth_drill)
        ax1.plot(-accu_drill*omega_drill , depth_drill)
        ax1.set_xlabel('vertical velocity (m/yr)')
        ax1.set_ylabel('depth (m)')
        ax1.invert_yaxis()
        plt.savefig(self.label+'vertical_v_'+name+'.pdf')

        # header for saving data
        output = np.vstack((depth_drill[:-1], age_drill[:-1], sigmaage_drill[:-1],age_density_drill, accu_drill[:-1], omega_drill[:-1],vv[:-1]))
        header = '# Total ice thickness is (m): ' + str(bed_drill) +'+-' + str(stigmastag_drill) +'\n'
        if stag_drill > 0:
            header += '# Maximum age (yrs): ' +  str(age_drill_bot) + '+-' + str(sigmaage_drill_bot) + '\n'
            header += '# Stagnant ice thickness (m): ' + str(stag_drill) +'+-' + str(stigmastag_drill) +'\n'
        else:
            header += '# Age at bedrock (yrs): ' + str(age_drill_bot) +'+-' + str(sigmaage_drill_bot) +'\n'
            header += '# No stagnant ice\n'
        header += '# Depth of maximum age (m) at '+ name + ' is: ' + str(depth_drill_bot) + '\n'
        header += '# Accumulation at '+ name + ' is: ' + str(a_drill) + '\n'
        header += '# p parameter at '+ name + ' is: ' + str(p_drill) + '\n'
        header += '# Depth 1.2 Myr ice at '+ name + ' is: ' + str(depth1dot2_drill) + '\n'
        header += '# Depth 1.5 Myr ice at '+ name + ' is: ' + str(depth1dot5_drill) + '\n'
        header += '# Age density of 1.2 Myr ice (ka/m) at '+ name + ' is: ' + str(agedens1dot2_drill/1000) + '\n'
        header += '# Age density of 1.5 Myr ice (ka/m) at '+ name + ' is: ' + str(agedens1dot5_drill/1000) + '\n'

        # data to save at drill site
        header += 'depth_(m)\tage_(yrs)\tsigma_age(yrs)\tage_density(kyr/m)\taccumulation\tthinning\tvertical_velocity(m/yr)\n'
        with open(self.label+'Age_'+ name + '.txt', 'w') as f:
            f.write(header)
            np.savetxt(f, np.transpose(output), delimiter="\t")

        if isinstance(self.iso_accu_sigma, float):
            self.iso_accu_sigma = np.empty(self.iso_age.shape)
            self.iso_accu_sigma[:] = np.nan
        # save isochrone ages and uncertainties
        output = np.hstack((self.iso_age, self.iso_sigma, self.iso_accu_sigma, iso_drill))
        with open(self.label+'Depths_'+ name + '.txt', 'w') as f:
            f.write('#age_(yr_BP)\tsigma_age_(yr_BP)\tsigma_accu\tdepths_(m)\n')
            np.savetxt(f, output, delimiter="\t")

    # run entire model
    def run_model(self):

        # set up parameters
        self.default_params()
        print('Initialization of radar line')
        self.load_parameters()
        self.load_radar_data()
        self.interp_radar_data()
        self.iso_data()
        self.init_arrays()
        self.data_display()


        # run model
        if self.opt_method == 'leastsq1D':
            print('Optimization by leastsq1D')
            for j in range(np.size(self.distance)):
                print('index along the radar line: ', j)
                bounds = [-np.inf, -np.inf], [np.inf, np.inf]
                self.variables1D = np.array([self.a[j], self.p_prime[j]])
                if self.invert_thk:
                    self.variables1D = np.append(self.variables1D, m.log(self.thk[j]))
                    bounds = ([-np.inf, -np.inf, m.log(np.max(self.iso[~np.isnan(self.iso[:,j]),j]))], [np.inf, np.inf, np.inf])
                if self.invert_s:
                    self.variables1D = np.append(self.variables1D, self.s[j])
                # do least square fit to get variables and hessian matrix
                leastsq_fit1D = least_squares(self.residuals1D, self.variables1D, bounds=bounds, args=(j,), method='trf')
                self.variables1D = leastsq_fit1D.x

                self.hess1D = np.linalg.inv(np.dot(np.transpose(leastsq_fit1D.jac), leastsq_fit1D.jac))

                print(self.variables1D)
                # calc residuals and save results
                resi=self.residuals1D(self.variables1D, j)
                self.resi_sd[j] = m.sqrt(np.mean(self.age_resi**2))
                self.niso[j] = np.sum(~np.isnan(self.iso[:,j]))
                self.bic[j] = -2 * np.log(self.niso[j]*self.resi_sd[j]) + len(self.variables1D) * np.log(self.niso[j])

                self.model1D_finish(j)
                if not self.calc_sigma:
                    self.hess1D = np.zeros((np.size(self.variables1D), np.size(self.variables1D)))
                if np.size(self.hess1D) != 1:
                    self.sigma1D(j)

            self.agebotmin = self.agebot-self.sigmabotage
            self.agebotmax = self.agebot+self.sigmabotage


        np.savetxt(self.label+'jacobian.csv', leastsq_fit1D.jac, delimiter=',')



        # layers for showing data
        self.accu_layers()
        self.model_display()
        # save data
        self.bot_age_save()
        self.parameters_save()
        self.iso_age_save()
        self.parameters_display()

        # save data for drills
        if self.is_EDC: self.drill('EDC', self.distance_EDC)
        if self.is_BELDC: self.drill('BELDC', self.distance_BELDC)
        if self.is_core: self.drill(self.name_core, self.distance_core)


# get radar line name from terminal and run
RLlabel = sys.argv[1]
if RLlabel[-1] != '/':
    RLlabel = RLlabel+'/'
print('Radar line is: ', RLlabel)
RL = RadarLine(RLlabel)
RL.run_model()
