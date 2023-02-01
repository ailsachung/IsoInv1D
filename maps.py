'''
Class to plot radar age line maps onto bedrock and surface contours
using cartopy. Bedrock and surface data is imported in a named 2d array from a
.txt file and matching x.txt and y.txt files data is in epsg 3031 and
centred on the South Pole. An accompanying parameters_maps.yml is required in
order to set class variables.

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import sys
import cartopy.crs as ccrs
import yaml
import pandas as pd


class Maps(object):

    def __init__(self):

        # set radar line directory from terminal input
        self.RLDir=sys.argv[1]
        if self.RLDir[-1]!='/':
            self.RLDir=self.RLDir+'/'

        # access parameters stored in .yml file
        filename = self.RLDir+'/parameters_maps.yml'
        yamls = open(filename).read()
        para = yaml.load(yamls, Loader=yaml.FullLoader)
        self.__dict__.update(para)

        self.first_map = True
        self.saved = False

    # after initiation, load data and set up map names
    def next(self):

        self.load_model_data(self.list_RL)

        self.lons = self.botage_array[:,0]
        self.lats = self.botage_array[:,1]

        # list of maps to generate
        self.make_list_maps(nbiso = self.nbiso, nbhor = self.nbhor)

    # load and cut surface and bedrock elevation arrays to map extent
    def bedmachine(self,name,extent,flipy):

        folder = name.split('/')[0]
        # load data from file
        grid = np.loadtxt(self.RLDir+name+'.txt')
        x = np.loadtxt(self.RLDir+folder+'/x.txt')
        y = np.loadtxt(self.RLDir+folder+'/y.txt')

        if flipy:                                           # y axis inverted for bedmachine bed
            extent[2], extent[3] = extent[3], extent[2]
        # find arguments of map extent
        x0 =  np.abs(x - extent[0]).argmin()
        x1 =  np.abs(x - extent[1]).argmin()
        y0 =  np.abs(y - extent[2]).argmin()      # negative values so inverse
        y1 =  np.abs(y - extent[3]).argmin()

        # shrink data sets to the size of the rendered map
        x_lim = x[x0-2:x1+2]
        y_lim = y[y0-2:y1+2]
        xx, yy = np.meshgrid(x_lim, y_lim)
        # 2d grid
        xy_grid = grid[y0-2:y1+2,x0-2:x1+2]

        return x_lim, y_lim, xy_grid, xx, yy

    # extracts data from models
    def load_model_data(self, RLs):

        # load data for each radar line
        for i,RLlabel in enumerate(RLs):
            directory=self.RLDir+RLlabel

            # arrays to a single varible data for all radar lines
            if i == 0:
                if self.place == 'Ridge B':
                    self.accu_array = np.loadtxt(directory+'/a.txt')[:,0:5]
                else:
                    self.accu_array = np.loadtxt(directory+'/a.txt')
                self.botage_array = np.loadtxt(directory+'/agebottom.txt')
                self.stag_array = np.loadtxt(directory+'/stagnant.txt')
                self.m_array = np.loadtxt(directory+'/m.txt')
                self.p_array = np.loadtxt(directory+'/p.txt')
                self.resi_sd_array = np.loadtxt(directory+'/resi_sd.txt')
                self.res_max = np.loadtxt(directory+'/res_max.txt')
                self.lines = [RLlabel] * len(self.botage_array)

            else:
                if self.place == 'Ridge B':
                    self.accu_array=np.concatenate((self.accu_array, np.loadtxt(directory+'/a.txt')[:,0:5]))
                else:
                    self.accu_array=np.concatenate((self.accu_array, np.loadtxt(directory+'/a.txt')))
                self.botage_array=np.concatenate((self.botage_array,np.loadtxt(directory+'/agebottom.txt')))
                self.stag_array = np.concatenate((self.stag_array, np.loadtxt(directory+'/stagnant.txt')))
                self.m_array=np.concatenate((self.m_array,np.loadtxt(directory+'/m.txt')))
                self.p_array=np.concatenate((self.p_array,np.loadtxt(directory+'/p.txt')))
                self.resi_sd_array=np.concatenate((self.resi_sd_array,np.loadtxt(directory+'/resi_sd.txt')))
                self.res_max=np.concatenate((self.res_max,np.loadtxt(directory+'/res_max.txt')))
                self.lines = self.lines + [RLlabel] * len(self.botage_array)

            self.lakes = pd.read_csv(self.RLDir+'../bedmap2/lakes.txt',header=0, sep='\t')

    # list of maps to generate
    def make_list_maps(self, nbiso = 0, nbhor = 0):

        #Reading isochrones' ages
        if self.place != 'Ridge B':
            readarray=np.loadtxt(self.RLDir+'ages.txt')
            iso_age=np.concatenate((np.array([0]),readarray[:,0]))

        for i in range(nbiso):
            self.list_maps.append('Accu-layer'+ "%02i"%(i+1) +'_'+str(int(iso_age[i]/1000.))+'-'+str(int(iso_age[i+1]/1000.))+'kyr' )
        for i in range(nbhor):
            self.list_maps.append('age-hor'+"%02i"%(i+1))

        # has to be last
        if self.list_RL_highlight:
            self.list_maps.append('radar-lines_highlights')
            self.limits['radar-lines_highlights'] = {'vmin': 0., 'vmax': 1., 'ticks': 'auto'}

    # intitiate map figure with surface and bedrock data
    def init_map(self):

        # initiate cartopy figure
        self.fig = plt.figure(figsize=(16/2.54,16/2.54))
        self.cart_map1 = self.fig.add_subplot(1,1,1, projection = ccrs.SouthPolarStereo(true_scale_latitude = -71, central_longitude = 0))
        # limit to area of interest
        self.cart_map1.set_extent([self.lon1, self.lon2, self.lat1,self.lat2], ccrs.PlateCarree())
        #add gridlines and labels
        cart_grid1 = self.cart_map1.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, color='gray', alpha=0.5,linestyle = '--')
        cart_grid1.rotate_labels = False

        norm = Normalize(vmin=self.limits['bedrock']['vmin'],vmax=self.limits['bedrock']['vmax'])

        # get bedrock and surface maps- only needs to be done once
        if self.first_map:
            #  extent is x0,x1,y0,y1 in m (or epsg: 3031)
            extent = np.array(self.cart_map1.get_extent())
            # get bedrock array which fits map
            self.x_lim_bed, self.y_lim_bed, self.xy_bedrock, xx_bed, yy_bed = self.bedmachine(self.bed_name, extent, True)
            self.xy_bedrock = np.fliplr(self.xy_bedrock)           # bedmachine is completely mirrored
            self.xy_bedrock = np.flipud(self.xy_bedrock)           # so flip left/right and up/down
            # get surface elevation array
            extent = np.array(self.cart_map1.get_extent())
            x_lim_surf, y_lim_surf, self.xy_surface, self.xx_surf, self.yy_surf = self.bedmachine(self.surf_name, extent, False)

            self.first_map = False
            # plot bedrock elevation heatmap and colorbar
            cs_bed = self.cart_map1.imshow(self.xy_bedrock, extent=[max(self.x_lim_bed),min(self.x_lim_bed),max(self.y_lim_bed),min(self.y_lim_bed)], transform=ccrs.epsg(3031), norm=norm, alpha=0.3, cmap='gist_earth')
        else:
            cs_bed = self.cart_map1.imshow(self.xy_bedrock, extent=[max(self.x_lim_bed),min(self.x_lim_bed),max(self.y_lim_bed),min(self.y_lim_bed)], transform=ccrs.epsg(3031), norm=norm, alpha=0.5, cmap='gray')

        cb_bed = self.fig.colorbar(cs_bed, ax=self.cart_map1,  orientation='horizontal', shrink=0.7, pad=0.0)
        cb_bed.set_label('Bedrock elevation (m)')

        # contours for surface heights to plot
        levels=np.concatenate(( np.arange(self.levels[0], self.levels[1], self.levels[2]),np.arange(self.levels[3],self.levels[4], self.levels[5]) ))
        cs_surf=self.cart_map1.contour(self.xx_surf,self.yy_surf, self.xy_surface, transform=ccrs.epsg(3031), levels=levels, alpha=0.2, colors = 'k')
        plt.clabel(cs_surf, inline=1, fontsize=10,fmt='%1.0f')

    # complete map by adding colorbar and plotting points of interest
    def finish_map(self, MapLabel):

        # for paper: remove one s
        if MapLabel[0:11] !='radar-liness':
            # cb = plt.colorbar(self.scatter, ax=self.cart_map1,  orientation='vertical', shrink =0.6, pad=0.12)
            cb = plt.colorbar(self.scatter, ax=self.cart_map1,  orientation='horizontal', shrink =0.7, pad=0.07)
            cb.set_label(self.cb_label)
            if self.cb_ticks != 'auto':
                cb.set_ticks(self.cb_ticks)
                cb.set_ticklabels(self.cb_ticks)

        # plot and label sites
        if self.labels:
            labels = list(zip(*self.labels))        # transpose of list
            for i, name in enumerate(self.labels):
                self.cart_map1.scatter(x=labels[0][i], y=labels[1][i], color = 'black', s=50,  marker=labels[3][i],transform=ccrs.PlateCarree()) ## Important
                self.cart_map1.text(labels[0][i]+self.label_offset,labels[1][i]+self.label_offset,labels[2][i],horizontalalignment='center',verticalalignment='bottom',color='black', transform=ccrs.PlateCarree())

        # show position of drilling location
        if self.is_drill:
            self.cart_map1.scatter(x=self.lon_drill, y=self.lat_drill, color = 'r', s=50,  alpha=1, marker = '+',transform=ccrs.PlateCarree()) ## Important

        # plt.show()
        plt.tight_layout()
        plt.savefig(self.RLDir+MapLabel+'_cart.pdf', bbox_inches='tight')
        plt.close()

    # create all maps with radar lines
    def make_maps(self):

        # specify variables which are true for all maps
        resi_sd = self.resi_sd_array[:,3]               # reliablity
        ok_res = resi_sd[resi_sd<self.reliability]      # reliablity mask (<2)
        # radar line longitudes and latitudes
        self.lons, self.lats = np.transpose(self.botage_array[resi_sd<self.reliability][:,:2])
        # get data for each graph from arrays
        botage, minbotage, age100, age150, age200, age250,res1, res12, res15 =  np.transpose(self.botage_array[resi_sd<self.reliability][:,4:13])
        HAB08, HAB1, HAB12, HAB15,bed_real = np.transpose(self.botage_array[resi_sd<self.reliability][:,14:19])
        pp_lons, pp_lats, not_used, p, sigma_p = np.transpose(self.p_array[resi_sd<self.reliability][:,0:5])
        m_lons, m_lats, not_used, melting, sigma_melting = np.transpose(self.m_array[resi_sd<self.reliability][:,0:5])
        stagnant, inv_depth, basal, diff = np.transpose(self.stag_array[resi_sd<self.reliability][:,3:7])
        accu_lons, accu_lats, accu_dist, accu, accu_sigma = np.transpose(self.accu_array[resi_sd<self.reliability][:,:5])
        max_lons, max_lats, max_dist,max_depth, max_age = np.transpose(self.res_max[resi_sd<self.reliability])

        # for paper: save file with data for all map figures
        if not self.saved:
            stagnant[stagnant<0] = 0
            output = np.transpose(np.vstack((self.lons, self.lats, melting, stagnant, accu, p, botage, res12, ok_res)))
            names = ['lon', 'lat', 'melting(m/yr)','stagnant_ice_thickness(m)', 'steady_accu(m/yr)', 'p', 'max_age(a)', 'age_density(yr/m)', 'reliability']
            to_save = pd.DataFrame(output, columns = names)
            mask = resi_sd<self.reliability
            to_save['line'] = [b for a, b in zip(mask.tolist(), self.lines) if a]
            to_save.to_csv(self.RLDir+"figures.csv",na_rep='nan',index=False, header=True)
            to_save.to_csv(self.RLDir+"figures.txt",sep='\t',na_rep='nan',index=False, header=True)
            self.saved = True

        # loop to plot each map
        for i,MapLabel in enumerate(self.list_maps):

            print(MapLabel)

            # generate map with bedrock and surface contours
            self.init_map()

            if not MapLabel[0:4]=='Accu':
                self.cb_ticks = self.limits[MapLabel]['ticks']
                norm = Normalize(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])

            if MapLabel=='radar-lines':
                self.scatter = self.cart_map1.scatter(x=self.lons, y=self.lats, color = 'blue', marker='o', s=self.dotsize, lw=.0, transform=ccrs.PlateCarree(),alpha=0.3)
                self.cb_label = 'radar lines'

            elif MapLabel=='radar-lines_highlights':
                self.cart_map1.scatter(x=self.lons, y=self.lats, color = 'b', marker='o', s=self.dotsize, lw=.0, transform=ccrs.PlateCarree())
                self.load_model_data(self.list_RL_highlight)
                self.cart_map1.scatter(x=self.botage_array[:,0], y=self.botage_array[:,1], color = 'r', marker='o', s=self.dotsize, lw=.0, transform=ccrs.PlateCarree())


            elif MapLabel=='residual-sd':
                self.scatter = self.cart_map1.scatter(self.botage_array[:,0], self.botage_array[:,1], c=resi_sd, norm=norm, cmap='PiYG_r', marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Reliablity index'


            elif MapLabel=='bottom-age':
                print('max age', np.average(botage[~np.isnan(botage)]/1e6))
                print('sd max age', np.std(botage[~np.isnan(botage)]/1e6))
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=botage/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Maximum age (Ma)'

            elif MapLabel=='bottom-age-depth':
                norm=Normalize()
                # norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=max_depth, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Depth of bottom age (m)'

            # not for paper
            elif MapLabel=='min-bottom-age':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=minbotage/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Minimum bottom age (Myr)'

            elif MapLabel=='age-50m':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=age100/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Age (Myr)'

            elif MapLabel=='age-100m':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=age100/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Age (Myr)'

            elif MapLabel=='age-150m':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=age150/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Age (Myr)'

            elif MapLabel=='age-200m':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=age200/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Age (Myr)'

            elif MapLabel=='age-250m':
                norm = LogNorm(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=age250/1e6, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label ='Age (Myr)'


            elif MapLabel=='age_density-1Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=res1/1e3, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Age Density at 1 Myr (kyr m$^{-1}$)'

            elif MapLabel=='age_density-1.2Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=res12/1e3, marker='o', cmap='viridis_r', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                # norm2 = Normalize(vmin=4.,vmax=9.)            # NP colour scheme
                # self.scatter = self.cart_map1.scatter(self.lons[res12<9000], self.lats[res12<9000], c=res12[res12<9000]/1e3, cmap='magma',marker='o', lw=0., norm = norm2,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Age Density at 1.2 Ma (kyr m$^{-1}$)'

            elif MapLabel=='age_density-1.5Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=res15/1e3, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Age Density at 1.5 Ma (kyr m$^{-1}$)'

            elif MapLabel=='resolution-1Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=1e3/res1, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Resolution at 1 Ma (m kyr$^{-1}$)'

            elif MapLabel=='resolution-1.2Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=1e3/res12,  marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Resolution at 1.2 Ma (m kyr$^{-1}$)'

            elif MapLabel=='resolution-1.5Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=1e3/res15, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Resolution at 1.5 Ma (m kyr$^{-1}$)'

            elif MapLabel=='Height-Above-Bed-0.8Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=HAB08, norm=norm, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Height above bed (m) at 0.8 Ma'

            elif MapLabel=='Height-Above-Bed-1Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=HAB1, norm=norm, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Height above bed (m) at 1 Ma'

            elif MapLabel=='Height-Above-Bed-1.2Myr':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=HAB12, norm=norm, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Height above bed (m) at 1.2 Ma'

            elif MapLabel=='Height-Above-Bed-1.5Myr':
                self.scatter = self.cart_map1.scatter(self.lons[HAB15>60.], self.lats[HAB15>60.], c=HAB15[HAB15>60.], norm=norm, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Height above bed (m) at 1.5 Ma'


            elif MapLabel=='stagnant-ice':
                no_stag = self.cart_map1.scatter(self.lons[stagnant<=0], self.lats[stagnant<=0], c='r', marker='o', lw=0., s=self.dotsize, label = "No stagnant ice", transform=ccrs.PlateCarree())
                self.scatter = self.cart_map1.scatter(self.lons[stagnant>0], self.lats[stagnant>0], c=stagnant[stagnant>0], norm=norm,  marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                lgnd = plt.legend(loc='lower left')
                lgnd.legendHandles[0]._sizes = [15]
                self.cb_label = 'Stagnant ice (m)'

            elif MapLabel=='elevation-non-stagnant-ice':
                elev = np.minimum(self.botage_array[self.resi_sd_array[:,3]<self.reliability][:,4], self.botage_array[self.resi_sd_array[:,3]<self.reliability][:,20])
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=elev, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Mechanical ice thickness $H_{m}$ (m)'

            elif MapLabel=='stagnant-basal':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=diff, norm=norm,  marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree(),cmap='coolwarm')
                self.cb_label = 'Stagnant ice depth - basal unit depth (m)'
                # save file
                output = np.transpose(np.vstack((self.lons, self.lats,diff)))
                names = ['lon', 'lat', 'stag_depth-basal_depth']
                to_save = pd.DataFrame(output, columns = names)
                mask = resi_sd<self.reliability
                to_save['line'] = [b for a, b in zip(mask.tolist(), self.lines) if a]
                to_save.to_csv(self.RLDir+"stag_bas_diff.csv",na_rep='nan',index=False, header=True)
                to_save.to_csv(self.RLDir+"stag_bas_diff.txt",sep='\t',na_rep='nan',index=False, header=True)


            elif MapLabel=='stagnant-depth':
                stag_depth = inv_depth
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=stag_depth,  norm=norm,  marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Top of stagnant ice depth (m)'
                #save file
                output = np.transpose(np.vstack((self.lons, self.lats,basal,stag_depth,)))
                names = ['lon', 'lat', 'basal_unit_depth(m)', 'stagnant_ice_depth(m)']
                to_save = pd.DataFrame(output, columns = names)
                mask = resi_sd<self.reliability
                to_save['line'] = [b for a, b in zip(mask.tolist(), self.lines) if a]
                to_save.to_csv(self.RLDir+"stag_bas_depth.csv",na_rep='nan',index=False, header=True)
                to_save.to_csv(self.RLDir+"stag_bas_depth.txt",sep='\t',na_rep='nan',index=False, header=True)


            elif MapLabel=='basal-depth':
                self.scatter = self.cart_map1.scatter(self.lons, self.lats, c=basal,  norm=norm,  marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = 'Top of basal unit depth (m)'


            elif MapLabel=='melting':
                self.scatter = self.cart_map1.scatter(m_lons[melting>0], m_lats[melting>0], c=melting[melting>0]*1e3, marker='o', lw=0., s=self.dotsize,cmap = 'Reds', transform=ccrs.PlateCarree())
                no_melt = self.cart_map1.scatter(m_lons[melting<=0], m_lats[melting<=0], c='blue', marker='o', lw=0., s=self.dotsize, label = "No melting", transform=ccrs.PlateCarree())
                lgnd = plt.legend(loc='lower left')
                lgnd.legendHandles[0]._sizes = [15]
                self.cb_label = '$\overline{m}$ (mm yr$^{-1}$)'

            elif MapLabel=='melting-sigma':
                self.scatter = self.cart_map1.scatter(m_lons[melting>0], m_lats[melting>0], c=melting[melting>0]*1e3, marker='o', lw=0., s=self.dotsize,cmap = 'Reds', transform=ccrs.PlateCarree())
                self.scatter = self.cart_map1.scatter(m_lons[melting>0], m_lats[melting>0], c=sigma_melting[melting>0]*1e3, marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                no_melt = self.cart_map1.scatter(m_lons[melting<=0], m_lats[melting<=0], c='r', marker='o', lw=0., s=self.dotsize, label = "No melting", transform=ccrs.PlateCarree())
                lgnd = plt.legend(loc='lower left')
                lgnd.legendHandles[0]._sizes = [15]
                self.cb_label = '$\sigma$ Melting (mm yr$^{-1}$)'

            elif MapLabel=='melting-stagnant':
                lakes = self.cart_map1.scatter(x=self.lakes['Lon'], y=self.lakes['Lat'], color = 'darkviolet', marker='o', s=self.lakes['Length_m']/20*0.35, lw=.0, transform=ccrs.PlateCarree(), alpha=0.75)
                melt = self.cart_map1.scatter(m_lons[melting>0], m_lats[melting>0], c=melting[melting>0]*1e3, marker='o', lw=0., s=self.dotsize,cmap = 'Reds', norm = Normalize(vmin=0.,vmax=3.), transform=ccrs.PlateCarree())
                self.scatter = self.cart_map1.scatter(self.lons[stagnant>0], self.lats[stagnant>0], c=stagnant[stagnant>0], norm=norm,  cmap= 'Blues', marker='o', lw=0., s=self.dotsize,transform=ccrs.PlateCarree())
                # removed for paper
                # melt = self.cart_map1.scatter(m_lons[melting>0], m_lats[melting>0], c=melting[melting>0]*1e3, marker='o', lw=0., s=self.dotsize,cmap = 'Reds', norm = Normalize(vmin=0.,vmax=3.), transform=ccrs.PlateCarree())
                # cb = plt.colorbar(melt, ax=self.cart_map1,  orientation='horizontal', shrink =0.7, pad=0.0)
                # cb.set_label('$\overline{m}$ (mm a$^{-1}$)')
                self.cb_label = 'Stagnant ice thickness (m)'

            elif MapLabel=='p':
                self.scatter = self.cart_map1.scatter(pp_lons, pp_lats, c=p, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = '$p$'

            elif MapLabel=='p-sigma':
                self.scatter = self.cart_map1.scatter(pp_lons, pp_lats, c=sigma_p, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label = '$\sigma$ p'

            elif MapLabel=='accu-steady':
                # norm=Normalize()
                print('accu', np.average(accu))
                # Normalize(vmin=self.limits[MapLabel]['vmin'],vmax=self.limits[MapLabel]['vmax'])
                self.cb_label = 'Steady accumulation (mm yr$^{-1}$)'
                self.scatter = self.cart_map1.scatter(accu_lons, accu_lats, c=accu*1e3, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())

            elif MapLabel=='accu-sigma':
                norm=Normalize()
                self.cb_label = 'sigma accumulation'
                self.scatter = self.cart_map1.scatter(accu_lons, accu_lats, c=accu_sigma, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())


            elif MapLabel[0:4]=='Accu':

                norm = Normalize()
                i=int(MapLabel[10:12])
                accu=self.accu_array[:,i+4]

                self.scatter = self.cart_map1.scatter(accu_lons, accu_lats, c=accu, marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cb_label='$\overline{a}$ (mm-we a$^{-1}$)'
                self.cb_ticks = 'auto'


            if MapLabel[0:7]=='age-hor':

                age=self.hor_array[:,int(MapLabel[7:9])+2]

                if np.all(np.isnan(age)):
                    norm=Normalize(vmin=0.,vmax=1.)
                else:
                    norm=Normalize(vmin=np.nanmin(age/1000.),vmax=np.nanmax(age/1000.))

                self.scatter = self.cart_map1.scatter(self.hor_array[:,0], self.hor_array[:,1], c=age/1000., marker='o', lw=0., norm = norm,  s=self.dotsize,transform=ccrs.PlateCarree())
                self.cblabel='age (ka B1950)'
                self.cb_ticks = 'auto'


            # add colorbar and points of interest
            self.finish_map(MapLabel)





my_map = Maps()
if my_map.run_model:
    print(my_map.list_RL)
    for i,RLlabel in enumerate(my_map.list_RL):
        directory=my_map.RLDir+RLlabel
        sys.argv=['age_model.py', directory]
        exec(open('age_model.py').read())
        plt.close("all")
my_map.next()
my_map.make_maps()
