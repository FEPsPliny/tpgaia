import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
import xml.etree.ElementTree as ET
import warnings
import os
import htof
import glob
from datetime import datetime
#import exoplanet as xo
from astropy import constants
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.coordinates import get_body_barycentric
import pymc3 as pm
import pymc3_ext as pmx
import arviz as az
#from tabulate import tabulate
from aesara_theano_fallback import aesara as theano
import aesara_theano_fallback.tensor as tt
from pymc3 import find_MAP
import exoplanet as xo
import astropy.constants as sc
import pickle

root_savepath = os.path.join(os.path.dirname(__file__),'data')

from htof.sky_path import parallactic_motion, earth_ephemeris, earth_sun_l2_ephemeris
from htof import parse
from scipy import interpolate

import theano
theano.config.exception_verbosity = 'high'
# conversion constant from au to R_sun
au_to_R_sun = (constants.au / constants.R_sun).value
_radtomas = (180 * 3600 * 1000) / np.pi
_mastorad = np.pi / (180 * 3600 * 1000)
_kmps_to_aupyr = (u.year.to(u.s) * u.km.to(u.m)) / constants.au.value

import requests
import xml.etree.ElementTree as ET
import warnings
import pandas as pd

from astropy.coordinates.sky_coordinate import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs

# Unit conversions related to HTOF functions
_degtorad = np.pi/180
_radtomas = (180 * 3600 * 1000) / np.pi
_mastorad = np.pi / (180 * 3600 * 1000)
_kmps_to_aupyr = (u.year.to(u.s) * u.km.to(u.m)) / constants.au


class AstrometricModel():
    
    def __init__(self, starname, overwrite=False, save_file_loc=None, **kwargs):
        """
        Initialising astrometric model
        """
        self.starname=starname
        
        #Setting some default paramaters
        self.defaults={'overwrite':False,       # overwrite - bool - Whether to wipe past planets
                       'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'load_from_file':False,  # load_from_file - bool - Load previous model?
                       'planet_mass_prior':'log',# planet_mass_prior - str - Whether to have a 'log' or 'linear' mass prior. Default is log
                       'ecc_prior':'kipping',   # ecc_prior - string - 'kipping' or 'sqrtesinomega'. If 'auto' we decide based on multiplicity
                       'omegas_prior':'p&m',    # omegas_prior - string - 'p&m' or 'omega&Omega'. The two reparameterisations of omega, with p&m more typically used in astrometric fits
                       'n_tune':3000,           # n_tune - int - Number of tuning samples per chain. Should be 2000<n<20000
                       'n_draws':10000,         # n_samples - int - Number of samples per chain. Should be 10000<n<100000
                       'n_chains':4,            # n_chains - int - Number of HMC chains. Should be 4<n<20
                       'n_cores':4,             # n_cores - int - Number of CPU cores
                       'regularization_steps':4,# regularisation_steps - int - Number of steps to perform regularisation - necessary in highly degenerate parameter spaces
                       'data_release':"DR5",    # data_release - str - Which Gaia Data Release to use. Default = DR5 (i.e. all 10yrs of data)
                       'fine_sampling':False,   # fine_sampling - bool - Whether to include fine (daily) modelling of the orbital parameters
                      }
        self.update_global_params(**kwargs)
        
        if self.load_from_file and not self.overwrite:
            #Catching the case where the file doesnt exist:
            success = self.load_model(loadfile=save_file_loc)
            self.load_from_file = success

        
    def update_global_params(self,**kwargs):
        """
        Updating the global (i.e. default) settings
        """
        for param in self.defaults:
            if not hasattr(self,param) or self.overwrite:
                if param in kwargs:
                    setattr(self,param,kwargs[param])
                else:
                    setattr(self,param,self.defaults[param])

    def star_from_tic(self, tic=None, search_rad=1*u.arcsec,vrad=0,RA_offset_mas=0,DEC_offset_mas=0,randomise_true_params=True,**kwargs):
        """
        Adding stellar parameters direct from TESS input catalogue
        """
        self.update_global_params(**kwargs)
        
        if tic==None:
            #Sorting out the name - hoping it is in
            assert type(self.starname)==int or "TIC" in self.starname or self.starname[0] in "0123456789"
            tic=self.starname[3:] if "TIC" in self.starname else self.starname
        tic=int(tic)
        print("Assuming TIC="+str(tic))
        
        self.stardat=Catalogs.query_object("TIC"+str(int(tic)), catalog="TIC", radius=search_rad).to_pandas()
        #print(tess_cat)
        if self.stardat.shape[0]>1:
            self.stardat=self.stardat.loc[self.stardat['ID'].values.astype(int)==tic]
        self.stardat=self.stardat.iloc[0] if type(self.stardat)==pd.DataFrame else self.stardat
        self.GAIAmag=self.stardat['GAIAmag']
        self.dist_pc=self.stardat['d']
        self.edist_pc=self.stardat['e_d']
        self.rad_rsun=self.stardat['rad']
        self.erad_rsun=self.stardat['e_rad']
        self.mass_msun=self.stardat['mass']
        self.emass_msun=self.stardat['e_mass']
        self.pmRA_masyr=self.stardat['pmRA']
        self.epmRA_masyr=self.stardat['e_pmRA']
        self.pmDEC_masyr=self.stardat['pmDEC']
        self.epmDEC_masyr=self.stardat['e_pmDEC']
        self.RA_deg=self.stardat['ra']
        self.eRA_mas=self.stardat['e_RA']
        self.DEC_deg=self.stardat['dec']
        self.eDEC_mas=self.stardat['e_Dec']
        self.plx_mas=self.stardat['plx']
        self.eplx_mas=self.stardat['e_plx']
        self.radec = SkyCoord(self.stardat['ra']*u.deg, self.stardat['dec']*u.deg,
                              pm_ra_cosdec=self.stardat['pmRA']*u.mas/u.yr,
                              pm_dec=self.stardat['pmDEC']*u.mas/u.yr, equinox='J2015.5')
        self.vrad_kms=vrad
        self.RA_offset_mas=RA_offset_mas
        self.DEC_offset_mas=DEC_offset_mas
        
        for col in ['plx_mas','rad_rsun','mass_msun','pmRA_masyr','pmDEC_masyr','RA_offset_mas','DEC_offset_mas']:
            if hasattr(self,'e'+col) and getattr(self,'e'+col) is not None and randomise_true_params:
                setattr(self,'true_'+col,abs(np.random.normal(getattr(self,col),getattr(self,'e'+col))))
            elif 'offset_mas' in col and randomise_true_params:
                setattr(self,'true_'+col,abs(np.random.normal(getattr(self,col),getattr(self,'e'+col.split('_')[0]+'_mas'))))
            else:
                setattr(self,'true_'+col,getattr(self,col))

    def init_star(self, name, GAIAmag,
                  RA_deg,eRA_mas,
                  DEC_deg,eDEC_mas,
                  plx_mas, eplx_mas, 
                  rad_rsun, erad_rsun, 
                  mass_msun, emass_msun, 
                  pmRA_masyr, epmRA_masyr, 
                  pmDEC_masyr, epmDEC_masyr,
                  vrad_kms=0,RA_offset_mas=0,DEC_offset_mas=0,
                  randomise_true_params=True,**kwargs):
        """
        Manually adding stellar parameters
        """
        self.update_global_params(**kwargs)
        self.starname=name
        argdict = dict(locals())
        #[arg] for arg in 'RA_deg,eRA_deg,DEC_deg,eDEC_deg,dist_pc,edist_pc,rad_rsun,erad_rsun,emass_msun,mass_msun,pmRA_masyr,epmRA_masyr,pmDEC_masyr,epmDEC_masyr,vrad_kms,RA_offset_mas,DEC_offset_mas'.split(',')}
        for key in argdict:
            if key not in ['self','name','kwargs','randomise_true_params']:
                setattr(self,key,argdict[key])
        for col in ['plx_mas','rad_rsun','mass_msun','pmRA_masyr','pmDEC_masyr','RA_offset_mas','DEC_offset_mas']:
            if hasattr(self,'e'+col) and getattr(self,'e'+col) is not None and randomise_true_params:
                setattr(self,'true_'+col,abs(np.random.normal(getattr(self,col),getattr(self,'e'+col))))
            elif 'offset_mas' in col and randomise_true_params:
                setattr(self,'true_'+col,abs(np.random.normal(getattr(self,col),getattr(self,'e'+col.split('_')[0]+'_mas'))))
            else:
                setattr(self,'true_'+col,getattr(self,col))

    def get_Gaia_error(self):
        """
        Using the interpolation function to estimate along scan RMS error.
        """
        assert hasattr(self,"GAIAmag"), "Must have Gaia magnitude"
        
        ast_std=pd.read_csv("Gaia_AlongScan_Uncertainties.csv",header=None)
        interp=interpolate.interp1d(ast_std[0].values,ast_std[1].values)
        self.gaia_yerr_mas=np.tile(interp(self.GAIAmag),len(self.gaia_t))
        if hasattr(self,'planets') and len(self.planets)>0:
            for plname in self.planets:
                self.planets[plname]['SNR_alongscanrms'] = self.planets[plname]['astrometric_signal_mas']/self.gaia_yerr_mas[0]

    def init_gaia_data(self,**kwargs):
        """
        Access Gaia observing data (e.g. times and scan angles) via GOST.
        """
        self.update_global_params(**kwargs)
        
        assert hasattr(self,'RA_deg') and hasattr(self,'DEC_deg'),"Must have RA & Dec, e.g. using init_star"
        
        if not hasattr(self,'savenames'):
            self.get_savename(how='save')
        
        if not os.path.exists(self.savenames[1]+"GOST_"+self.starname+"_dat.csv"):
            url = f"https://gaia.esac.esa.int/gost/GostServlet?ra="+str(self.RA_deg)+"&dec="+str(self.DEC_deg)+"&service=1"
            try:
                with requests.Session() as s:
                    s.get(url)
                    headers = {"Cookie": f"JSESSIONID={s.cookies.get_dict()['JSESSIONID']}"}
                    response = s.get(url, headers=headers, timeout=180)
                    dat = response.text
            except:
                warnings.warn("Querying the GOST service failed.")
                dat = None

            try:
                root = ET.fromstring(dat)
            except:
                warnings.warn("The GOST service returned an invalid xml file.")
            columns = ["Target", "ra[rad]", "dec[rad]", "ra[h:m:s]", "dec[d:m:s]", "ObservationTimeAtGaia[UTC]",
                        "CcdRow[1-7]", "zetaFieldAngle[rad]", "scanAngle[rad]", "Fov[FovP=preceding/FovF=following]",
                        "parallaxFactorAlongScan", "parallaxFactorAcrossScan", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
            rows = []
            name = root.find('./targets/target/name').text
            raR = root.find('./targets/target/coords/ra').text
            decR = root.find('./targets/target/coords/dec').text
            raH = root.find('./targets/target/coords/raHms').text
            decH = root.find('./targets/target/coords/decDms').text
            for event in root.findall('./targets/target/events/event'):
                details = event.find('details')
                observationTimeAtGaia = event.find('eventUtcDate').text
                ccdRow = details.find('ccdRow').text
                zetaFieldAngle = details.find('zetaFieldAngle').text
                scanAngle = details.find('scanAngle').text
                fov = details.find('fov').text
                parallaxFactorAl = details.find('parallaxFactorAl').text
                parallaxFactorAc = details.find('parallaxFactorAc').text
                observationTimeAtBarycentre = event.find('eventTcbBarycentricJulianDateAtBarycentre').text
                rows.append([name, raR, decR, raH, decH, observationTimeAtGaia, ccdRow,
                              zetaFieldAngle, scanAngle, fov, parallaxFactorAl, parallaxFactorAc, observationTimeAtBarycentre])
            self.gaia_obs_data = pd.DataFrame(rows, columns=columns)
            self.gaia_obs_data = self.gaia_obs_data.astype({"Target": str,"ra[rad]": float, "dec[rad]": float,"ra[h:m:s]": str,"dec[d:m:s]": str,"ObservationTimeAtGaia[UTC]": str,"CcdRow[1-7]": int,"zetaFieldAngle[rad]": float,"scanAngle[rad]": float,"Fov[FovP=preceding/FovF=following]": str,"parallaxFactorAlongScan": float,"parallaxFactorAcrossScan": float,"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": float })
            self.gaia_obs_data.to_csv(self.savenames[1]+"GOST_"+self.starname+"_dat.csv")
        else:
            #Loading from file
            print("Loading GOST data from file "+self.savenames[1]+"GOST_"+self.starname+"_dat.csv")
            self.gaia_obs_data = pd.read_csv(self.savenames[1]+"GOST_"+self.starname+"_dat.csv")
        
        if self.data_release=="DR4":
            self.gaia_obs_data = self.gaia_obs_data[(self.gaia_obs_data["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"] >= 2456853.5) & (self.gaia_obs_data["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"] <= 2459229.5)]
        # The Gaia data is now loaded in as "gaia_obs_data" - accessing most important columns:
        self.gaia_t=Time(self.gaia_obs_data["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"],format='jd')
        self.gaia_tfine = Time(np.arange(np.min(self.gaia_t).jd,np.max(self.gaia_t).jd,1),format='jd')
        self.gaia_scanAng_rad=self.gaia_obs_data["scanAngle[rad]"].values
        
        # Normal triad, defined at the reference epoch.
        self.p = np.array([-np.sin(self.RA_deg*_degtorad), np.cos(self.RA_deg*_degtorad), 0.0])
        self.q = np.array([-np.sin(self.DEC_deg*_degtorad) * np.cos(self.RA_deg*_degtorad), -np.sin(self.DEC_deg*_degtorad) * np.sin(self.RA_deg*_degtorad), np.cos(self.DEC_deg*_degtorad)])
        self.r = np.array([np.cos(self.DEC_deg*_degtorad) * np.cos(self.RA_deg*_degtorad), np.cos(self.DEC_deg*_degtorad) * np.sin(self.RA_deg*_degtorad), np.sin(self.DEC_deg*_degtorad)])

        self.bO_bcrs = earth_sun_l2_ephemeris(self.gaia_t.jyear)

        # Calculate the Roemer delay, take units into account.
        self.tB = self.gaia_t + np.dot(self.r, self.bO_bcrs) * constants.au / constants.c / u.year.to(u.s)
        self.uO_init = np.repeat(self.r, self.gaia_t.size).reshape((self.r.size, self.gaia_t.size))
        if self.fine_sampling:
            self.bO_bcrs_fine=earth_sun_l2_ephemeris(self.gaia_tfine.jyear)
            self.tB_fine = self.gaia_tfine + np.dot(self.r, self.bO_bcrs_fine) * constants.au / constants.c / u.year.to(u.s)
            self.uO_init_fine = np.repeat(self.r, self.gaia_tfine.size).reshape((self.r.size, self.gaia_tfine.size))
        
        #Also initialising the expected Gaia error:
        self.get_Gaia_error()
    
    def init_planet(self,plname,
                    mpl_mjup,empl_mjup,
                    per_d,eper_d,
                    t0_jd,et0_jd,
                    b,eb,
                    ecc=None,eecc=None,
                    little_omega_rad=None,elittle_omega_rad=None,
                    big_Omega_rad=None,ebig_Omega_rad=None,randomise_true_params=True,**kwargs):
        """
        Initialising a planet from vital parameters. 
        These are used as inputs to the true model.
        The values are also used to build the PyMC3 parameters,
        however `randomise_true_params=True` means the "guess" values for these distributions vary Normally.
        """
        
        self.update_global_params(**kwargs)
        
        if not hasattr(self,'planets'):
            self.planets={}
        self.planets[plname]={}
        #argdict = {arg: locals()[arg] for arg in 'mpl_mjup,empl_mjup,per_d,eper_d,t0_jd,et0_jd,b,eb,ecc,eecc,little_omega_rad,elittle_omega_rad,big_Omega_rad,ebig_Omega_rad'.split(',')}        
        argdict=dict(locals())
        for key in argdict:
            if key not in ["self","plname","kwargs","randomise_true_params"]:
                self.planets[plname][key]=argdict[key]
                
        self.planets[plname]['ecc']=0.1 if 'ecc' in self.planets[plname] or self.planets[plname]['ecc']==None else self.planets[plname]['ecc']
        self.planets[plname]['little_omega_rad']=np.random.random()*2*np.pi-np.pi if self.planets[plname]['little_omega_rad']==None else self.planets[plname]['little_omega_rad']
        self.planets[plname]['big_Omega_rad']=np.random.random()*2*np.pi-np.pi if self.planets[plname]['big_Omega_rad']==None else self.planets[plname]['big_Omega_rad']
        self.planets[plname]['sma']=((((self.planets[plname]['per_d']*u.d)**2*sc.G*self.mass_msun*u.Msun)/(4*np.pi**2))**(1/3.)).to(u.au).value
        self.planets[plname]['astrometric_signal_mas'] = (((self.planets[plname]['mpl_mjup']*sc.M_jup/sc.M_sun * self.planets[plname]['sma'])/(self.dist_pc * self.mass_msun))*1000).value
        if hasattr(self,'gaia_yerr_mas'):
            self.planets[plname]['SNR_alongscanrms'] = self.planets[plname]['astrometric_signal_mas']/self.gaia_yerr_mas[0]
        
        #Randomising the true parameters such that they use the independently determined values/errors:
        for col in ['mpl_mjup','per_d','t0_jd','b','ecc','little_omega_rad','big_Omega_rad']:
            if 'e'+col in self.planets[plname] and self.planets[plname]['e'+col] is not None and randomise_true_params:
                self.planets[plname]['true_'+col]=abs(np.random.normal(self.planets[plname][col],self.planets[plname]['e'+col]))
            else:
                self.planets[plname]['true_'+col]=self.planets[plname][col]

    def init_model(self, w_planets=True, w_loglik=True, **kwargs):
        """
        Initialising PyMC3 model. Can be performed with and without planets. Creates pm_mod_w_planets and pm_mod_no_planets arguments
        """
        self.update_global_params(**kwargs)
        
        assert hasattr(self,'gaia_obs_data'), "Must have initialised Gaia observation data first"
        
        if not hasattr(self,'gaia_y_mas') and w_loglik:
            self.init_injected_planet_model()
        
        modname = "pm_mod_w_planets" if w_planets else "pm_mod_no_planets"
        model_params={}
        if hasattr(self,modname):
            delattr(self,modname)
            delattr(self,modname+"_params")
        with pm.Model() as mod:            
            #initialising star
            model_params['rad_rsun'] = pm.Normal('rad_rsun', mu=self.rad_rsun, sd =self.erad_rsun)
            model_params['mass_msun'] = pm.Normal('mass_msun', mu=self.mass_msun, sd = self.emass_msun)
            model_params['plx_mas'] = pm.Normal('plx_mas', mu=self.plx_mas, sd=self.eplx_mas)
            model_params['RA_offset_mas'] = pm.Normal("RA_offset_mas", mu=0, sd=self.eRA_mas)
            model_params['DEC_offset_mas'] = pm.Normal("DEC_offset_mas", mu=0, sd=self.eDEC_mas)
            model_params['pmRA_masyr'] = pm.Normal("pmRA_masyr", mu=self.pmRA_masyr, sigma=self.epmRA_masyr)
            model_params['pmDEC_masyr'] = pm.Normal("pmDEC_masyr", mu=self.pmDEC_masyr, sigma=self.epmDEC_masyr)

            #initialising planets
            if w_planets:
                model_params['t0_jds']={};model_params['per_ds']={};model_params['bs']={};
                model_params['logmpl_mjups']={};model_params['mpl_mjups']={};model_params['orbits']={};
                model_params['Deltara_pl']={};model_params['Deltadec_pl']={};self.plfitparams={}
                if self.fine_sampling:
                    model_params['Deltara_pl_fine']={};model_params['Deltadec_pl_fine']={};
                if not self.assume_circ:
                    model_params['eccs']={};model_params['ecs']={};model_params['little_omega_rads']={};model_params['big_Omega_rads']={}
                    if self.omegas_prior=='p&m':
                        model_params['omega_ps']={};model_params['omega_ms']={}

                # Defining prior probability distributions for all orbital parameters.
                for pl in self.planets:
                    model_params['t0_jds'][pl] = pm.Normal('t0_jd_'+pl, mu=self.planets[pl]['t0_jd'], sd=self.planets[pl]['et0_jd'])
                    model_params['per_ds'][pl] = pm.Normal('per_d_'+pl, mu=self.planets[pl]['per_d'], sd=self.planets[pl]['eper_d'])
                    model_params['bs'][pl] = pm.Normal('b_'+pl, mu=self.planets[pl]['b'], sd=self.planets[pl]['eb'])
                    self.plfitparams[pl]=[model_params['t0_jds'][pl],model_params['per_ds'][pl],model_params['bs'][pl]]
                    if not self.assume_circ:
                        
                        # For non-zero eccentricity, it can sometimes be better to use
                        # sqrt(e)*sin(omega) and sqrt(e)*cos(omega) as your parameters:
                        #
                        if self.ecc_prior=="sqrtesinomega":
                            model_params['ecs'][pl] = pmx.UnitDisk("ecs_"+pl)
                            self.plfitparams[pl]+=[model_params['ecs'][pl]]
                            model_params['eccs'][pl] = pm.Deterministic("ecc_"+pl, tt.sum(model_params['ecs'][pl] ** 2, axis=0))
                            model_params['little_omega_rads'][pl] = pm.Deterministic("little_omega_rad_"+pl, tt.arctan2(model_params['ecs'][pl][1], model_params['ecs'][pl][0]))
                            if self.omegas_prior=='p&m':
                                model_params['omega_ps'][pl]=pmx.Angle("omega_p_"+pl)
                                self.plfitparams[pl]+=[model_params['omega_ps'][pl]]
                                model_params['big_Omega_rads'][pl]=pm.Deterministic("big_Omega_rad_"+pl,2*model_params['omega_ps'][pl]-model_params['little_omega_rads'][pl])
                                model_params['omega_ms'][pl] = pm.Deterministic('omega_ms',0.5*(model_params['big_Omega_rads'][pl]-model_params['little_omega_rads'][pl]))
                            else:
                                model_params['big_Omega_rads'][pl] = pmx.Angle('big_Omega_rad_'+pl)
                                self.plfitparams[pl]+=[model_params['big_Omega_rads'][pl]]
                        else:
                            if self.ecc_prior=='auto' or self.ecc_prior=='kipping':
                                model_params['eccs'][pl] = pm.Beta('ecc_'+pl, alpha=1.12, beta=3.09)
                            elif self.ecc_prior=='vaneylen':
                                model_params['eccs'][pl] = xo.distributions.eccentricity.vaneylen19('ecc_'+pl, fixed=True, testval=0.05)
                            self.plfitparams[pl]+=[model_params['eccs'][pl]]
                            if self.omegas_prior=='p&m':
                                model_params['omega_ps'][pl]=pmx.Angle("omega_p_"+pl)
                                model_params['omega_ms'][pl]=pmx.Angle("omega_m_"+pl)
                                self.plfitparams[pl]+=[model_params['omega_ps'][pl],model_params['omega_ms'][pl]]
                                model_params['little_omega_rads'][pl]=pm.Deterministic("little_omega_rad_"+pl, model_params['omega_ps'][pl]-model_params['omega_ms'][pl])
                                model_params['big_Omega_rads'][pl]=pm.Deterministic("big_Omega_rad_"+pl,
                                                                                    model_params['omega_ps'][pl] + model_params['omega_ms'][pl])

                            else:
                                model_params['little_omega_rads'][pl] = pmx.Angle('little_omega_rad_'+pl)
                                model_params['big_Omega_rads'][pl] = pmx.Angle('big_Omega_rad_'+pl)
                                self.plfitparams[pl]+=[model_params['little_omega_rads'][pl],model_params['big_Omega_rads'][pl]]
                    if self.planet_mass_prior=="log":
                        model_params['logmpl_mjups'][pl] = pm.Uniform('logmpl_mjup_'+pl, lower=np.log(1/300), upper=np.log(150))
                        self.plfitparams[pl]+=[model_params['logmpl_mjups'][pl]]
                        model_params['mpl_mjups'][pl] = pm.Deterministic("mpl_mjup_"+pl, tt.exp(model_params['logmpl_mjups'][pl]))
                    elif self.planet_mass_prior=="linear":
                        model_params['mpl_mjups'][pl] = pm.Uniform('mpl_mjup_'+pl, lower=1/300, upper=150)
                        self.plfitparams[pl]+=[model_params['mpl_mjups'][pl]]
                #Instantiating Orbit
                for pl in self.planets:
                    if not self.assume_circ:
                        model_params['orbits'][pl] = xo.orbits.KeplerianOrbit(
                            t0=model_params['t0_jds'][pl], period=model_params['per_ds'][pl], incl=None, b=model_params['bs'][pl], e = model_params['eccs'][pl],
                            omega=model_params['little_omega_rads'][pl], Omega=model_params['big_Omega_rads'][pl], r_star=model_params['rad_rsun'], m_star=model_params['mass_msun'],
                            m_planet=model_params['mpl_mjups'][pl], m_planet_units=u.Mjup
                            )
                    else:
                        model_params['orbits'][pl] = xo.orbits.KeplerianOrbit(
                            t0=model_params['t0_jds'][pl], period=model_params['per_ds'][pl], incl=None, b=model_params['bs'][pl], 
                            r_star=model_params['rad_rsun'], m_star=model_params['mass_msun'], m_planet=model_params['mpl_mjups'][pl], m_planet_units=u.Mjup
                            )

                    if self.fine_sampling:
                        # Exoplanet motion. Automatically in same units as parallax
                        model_params['Deltara_pl_fine'][pl] = pm.Deterministic("Deltara_pl_fine_"+pl, model_params['orbits'][pl].get_star_position(self.gaia_tfine.jd, parallax=model_params['plx_mas'])[0]) # R.A. motion for a fine grid of time points
                        model_params['Deltadec_pl_fine'][pl] = pm.Deterministic("Deltadec_pl_fine_"+pl, model_params['orbits'][pl].get_star_position(self.gaia_tfine.jd, parallax=model_params['plx_mas'])[1]) # Dec. motion for a fine grid of time points
                    model_params['Deltara_pl'][pl] = pm.Deterministic("Deltara_pl_"+pl, model_params['orbits'][pl].get_star_position(self.gaia_t.jd, parallax=model_params['plx_mas'])[0]) # Best fit exoplanet motion in R.A
                    model_params['Deltadec_pl'][pl] = pm.Deterministic("Deltadec_pl_"+pl, model_params['orbits'][pl].get_star_position(self.gaia_t.jd, parallax=model_params['plx_mas'])[1]) # # Best fit exoplanet motion in dEC.
            #tt.printing.Print("planet_stack_dec")(tt.sum(tt.stack([model_params['Deltadec_pl'][pl] for pl in self.planets],axis=0),axis=0))
            #tt.printing.Print("planet_stack_ra")(tt.sum(tt.stack([model_params['Deltara_pl'][pl] for pl in self.planets],axis=0),axis=0))

            # Local plane coordinates which approximately equal delta_alpha*cos(delta) and delta_delta
            if self.fine_sampling:
                model_params['uO_fine'] = self.uO_init_fine[:,:] + tt.tensordot((self.p*model_params['pmRA_masyr']*_mastorad + self.q*model_params['pmDEC_masyr']*_mastorad), 
                                             (self.tB_fine.jyear-2015.5), axes=0) - model_params['plx_mas']*_mastorad*self.bO_bcrs_fine
                model_params['Deltara_par_pm_fine'] = pm.Deterministic("Deltara_par_pm_fine", tt.dot(self.p, model_params['uO_fine'])/tt.dot(self.r, model_params['uO_fine'])/_mastorad) # Xi in the parallactic motion = local Delta_ra*cos(dec) (in rad - need to convert to mas)
                model_params['Deltadec_par_pm_fine'] = pm.Deterministic("Deltadec_par_pm_fine", tt.dot(self.q, model_params['uO_fine'])/tt.dot(self.r, model_params['uO_fine'])/_mastorad) # Eta in the parallactic motion = local Delta_dec (in rad - need to convert to mas)
            model_params['uO'] = self.uO_init[:,:] + tt.tensordot((self.p*model_params['pmRA_masyr']*_mastorad + self.q*model_params['pmDEC_masyr']*_mastorad), 
                                             (self.tB.jyear-2015.5), axes=0) - model_params['plx_mas']*_mastorad*self.bO_bcrs
            model_params['Deltara_par_pm'] = pm.Deterministic("Deltara_par_pm", tt.dot(self.p, model_params['uO'])/tt.dot(self.r, model_params['uO'])/_mastorad) # Xi in the parallactic motion = local Delta_ra*cos(dec) (in rad - need to convert to mas)
            model_params['Deltadec_par_pm'] = pm.Deterministic("Deltadec_par_pm", tt.dot(self.q, model_params['uO'])/tt.dot(self.r, model_params['uO'])/_mastorad) # Eta in the parallactic motion = local Delta_dec (in rad - need to convert to mas)
            
            if w_planets:
                #Objective function:
                model_params['ymodel'] = pm.Deterministic("ymodel", (model_params['RA_offset_mas']+model_params['Deltara_par_pm']+tt.sum(tt.stack([model_params['Deltadec_pl'][pl] for pl in self.planets],axis=0),axis=0))*tt.sin(self.gaia_scanAng_rad)+(model_params['DEC_offset_mas']+model_params['Deltadec_par_pm']+tt.sum(tt.stack([model_params['Deltara_pl'][pl] for pl in self.planets],axis=0),axis=0))*tt.cos(self.gaia_scanAng_rad))
            else:
                model_params['ymodel'] = pm.Deterministic("ymodel", (model_params['RA_offset_mas']+model_params['Deltara_par_pm'])*tt.sin(self.gaia_scanAng_rad)+(model_params['DEC_offset_mas']+model_params['Deltadec_par_pm'])*tt.cos(self.gaia_scanAng_rad))
            
            if hasattr(self,'gaia_y_mas') and w_loglik:
                model_params['log_likelihood']=pm.Normal("obs", mu=model_params['ymodel'], sd=self.gaia_yerr_mas, observed=self.gaia_y_mas)
            
        #Making sure we set the pymc3 model as "pm_mod_w_planets"/"pm_mod_no_planets":
        setattr(self,modname,mod)
        setattr(self,modname+"_params",model_params)
        
    def init_injected_planet_model(self,**kwargs):
        """
        Initialise injected planet model. This uses the pymc3 model built above.
        """
        self.update_global_params(**kwargs)
        if not hasattr(self,"pm_mod_w_planets"):
            self.init_model(w_planets=True, w_loglik=False)
        init_dict={s:getattr(self,"true_"+s) for s in 'rad_rsun,mass_msun,plx_mas,RA_offset_mas,DEC_offset_mas,pmRA_masyr,pmDEC_masyr'.split(',')}
        for pl in self.planets:
            init_dict.update({s+"_"+pl:self.planets[pl]['true_'+s] for s in ['t0_jd','per_d','b']})
            if not self.assume_circ:
                #pmx.distributions.UnitDisk.dist(testval=0.01*np.ones(2)).transform.forward(0.01*np.ones(2)).eval()
                if self.ecc_prior=="sqrtesinomega":
                    init_dict.update({'ecs_'+pl+'_unitdisk+interval__': pmx.distributions.UnitDisk.dist().transform.forward(np.array([np.sqrt(self.planets[pl]['true_ecc'])*np.sin(self.planets[pl]['true_little_omega_rad']), np.sqrt(self.planets[pl]['true_ecc'])*np.cos(self.planets[pl]['true_little_omega_rad'])])).eval()})
                    if self.omegas_prior=='p&m':
                        init_dict.update({'omega_p_'+pl+'_angle__':(np.sin(self.planets[pl]['true_little_omega_rad']), np.cos(self.planets[pl]['true_little_omega_rad']))})
                    else:
                        init_dict.update({'big_Omega_rad_'+pl+'_angle__':(np.sin(self.planets[pl]['true_big_Omega_rad']), np.cos(self.planets[pl]['true_big_Omega_rad']))})
                else:
                    init_dict.update({'ecc_'+pl+'_logodds__': pm.transforms.logodds.apply(pm.distributions.Beta.dist(alpha=1.12,beta=3.09,testval=self.planets[pl]['true_ecc'])).default()})
                    if self.omegas_prior=='p&m':
                        init_dict.update({'omega_p_'+pl+'_angle__':(np.sin(0.5*(self.planets[pl]['true_little_omega_rad']+self.planets[pl]['true_big_Omega_rad'])), np.cos(0.5*(self.planets[pl]['true_little_omega_rad']+self.planets[pl]['true_big_Omega_rad'])))})
                        init_dict.update({'omega_m_'+pl+'_angle__':(np.sin(0.5*(self.planets[pl]['true_big_Omega_rad']-self.planets[pl]['true_little_omega_rad'])), np.cos(0.5*(self.planets[pl]['true_big_Omega_rad']-self.planets[pl]['true_little_omega_rad'])))})
                    else:
                        init_dict.update({'little_omega_rad_'+pl+'_angle__':(np.sin(self.planets[pl]['true_little_omega_rad']), np.cos(self.planets[pl]['true_little_omega_rad']))})
                        init_dict.update({'big_Omega_rad_'+pl+'_angle__':(np.sin(self.planets[pl]['true_big_Omega_rad']), np.cos(self.planets[pl]['true_big_Omega_rad']))})

            if self.planet_mass_prior=='linear':
                init_dict.update({'mpl_mjup_'+pl+'_interval__':pm.transforms.Interval(1/300, 150).forward(self.planets[pl]['true_mpl_mjup']).eval()})
            else:
                init_dict.update({'logmpl_mjup_'+pl+'_interval__':pm.transforms.Interval(np.log(1/300), np.log(150)).forward(np.log(self.planets[pl]['true_mpl_mjup'])).eval()})
        with self.pm_mod_w_planets:
            self.true_gaia_y_w_planets = pmx.eval_in_model(self.pm_mod_w_planets_params['ymodel'],point=init_dict)
        self.gaia_y_mas = np.random.normal(self.true_gaia_y_w_planets,self.gaia_yerr_mas)

    def optimize_model(self, w_planets=True, iterate_test_mass=True, n_optimizations=10, **kwargs):
        """
        Optimizing the PyMC3 model.
        """
        self.update_global_params(**kwargs)
        if not hasattr(self,'true_gaia_y_w_planets'):
            self.init_injected_planet_model()

        if w_planets:
            if not hasattr(self,'pm_mod_w_planets') or 'log_likelihood' not in self.pm_mod_w_planets_params:
                self.init_model(w_planets=True)
            llk=-1e9
            for n_it in range(n_optimizations):
                # Because of degeneracies, we really need to start in a good place - performing multiple optimizations with multiple start points will help...
                with self.pm_mod_w_planets:
                    #First, getting a "test_point" and varying it, especially key planet params like mass, period, etc.
                    tp_start=self.pm_mod_w_planets.test_point
                    tp_start={col:tp_start[col]*np.random.normal(1.0,1e-5) for col in tp_start}
                    for pl in self.planets:
                        mplrand=abs(np.random.normal(self.planets[pl]['mpl_mjup'],self.planets[pl]['empl_mjup']))
                        tp_start['per_d_'+pl]=np.random.normal(self.planets[pl]['per_d'],self.planets[pl]['eper_d'])
                        if self.planet_mass_prior=='log':
                            tp_start['logmpl_mjup_'+pl+'_interval__']=pm.transforms.Interval(np.log(1/300), np.log(150)).forward(np.log(mplrand)).eval()
                        else:
                            tp_start['mpl_mjup_'+pl+'_interval__']=pm.transforms.Interval(np.log(1/300), np.log(150)).forward(mplrand).eval()
                        if not self.assume_circ:
                            erand=np.random.random()*0.66
                            omrand=np.random.random()*np.pi*2-np.pi
                            if self.ecc_prior=="sqrtesinomega":
                                tp_start['ecs_'+pl+'_unitdisk+interval__']= pmx.distributions.UnitDisk.dist(testval=0.01*np.ones(2)).transform.forward(np.array([np.sqrt(erand)*np.sin(erand), np.sqrt(erand)*np.cos(erand)])).eval()
                            elif self.ecc_prior=="kipping":
                                tp_start['little_omega_rad_'+pl+'_angle__']=(np.sin(omrand), np.cos(omrand))
                                tp_start['ecc_'+pl+'_logodds__']= pm.transforms.logodds.apply(pm.distributions.Beta.dist(alpha=1.12,beta=3.09,testval=erand)).default()
                    # Now optimizing, starting with the stellar params:
                    init_cols=[self.pm_mod_w_planets_params[col] for col in ['plx_mas','pmRA_masyr','pmDEC_masyr','RA_offset_mas','DEC_offset_mas']]
                    i_w_planets_init_soln = pmx.optimize(vars=init_cols,start=tp_start,progress=False)
                    #Then using key planet parameters:
                    i_w_planets_init_soln = pmx.optimize(start=i_w_planets_init_soln,vars=init_cols+sum(self.plfitparams.values(),[]),progress=False)
                    #Finally using all parameters:
                    i_w_planets_init_soln = pmx.optimize(start=i_w_planets_init_soln)
                    no_nans=np.all([~np.any(np.isnan(i_w_planets_init_soln[c])) for c in i_w_planets_init_soln])
                    illk=pmx.eval_in_model(self.pm_mod_w_planets.logpt,point=i_w_planets_init_soln)
                    if not np.isnan(illk) and no_nans and illk>llk:
                        #Checking if we have a valid and better solution - if so we update the llk and start-point
                        self.w_planets_init_soln=i_w_planets_init_soln
                        llk=illk
            if not hasattr(self,'pm_mod_no_planets') or 'log_likelihood' not in self.pm_mod_no_planets_params:
                self.init_model(w_planets=False)
            with self.pm_mod_no_planets:
                #Optimizing - starting with key stellar parameters
                init_cols=[self.pm_mod_no_planets_params[col] for col in ['plx_mas','pmRA_masyr','pmDEC_masyr','RA_offset_mas','DEC_offset_mas']]
                self.no_planets_init_soln = pmx.optimize(vars=init_cols)
                #Finally using all parameters:
                self.no_planets_init_soln = pmx.optimize(start=self.no_planets_init_soln)
    
    def assess_planet_detection(self):
        """
        How well-detected in the planet detection by the model? TBD
        """
        assert hasattr(self,'w_planets_trace'), "Must have sampled the with-planets model"
        
        # 1) Fitting a Gaussian to the planet detection
        
        
        # 2) Fitting a sigmoid curve to the planet detection
        
        
        # 3) Comparing the models
        
        
    
    def sample_model(self,w_planets=True,progressbar=True,**kwargs):
        """
        Use PYMC3_ext to sample the astrometric model
        """
        self.update_global_params(**kwargs)

        assert hasattr(self,'w_planets_init_soln'), "Must have initialised and optimized model first"
        
        if w_planets:
            with self.pm_mod_w_planets:
                self.w_planets_trace = pmx.sample(draws=self.n_draws, tune=self.n_tune, start=self.w_planets_init_soln,
                                                  chains=self.n_chains, cores=self.n_cores, 
                                                  progressbar = progressbar, regularization_steps=self.regularization_steps)
        else:
            with self.pm_mod_no_planets:
                self.no_planets_trace = pmx.sample(draws=self.n_draws, tune=self.n_tune, start=self.no_planets_init_soln,
                                                   chains=self.n_chains, cores=self.n_cores, 
                                                   progressbar = progressbar, regularization_steps=self.regularization_steps)
        if not hasattr(self,'savename'):
            self.get_savename(how='save')
        self.save_model()

    def make_summary(self):
        """
        Making summary DataFrame (and csvs) for each of the two models
        """
        if hasattr(self,'w_planets_trace'):
            var_names=[var for var in self.w_planets_trace.varnames if '__' not in var and np.product(self.w_planets_trace[var].shape)<6*np.product(self.w_planets_trace['rad_rsun'].shape)]
            self.w_planets_param_table = pm.summary(self.w_planets_trace,var_names=var_names,round_to=7)
            self.w_planets_param_table.to_csv(self.savenames[0]+'_trace_output_w_planets.csv')
        if hasattr(self,'no_planets_trace'):
            var_names=[var for var in self.no_planets_trace.varnames if '__' not in var and np.product(self.no_planets_trace[var].shape)<6*np.product(self.no_planets_trace['rad_rsun'].shape)]
            self.no_planets_param_table = pm.summary(self.no_planets_trace,var_names=var_names,round_to=7)
            self.no_planets_param_table.to_csv(self.savenames[0]+'_trace_output_no_planets.csv')
        
    def compare_models(self, ic='waic', **kwargs):
        """
        Comparing the two trace models using arviz comparison. By default uses the "waic" comparison. 
        """
        assert (hasattr(self,'w_planets_trace') and hasattr(self,'no_planets_trace')), "Must have initialised, optimized and sampled both w_planets=True and w_planets=False models"
        import arviz as az
        with self.pm_mod_w_planets:
            w_planets_posterior_predictive = pm.sample_posterior_predictive(self.w_planets_trace)
            w_planets_prior = pm.sample_prior_predictive(150)
            self.w_planets_idata_pymc3 = az.from_pymc3(
                self.w_planets_trace,
                prior=w_planets_prior,
                posterior_predictive=w_planets_posterior_predictive,
            )
        with self.pm_mod_no_planets:
            no_planets_posterior_predictive = pm.sample_posterior_predictive(self.no_planets_trace)
            no_planets_prior = pm.sample_prior_predictive(150)
            self.no_planets_idata_pymc3 = az.from_pymc3(
                self.no_planets_trace,
                prior=no_planets_prior,
                posterior_predictive=no_planets_posterior_predictive,
            )
        self.comparison = az.compare({'w_planets':self.w_planets_idata_pymc3,'no_planets':self.no_planets_idata_pymc3},ic=ic)
        self.comparison.to_csv(self.savenames[0]+'_model_comparison.csv')
        return self.comparison
#         self.no_planets_waic = pm.stats.waic(self.no_planets_trace)
#         self.w_planets_waic = pm.stats.waic(self.w_planets_trace)
#         self.delta_waic=self.no_planets_waic[0]-self.w_planets_waic[0]
#         waic_dic={'w_planets_elpd_waic':self.w_planets_waic[0],'w_planets_elpd_waic_SE':self.w_planets_waic[1],'w_planets_p_waic':self.w_planets_waic[2],
#                   'no_planets_elpd_waic':self.no_planets_waic[0],'no_planets_elpd_waic_SE':self.no_planets_waic[1],'no_planets_p_waic':self.no_planets_waic[2],
#                   'DeltaWAIC':self.delta_waic,"Waic_prefers":['with','without'][np.argmin([self.w_planets_waic[0],self.no_planets_waic[0]])]+" planet",
#                   'Waic_strength':['weak','moderate','strong'][np.searchsorted(np.array([0,5,25,1e9]),abs(self.delta_waic))-1]}
#         for pl in self.planets:
#             waic_dic.update({pl+'_'+col:self.planets[pl][col] for col in self.planets[pl]})
#         self.waic_info=pd.Series(waic_dic)
#         self.waic_info.to_csv(self.savenames[0]+'_waic_model_comparison.csv')
#         print("The Delta WAIC "+self.waic_info['Waic_strength']+"ly supports a model "+self.waic_info['Waic_prefers']+" this astrometric planet model.")
#         print("WAIC planet = "+str(self.waic_info['w_planets_elpd_waic'])+"  |  WAIC no planet = "+str(self.waic_info['no_planets_elpd_waic']))

    def run_all(self):
        """
        Running the entire code base for both with and withing planet models. Will only work if the `starname` is a TIC and a planet has beenb initialised
        """
        self.star_from_tic()
        self.init_gaia_data()

        assert hasattr(self,'planets')
        
        #Optimizing and initialising
        self.init_model(w_planets=False)
        self.optimize_model(w_planets=False)
        self.sample_model(w_planets=False)
        
        self.init_model(w_planets=True)
        self.optimize_model(w_planets=True)
        self.sample_model(w_planets=True)
        
        self.make_summary()
        self.compare_models()
        self.plot_corners()
        self.plot_planet_histograms()
        
    def get_savename(self, how='load', overwrite=None):
        """
        Adds unique savename prefixes to class (self.savenames) with two formats:
        '[save_file_loc]/name_[20YY-MM-DD]_[n]...'
        '[save_file_loc]/name_[n]...'

        Args:
            how (str, optional): 'load' or 'save'. Defaults to 'load'.
            overwrite (bool, optional): if how='save', whether to overwrite past save or not. Defaults to None.
        """
        if overwrite is None and hasattr(self,'overwrite'):
            overwrite=self.overwrite
        else:
            overwrite=True

        if not hasattr(self,'save_file_loc') or self.save_file_loc is None or overwrite:
            self.save_file_loc=os.path.join(root_savepath,self.starname)
        if not os.path.isdir(self.save_file_loc):
            os.system('mkdir '+self.save_file_loc)
        pickles=glob.glob(os.path.join(self.save_file_loc,self.starname+"*model.pickle"))
        pickles=[p for p in pickles if len(p.split('/')[-1].split('_'))==4]
        if how == 'load' and len(pickles)>1:
            #finding most recent pickle:
            date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
            datepickles=glob.glob(os.path.join(self.save_file_loc,self.starname+"_"+date+"_*model.pickle"))
            if len(datepickles)>1:
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            elif len(datepickles)==1:
                nsim=0
            elif len(datepickles)==0:
                print("problem - no saved mcmc files in correct format")
        elif how == 'load' and len(pickles)==1:
            date=pickles[0].split('_')[1]
            nsim=pickles[0].split('_')[2]
        else:
            #Either pickles is empty (no file to load) or we want to save a fresh file:
            #Finding unique
            date=datetime.now().strftime("%Y-%m-%d")
            datepickles=glob.glob(os.path.join(self.save_file_loc,self.starname+"_"+date+"_*"))
            if len(datepickles)==0:
                nsim=0
            elif overwrite:
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            else:
                #Finding next unused number with this date:
                nsim=1+np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
        self.savenames=[os.path.join(self.save_file_loc,self.starname+"_"+date+"_"+str(int(nsim))), os.path.join(self.save_file_loc,self.starname)]

    def plot_planet_histograms(self,log=True,minx=None,nbins=50,**kwargs):
        """
        Plot the classic planet-BD-star histogram
        """
        self.update_global_params(**kwargs)
        minx=150 if minx is None else minx
        bins={}
        for npl,pl in enumerate(self.planets):
            plt.subplot(len(self.planets),1,npl+1)
            if log:
                bins[pl]=plt.hist(self.w_planets_trace['mpl_mjup_'+pl],bins=np.geomspace(1/300,150,nbins),density=True)
                plt.xscale('log')
            else:
                bins[pl]=plt.hist(self.w_planets_trace['mpl_mjup_'+pl],bins=np.linspace(0,150,nbins),density=True)
            minx= np.min(bins[pl][1][:-1][bins[pl][0]>0]) if np.min(bins[pl][1][:-1][bins[pl][0]>0])<minx else minx
            plt.plot(np.tile(self.planets[pl]['true_mpl_mjup'],2),[0,np.max(bins[pl][0])],'--k',alpha=0.35,zorder=3)
            plt.text(self.planets[pl]['true_mpl_mjup']*1.015,np.max(bins[pl][0]),'true mass',fontsize=8,rotation=90,ha='left',va='top')
            plt.ylim(0,np.max(bins[pl][0]))
            
        for npl,pl in enumerate(self.planets):
            plt.subplot(len(self.planets),1,npl+1)
            if minx<13:
                plt.fill_between([minx,13],[0,0],np.tile(np.max(bins[pl][0]),2),zorder=-2,alpha=0.2)
                plt.text(minx+0.1,0.9*np.max(bins[pl][0]),"planets",ha='left',zorder=-1)
            else:
                minx=13
            if minx<80:
                plt.fill_between([13,80],[0,0],np.tile(np.max(bins[pl][0]),2),zorder=-2,alpha=0.2)
                plt.text(14,0.9*np.max(bins[pl][0]),"BDs",ha='left',zorder=-1)
            plt.fill_between([80,150],[0,0],np.tile(np.max(bins[pl][0]),2),zorder=-2,alpha=0.2)
            plt.text(149,0.9*np.max(bins[pl][0]),"stars",ha='right',zorder=-1)
            plt.xlim(minx,150)

        plt.xlabel("Planet mass [Mjup]")
        plt.savefig(self.savenames[0]+"_planet_hist.png",dpi=350)
            
    
    def plot_planet_corner(self,pl):
        """
        Creates a corner plot specifically for an individual planet paramter set
        """
        pars=['mpl_mjup','per_d','t0_jd','b']
        if not self.assume_circ:
            pars+=['ecc','little_omega_rad','big_Omega_rad']
        import corner
        fig=corner.corner(self.w_planets_trace,var_names=[p+'_'+pl for p in pars],
                          truths=[self.planets[pl]["true_"+col] for col in pars])
        fig.savefig(self.savenames[0]+"_planet"+pl+"_corner.png",dpi=350)
        
    def plot_star_corner(self):
        """
        Creates a corner plot specifically for the stellar parameters
        """
        pars=['plx_mas','rad_rsun','mass_msun','pmRA_masyr','pmDEC_masyr','RA_offset_mas','DEC_offset_mas']
        import corner
        fig=corner.corner(self.w_planets_trace,var_names=pars,truths=[getattr(self,"true_"+col) for col in pars])
        fig.savefig(self.savenames[0]+"_star_corner.png",dpi=350)
    
    def plot_corners(self):
        """
        Creates all the corner plots
        """

        for pl in self.planets:
            self.plot_planet_corner(pl)
        self.plot_star_corner()
        
    def plot_residual_timeseries(self):
        """
        Plots the residual timeseries for both with and without planet cases.
        """
        plt.subplot(211)
        plt.plot(self.gaia_t.jd,self.w_planets_init_soln['ymodel'],'.-',label="w/ planet model",alpha=0.6)
        plt.plot(self.gaia_t.jd,self.no_planets_init_soln['ymodel'],'.-',label="no planet model",alpha=0.6)
        plt.plot(self.gaia_t.jd,self.gaia_y_mas,'.',markersize=1.2,label="raw data",alpha=0.6)
        plt.ylabel("model [mas]")
        plt.legend()

        plt.subplot(212)
        plt.plot(self.gaia_t.jd,self.gaia_y_mas-np.nanmedian(self.w_planets_trace['ymodel'],axis=0),'.',markersize=1.2,label="w/ planet residuals",alpha=0.6)
        plt.plot(self.gaia_t.jd,0.3+self.gaia_y_mas-np.nanmedian(self.no_planets_trace['ymodel'],axis=0),'.',markersize=1.2,label="no planet residuals",alpha=0.6)
        plt.ylabel("residuals [mas]")
        plt.legend()
        
        plt.savefig(self.savenames[0]+"_resids_timeseries.png",dpi=350)
    
    def save_model(self,savefile=None,limit_size=False):
        """
        Saves entire model to file. This can be loaded using `mod.load_model`
        """

        if savefile is None:
            if not hasattr(self,'savenames'):
                self.get_savename(how='save')
            savefile=self.savenames[0]+'_model.pickle'

        #Loading from pickled dictionary
        saving={}
        n_bytes = 2**31
        max_bytes = 2**31-1
        bytes_out = pickle.dumps({d:self.__dict__[d] for d in self.__dict__ if 'pm_mod_' not in d})
        #bytes_out = pickle.dumps(self)
        with open(savefile, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
        del saving
        #pick=pickle.dump(self.__dict__,open(loadfile,'wb'))

    def load_model(self, loadfile=None):
        """Load a model object direct from file.

        Args:
            loadfile (str, optional): File to load from, otherwise it takes the default location using `GetSavename`. Defaults to None.

        Returns:
            bool: Whether the load is successful
        """
        if loadfile is None:
            self.get_savename(how='load')
            loadfile=self.savenames[0]+'_model.pickle'
        if os.path.exists(loadfile):
            #Loading from pickled dictionary
            pick=pickle.load(open(loadfile,'rb'))
            assert not isinstance(pick, monoModel)
            #print(In this case, unpickle your object separately)
            for key in pick:
                setattr(self,key,pick[key])
            return True
        else:
            return False