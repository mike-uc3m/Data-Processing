
from ism.src.initIsm import initIsm
import numpy as np
from common.io.writeToa import writeToa, readToa
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
import sys

class detectionPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

        # Initialise the random see for the PRNU and DSNU
        np.random.seed(self.ismConfig.seed)


    def compute(self, toa, band):

        self.logger.info("EODP-ALG-ISM-2000: Detection stage")

        # Irradiance to photons conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-2010: Irradiances to Photons")
        area_pix = self.ismConfig.pix_size * self.ismConfig.pix_size # [m2]
        toa = self.irrad2Phot(toa, area_pix, self.ismConfig.t_int, self.ismConfig.wv[int(band[-1])])

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [ph]")

        # Photon to electrons conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-2030: Photons to Electrons")
        toa = self.phot2Electr(toa, self.ismConfig.QE)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_ph2e:
            saveas_str = self.globalConfig.ism_toa_e + band
            writeToa(self.outdir, saveas_str, toa)

        # PRNU
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_prnu:

            self.logger.info("EODP-ALG-ISM-2020: PRNU")
            toa = self.prnu(toa, self.ismConfig.kprnu)

            self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

            if self.ismConfig.save_after_prnu:
                saveas_str = self.globalConfig.ism_toa_prnu + band
                writeToa(self.outdir, saveas_str, toa)

        # Dark-signal
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_dark_signal:

            self.logger.info("EODP-ALG-ISM-2020: Dark signal")
            toa = self.darkSignal(toa, self.ismConfig.kdsnu, self.ismConfig.T, self.ismConfig.Tref,
                                  self.ismConfig.ds_A_coeff, self.ismConfig.ds_B_coeff)

            self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

            if self.ismConfig.save_after_ds:
                saveas_str = self.globalConfig.ism_toa_ds + band
                writeToa(self.outdir, saveas_str, toa)

        # Bad/dead pixels
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_bad_dead:

            self.logger.info("EODP-ALG-ISM-2050: Bad/dead pixels")
            toa = self.badDeadPixels(toa,
                               self.ismConfig.bad_pix,
                               self.ismConfig.dead_pix,
                               self.ismConfig.bad_pix_red,
                               self.ismConfig.dead_pix_red)


        # Write output TOA
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_detection_stage:
            saveas_str = self.globalConfig.ism_toa_detection + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the detection phase [e-]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        #Cheking that difference is under the margin
        #---------------------------------------------------------------------------------
        toa_lucia=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/',self.globalConfig.ism_toa_detection + band + '.nc')

        self.toadiff(toa,toa_lucia,band)

        return toa


    def irrad2Phot(self, toa, area_pix, tint, wv):
        """
        Conversion of the input Irradiances to Photons
        :param toa: input TOA in irradiances [mW/m2]
        :param area_pix: Pixel area [m2]
        :param tint: Integration time [s]
        :param wv: Central wavelength of the band [m]
        :return: Toa in photons
        """
        toa=toa*1e-3 #Pasamos de mW a W
        Ein=toa*area_pix*tint
        Eph=self.constants.h_planck*self.constants.speed_light/wv
        toa_ph=Ein/Eph
        conv_factor=area_pix*tint/Eph

        with open(self.outdir+'Instrument module.txt', 'a') as f:
            f.write('Irradiance to photon conversion: ' + str(conv_factor)+'\n')
        return toa_ph

    def phot2Electr(self, toa, QE):
        """
        Conversion of photons to electrons
        :param toa: input TOA in photons [ph]
        :param QE: Quantum efficiency [e-/ph]
        :return: toa in electrons
        """
        toae=toa*QE
        for i in range(toae.shape[0]):
            for j in range(toae.shape[1]):
                if toae[i,j]>self.ismConfig.FWC:
                   toae[i,j]=self.ismConfig.FWC

        with open(self.outdir+'Instrument module.txt', 'a') as f:
            f.write('Photon to electron conversion: ' + str(QE)+'\n')
        return toae

    def badDeadPixels(self, toa,bad_pix,dead_pix,bad_pix_red,dead_pix_red):
        """
        Bad and dead pixels simulation
        :param toa: input toa in [e-]
        :param bad_pix: Percentage of bad pixels in the CCD [%]
        :param dead_pix: Percentage of dead pixels in the CCD [%]
        :param bad_pix_red: Reduction in the quantum efficiency for the bad pixels [-, over 1]
        :param dead_pix_red: Reduction in the quantum efficiency for the dead pixels [-, over 1]
        :return: toa in e- including bad & dead pixels
        """
        nbad = int(bad_pix/100*toa.shape[1])
        ndead = int(dead_pix/100*toa.shape[1])
        if nbad!=0:
            step_bad = int(toa.shape[1]/nbad)
            for i in np.arange(5,toa.shape[1],step_bad):
               toa[:,i]=toa[:,i]*(1-bad_pix_red)
        if ndead!=0:
            step_dead=int(toa.shape[1]/ndead)
            for i in np.arange(0,toa.shape[1],step_dead):
               toa[:,i]=toa[:,i]*(1-dead_pix_red)

        return toa

    def prnu(self, toa, kprnu):
        """
        Adding the PRNU effect
        :param toa: TOA pre-PRNU [e-]
        :param kprnu: multiplicative factor to the standard normal deviation for the PRNU
        :return: TOA after adding PRNU [e-]
        """
        PRNU=np.random.normal(0,1,toa.shape[1])*kprnu
        for i in range(PRNU.shape[0]):
            toa[:,i]=toa[:,i]*(1+PRNU[i])
        return toa


    def darkSignal(self, toa, kdsnu, T, Tref, ds_A_coeff, ds_B_coeff):
        """
        Dark signal simulation
        :param toa: TOA in [e-]
        :param kdsnu: multiplicative factor to the standard normal deviation for the DSNU
        :param T: Temperature of the system
        :param Tref: Reference temperature of the system
        :param ds_A_coeff: Empirical parameter of the model 7.87 e-
        :param ds_B_coeff: Empirical parameter of the model 6040 K
        :return: TOA in [e-] with dark signal
        """
        DSNU=np.abs(np.random.normal(0,1,toa.shape[1]))*kdsnu
        Sd=ds_A_coeff*((T/Tref)**3)*np.exp(-ds_B_coeff*(1/T-1/Tref))
        DS=Sd*(1+DSNU)
        for i in range(DS.shape[0]):
            toa[:,i]=toa[:,i]+DS[i]

        return toa

    def toadiff(self,toa_out,toa_in,band):
        toa_diff=np.zeros([toa_out.shape[0],toa_out.shape[1]])
        count=0
        for i in range(0,len(toa_out)):
            for j in range(0,len(toa_out[0])):
                toa_diff[i,j]=np.abs(toa_out[i,j]-toa_in[i,j])
                a=np.abs(toa_out[i,j]*0.0001)

                if toa_diff[i,j]>a:
                    count=count+1

        n_elem=toa_out.shape[0]*toa_out.shape[1]
        if (count/n_elem)>0.00003:
            sys.exit('Difference check failed for '+band+' after detection stage')
