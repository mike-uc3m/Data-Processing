
from ism.src.initIsm import initIsm
from math import pi
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifft2, fft2
import numpy as np
from common.io.writeToa import writeToa, readToa
from common.io.readIsrf import readIsrf
from scipy.interpolate import interp1d, interp2d
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
from scipy.signal import convolve2d
from common.src.auxFunc import getIndexBand
import os
import matplotlib.pyplot as plt
import sys

class opticalPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, sgm_toa, sgm_wv, band):
        """
        The optical phase is in charge of simulating the radiance
        to irradiance conversion, the spatial filter (PSF)
        and the spectral filter (ISRF).
        :return: TOA image in irradiances [mW/m2/nm],
                    with spatial and spectral filter
        """
        self.logger.info("EODP-ALG-ISM-1000: Optical stage")

        # Calculation and application of the ISRF
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1010: Spectral modelling. ISRF")
        toa = self.spectralIntegration(sgm_toa, sgm_wv, band)
        toa_isrf=toa

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_isrf:
            saveas_str = self.globalConfig.ism_toa_isrf + band
            writeToa(self.outdir, saveas_str, toa)

        # Radiance to Irradiance conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1020: Radiances to Irradiances")
        toa = self.rad2Irrad(toa,
                             self.ismConfig.D,
                             self.ismConfig.f,
                             self.ismConfig.Tr)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        # Spatial filter
        # -------------------------------------------------------------------------------
        # Calculation and application of the system MTF
        self.logger.info("EODP-ALG-ISM-1030: Spatial modelling. PSF/MTF")
        myMtf = mtf(self.logger, self.outdir)
        Hsys = myMtf.system_mtf(toa.shape[0], toa.shape[1],
                                self.ismConfig.D, self.ismConfig.wv[getIndexBand(band)], self.ismConfig.f, self.ismConfig.pix_size,
                                self.ismConfig.kLF, self.ismConfig.wLF, self.ismConfig.kHF, self.ismConfig.wHF,
                                self.ismConfig.defocus, self.ismConfig.ksmear, self.ismConfig.kmotion,
                                self.outdir, band)

        # Apply system MTF
        toa = self.applySysMtf(toa, Hsys) # always calculated
        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        #Cheking that difference is under the margin
        #---------------------------------------------------------------------------------
        toa_lucia_isrf=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/','ism_toa_isrf_' + band + '.nc')
        toa_lucia_opt=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/','ism_toa_optical_' + band + '.nc')
        self.plotToa(toa_lucia_isrf, label="toa_isrf_reference")
        self.plotToa(toa_isrf, label="toa_output_isrf")
        plt.legend()
        plt.ylabel('[W/m2]')
        plt.xlabel('ACT pixels')
        plt.title('TOA after the ISRF ('+band+')')
        plt.savefig(self.outdir + "toa-comparison-isrf-"+band+'.png')
        plt.close()

        self.plotToa(toa_lucia_opt, label="toa_opt_reference")
        self.plotToa(toa, label="toa_output")
        plt.legend()
        plt.ylabel('[W/m2]')
        plt.xlabel('ACT pixels')
        plt.title('TOA after the optical stage('+band+')')
        plt.savefig('/home/luss/my_shared_folder/test_ism/' + "toa-comparison-opt-"+band+'.png')
        plt.close()

        self.toadiff(toa_isrf,toa_lucia_isrf,1,band)
        self.toadiff(toa,toa_lucia_opt,2,band)


        # Write output TOA & plots
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_optical_stage:
            saveas_str = self.globalConfig.ism_toa_optical + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the optical phase [mW/sr/m2]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa

    def rad2Irrad(self, toa, D, f, Tr):
        """
        Radiance to Irradiance conversion
        :param toa: Input TOA image in radiances [mW/sr/m2]
        :param D: Pupil diameter [m]
        :param f: Focal length [m]
        :param Tr: Optical transmittance [-]
        :return: TOA image in irradiances [mW/m2]
        """
        rad_factor=Tr*(pi/4)*(D/f)**2
        toa=toa*rad_factor

        with open(self.outdir+'Instrument module.txt', 'a') as f:
            f.write('Radiance to irradiance conversion: ' + str(rad_factor)+'\n')
        return toa


    def applySysMtf(self, toa, Hsys):
        """
        Application of the system MTF to the TOA
        :param toa: Input TOA image in irradiances [mW/m2]
        :param Hsys: System MTF
        :return: TOA image in irradiances [mW/m2]
        """
        F=fft2(toa)
        Hsys=fftshift(Hsys)
        G=F*Hsys
        toa_ft=ifft2(G)
        toa_ft=np.real(toa_ft)

        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integration with the ISRF to retrieve one band
        :param sgm_toa: Spectrally oversampled TOA cube 3D in irradiances [mW/m2]
        :param sgm_wv: wavelengths of the input TOA cube
        :param band: band
        :return: TOA image 2D in radiances [mW/m2]
        """
        isrf, wv_isrf = readIsrf(os.path.join(self.auxdir,self.ismConfig.isrffile), band)
        wv_isrf=wv_isrf*1000
        isrf_n=isrf/np.sum(isrf)

        sgm_toa=np.array(sgm_toa)
        L=np.zeros([sgm_toa.shape[0],sgm_toa.shape[1]])
        for i in range(sgm_toa.shape[0]):
            for j in range(sgm_toa.shape[1]):
                cs=interp1d(sgm_wv,sgm_toa[i,j,:],fill_value=(0,0),bounds_error=False)
                toa_i=cs(wv_isrf)
                L[i,j]=np.sum(isrf_n*toa_i)
        return L

    def plotToa(self, toa,label,index=0):

        act_pixels=range(0,len(toa[0]))
        fig=plt.plot(act_pixels,toa[index,:],label=label)
        return fig

    def toadiff(self,toa_out,toa_in,index,band):
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
            if index==1:
                sys.exit('Difference check failed for '+band+' after ISRF in optical stage')
            elif index==2:
                sys.exit('Optical difference check failed for '+band+' after optical stage')
