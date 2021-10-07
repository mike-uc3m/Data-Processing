from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os
from common.io.writeToa import writeToa, readToa
from scipy.io import netcdf
from netCDF4 import Dataset

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        c_hdiff=0
        for i in range (0,len(Hdiff)):
            for j in range (0,len(Hdiff)):
                if fr2D[i,j]**2>1 and H[i,j]!=0:
                    c_hdiff=c_hdiff+1

        if c_hdiff!=0:
            print('H_diff check failed')

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff*Hwfe*Hdefoc*Hdet*Hsmear*Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)
        # fig1.show()
        # fig1.savefig('/home/luss/my_shared_folder/test_ism/MTF_ALT.png')
        # fig2.savefig('/home/luss/my_shared_folder/test_ism/MTF_ACT.png')


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        fstepAlt = 1/nlines/w
        fstepAct = 1/ncolumns/w

        eps=1e-6
        fAlt = np.arange(-1/(2*w),1/(2*w)-eps,fstepAlt)
        fAct = np.arange(-1/(2*w),1/(2*w)-eps,fstepAct)

        ep_cutoff=D/(lambd*focal)

        fnAlt=fAlt/(1/w)
        fnAct=fAct/(1/w)
        frAlt=fAlt/ep_cutoff
        frAct=fAct/ep_cutoff

        [fnAltxx,fnActxx] = np.meshgrid(fnAlt,fnAct,indexing='ij') # Please use ‘ij’ indexing or you will get the transpose
        fn2D=np.sqrt(fnAltxx*fnAltxx + fnActxx*fnActxx)
        [frAltxx,frActxx] = np.meshgrid(frAlt,frAct,indexing='ij')
        fr2D=np.sqrt(frAltxx*frAltxx + frActxx*frActxx)*(1/w)/ep_cutoff

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff=np.empty([len(fr2D),len(fr2D[0])])
        for i in range(0,len(fr2D)):
            for j in range(0,len(fr2D[0])):
                Hdiff[i,j]=(2/pi)*(m.acos(fr2D[i,j])-fr2D[i,j]*(1-fr2D[i,j]**2)**(1/2))

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """

        x=pi*defocus*fr2D*(1-fr2D)
        J1=j1(x)

        Hdefoc=2*J1/x

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """

        Hwfe=np.empty([len(fr2D),len(fr2D[0])])
        for i in range(0,len(fr2D)):
            for j in range(0,len(fr2D[0])):
                Hwfe[i,j]=m.exp(-fr2D[i,j]*(1-fr2D[i,j])*(kLF*(wLF/lambd)**2+kHF*(wHF/lambd)**2))

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """

        Hdet=np.sinc(fn2D)

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """

        hsmear=np.sinc(ksmear*fnAlt)
        Hsmear=np.transpose(repmat(hsmear,ncolumns,1))

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """

        Hmotion=np.sinc(kmotion*fn2D)

        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """

        #Along track plot
        plt.figure(figsize=(8, 6))
        a=int(ncolumns/2)

        plt.plot(fnAlt,Hdiff[:,a],label='Hdiff')
        plt.plot(fnAlt,Hdefoc[:,a],label='Hdefoc')
        plt.plot(fnAlt,Hwfe[:,a],label='Hwfe')
        plt.plot(fnAlt,Hdet[:,a],label='Hdet')
        plt.plot(fnAlt,Hsmear[:,a],label='Hsmear')
        plt.plot(fnAlt,Hmotion[:,a],label='Hmotion')
        plt.plot(fnAlt,Hsys[:,a],label='Hsys')

        plt.legend()
        plt.savefig(directory+'MTF_ALT_'+band+'.png')
        plt.close()

        #Across track plot
        plt.figure(figsize=(8, 6))
        b=int(nlines/2)
        ncfile = os.path.join('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/', 'Hmotion_' + band + '.nc')
        dset = Dataset(ncfile)
        h_lucia= np.array(dset.variables['isrf'][:])

        dset.close()
        #h_lucia=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/','Hdiff_' + band + '.nc')


        plt.plot(fnAct[a:],Hdiff[b,a:],label='Hdiff')
        plt.plot(fnAct[a:],Hdefoc[b,a:],label='Hdefoc')
        plt.plot(fnAct[a:],Hwfe[b,a:],label='Hwfe')
        plt.plot(fnAct[a:],Hdet[b,a:],label='Hdet')
        plt.plot(fnAct[a:],Hsmear[b,a:],label='Hsmear')
        plt.plot(fnAct[a:],Hmotion[b,a:],label='Hmotion')
        plt.plot(fnAct[a:],Hsys[b,a:],label='Hsys')
        plt.plot(fnAct[a:],h_lucia[b,a:],label='hlucia')
        plt.grid()

        plt.ylim([0,1.1])
        plt.xlim([0,0.5])

        plt.legend()
        plt.savefig(directory+'MTF_ACT_'+band+'.png')
        plt.close()



