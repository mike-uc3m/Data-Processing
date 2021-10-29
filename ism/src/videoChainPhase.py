
from ism.src.initIsm import initIsm
import numpy as np
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
import sys
from common.io.writeToa import writeToa, readToa

class videoChainPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, toa, band):
        self.logger.info("EODP-ALG-ISM-3000: Video Chain")

        # Electrons to Voltage - read-out & amplification
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-3010: Electrons to Voltage – Read-out and Amplification")
        toa = self.electr2Volt(toa,
                         self.ismConfig.OCF,
                         self.ismConfig.ADC_gain)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [V]")

        # Digitisation
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-3020: Voltage to Digital Numbers – Digitisation")
        toa = self.digitisation(toa,
                          self.ismConfig.bit_depth,
                          self.ismConfig.min_voltage,
                          self.ismConfig.max_voltage)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [DN]")

        # Plot
        if self.ismConfig.save_vcu_stage:
            saveas_str = self.globalConfig.ism_toa_vcu + band
            writeToa(self.outdir, saveas_str, toa)
            title_str = 'TOA after the VCU phase [DN]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        #Cheking that difference is under the margin
        #---------------------------------------------------------------------------------
        toa_lucia=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/',self.globalConfig.ism_toa + band + '.nc')

        self.toadiff(toa,toa_lucia,band)

        return toa

    def electr2Volt(self, toa, OCF, gain_adc):
        """
        Electron to Volts conversion.
        Simulates the read-out and the amplification
        (multiplication times the gain).
        :param toa: input toa in [e-]
        :param OCF: Output Conversion factor [V/e-]
        :param gain_adc: Gain of the Analog-to-digital conversion [-]
        :return: output toa in [V]
        """
        toa=toa*OCF*gain_adc

        with open(self.outdir+'Instrument module.txt', 'a') as f:
            f.write('Electron to volt conversion: ' + str(OCF*gain_adc)+'\n')
        return toa

    def digitisation(self, toa, bit_depth, min_voltage, max_voltage):
        """
        Digitisation - conversion from Volts to Digital counts
        :param toa: input toa in [V]
        :param bit_depth: bit depth
        :param min_voltage: minimum voltage
        :param max_voltage: maximum voltage
        :return: toa in digital counts
        """

        toa=np.round(((2**bit_depth)-1)*toa/(max_voltage-min_voltage))
        count=0
        for i in range(toa.shape[0]):
            for j in range (toa.shape[1]):
                if toa[i,j]>((2**bit_depth)-1):
                    toa[i,j]=(2**bit_depth)-1
                    count+=1
                elif toa[i,j]<0:
                    toa[i,j]=0
        sat_pix=count/(toa.shape[0]*toa.shape[1])
        conv_factor=((2**bit_depth)-1)/(max_voltage-min_voltage)
        with open(self.outdir+'Instrument module.txt', 'a') as f:
            f.write('Volt to digital number conversion: ' + str(conv_factor)+'\n'+'Saturated pixels: ' + str(sat_pix*100)+'%\n')
        return toa

    def toadiff(self,toa_out,toa_in,band):
        toa_diff=np.zeros([toa_out.shape[0],toa_out.shape[1]])
        count=0
        for i in range(0,len(toa_out)):
            for j in range(0,len(toa_out[0])):
                toa_diff[i,j]=toa_out[i,j]-toa_in[i,j]
                a=toa_out[i,j]*0.01

                if toa_diff[i,j]>a:
                    count=count+1

        n_elem=toa_out.shape[0]*toa_out.shape[1]
        if (count/n_elem)>0.003:
            sys.exit('Difference check failed for '+band+' after video stage')
