
# LEVEL-1B MODULE

from l1b.src.initL1b import initL1b
from common.io.writeToa import writeToa, readToa
from common.src.auxFunc import getIndexBand
from common.io.readFactor import readFactor, EQ_MULT, EQ_ADD, NC_EXT
import numpy as np
import os
import matplotlib.pyplot as plt

class l1b(initL1b):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1B Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the ISM in Digital Numbers
            # -------------------------------------------------------------------------------
            toa = readToa(self.indir, self.globalConfig.ism_toa + band + '.nc')

            # Equalization (radiometric correction)
            # -------------------------------------------------------------------------------
            if self.l1bConfig.do_equalization:
                self.logger.info("EODP-ALG-L1B-1010: Radiometric Correction (equalization)")

                # Read the multiplicative and additive factors from auxiliary/equalization/
                eq_mult = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_mult+band+NC_EXT),EQ_MULT)
                eq_add = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_add+band+NC_EXT),EQ_ADD)

                # Do the equalization and save to file
                toa_eq = self.equalization(toa, eq_add, eq_mult)
                writeToa(self.outdir, self.globalConfig.l1b_toa_eq + band, toa)


            # Restitution (absolute radiometric gain)
            # -------------------------------------------------------------------------------
            self.logger.info("EODP-ALG-L1B-1020: Absolute radiometric gain application (restoration)")
            toa_l1b = self.restoration(toa_eq, self.l1bConfig.gain[getIndexBand(band)])

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeToa(self.outdir, self.globalConfig.l1b_toa + band, toa_l1b)
            self.plotL1bToa(toa_l1b, self.outdir, band,label="toa_l1b")
            plt.legend()
            plt.savefig(self.outdir + "toa-l1b-"+band+'.png')

            #Cheking that difference is under the margin
            #---------------------------------------------------------------------------------
            toa_lucia=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-L1B/output/',self.globalConfig.l1b_toa + band + '.nc')
            self.plotL1bToa(toa_lucia, self.outdir, band,label="toa_lucia")
            plt.legend()
            plt.savefig(self.outdir + "toa-comparison-"+band+'.png')
            plt.close()

            self.toadiff(toa_l1b,toa_lucia)

            #Plotting against the isrf signal
            #--------------------------------------------------------------------------------
            self.plotL1bToa(toa_l1b, self.outdir, band,label="toa_l1b",index=50)
            toa_isrf=readToa('/home/luss/my_shared_folder/EODP_TER_2021/EODP-TS-ISM/output/', 'ism_toa_isrf_' + band + '.nc')
            self.plotL1bToa(toa_isrf, self.outdir, band,label="toa_isrf",index=50)
            plt.legend()
            plt.savefig(self.outdir + "toa-comparison-isrf-"+band+'.png')
            plt.close()

            #Equalization check
            #--------------------------------------------------------------------------------
            toa_eq_false= self.restoration(toa, self.l1bConfig.gain[getIndexBand(band)])
            self.plotL1bToa(toa_l1b, self.outdir, band,label="toa_l1b")
            self.plotL1bToa(toa_eq_false, self.outdir, band,label="toa_eq_false")
            plt.legend()
            plt.savefig(self.outdir + "toa-eq-false-"+band+'.png')
            plt.close()


            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1B Module!")



    def equalization(self, toa, eq_add, eq_mult):
        """
        Equlization. Apply an offset and a gain.
        :param toa: TOA in DN
        :param eq_add: Offset in DN
        :param eq_mult: Gain factor, adimensional
        :return: TOA in DN, equalized
        """

        toa_out=(toa-eq_add)/eq_mult

        return toa_out

    def restoration(self,toa,gain):
        """
        Absolute Radiometric Gain - restore back to radiances
        :param toa: TOA in DN
        :param gain: gain in [rad/DN]
        :return: TOA in radiances [mW/sr/m2]
        """

        toa_out=toa*gain

        self.logger.debug('Sanity check. TOA in radiances after gain application ' + str(toa[1,-1]) + ' [mW/m2/sr]')

        return toa_out

    def plotL1bToa(self, toa, outputdir, band,label,index=0):

        act_pixels=range(0,len(toa[0]))
        plt.plot(act_pixels,toa[index,:],label=label)


    def toadiff(self,toa_out,toa_in):
        toa_diff=np.zeros([toa_out.shape[0],toa_out.shape[1]])
        count=0
        for i in range(0,len(toa_out)):
            for j in range(0,len(toa_out[0])):
                toa_diff[i,j]=toa_out[i,j]-toa_in[i,j]
                a=toa_out[i,j]*0.01

                if toa_diff[i,j]>a:
                    count=count+1

        if count>0:
            print('Difference check failed for '+band)
        #else:
            #print('Difference check successful for '+band)
