"""
        1) load noise, AE and UP
        2) compute spectrogram
        3) flatten and add to matrix
        4) add label
        5) save data set as .mat

"""
import matplotlib as mpl
#mpl.use( 'Agg')
import os, glob
import numpy as np
import matplotlib.pyplot as plt
#import obspy.imaging.spectrogram
import scipy.signal as signal
import scipy.io

from obspy import read
#--------------------------my modules---------------------------------------------------

#===============================================================================
#                         parameters and dir
#===============================================================================
lPar = {       'l_ev_type' : ['AE', 'UP', 'noise'], # AE
               'load'      : 'Pc_150_Axial', #'Pc_120_Axial',
               'ID'        : 'Wgrc09',
                #---------spectrogram params-------------
               'logSpec' : False,
               'nperseg' : 20, 'noverlap' : 10,
               'nfft'    : 256*.5,
               'nx'      : 101,
               'ny'      :  65, # nfft*.5
               'duration'  : 102.4, #length of record in mu sec for trimming
               # -----------plotting-------------
               'showPlot' : False,
               }

data_dir = "data"
file_out = "labquake_spec.mat"

dSpec = { 'mSpec'   : np.array([]), #len = nx*ny+1
          'file'    : [],
          'delta_t' : float(),
          'nx' : int(), 'ny' : int(),
          }
#-----------initialize matrix---------------------
N_ev = 0
for i_ev, ev_type in enumerate( lPar['l_ev_type']):
    l_files = glob.glob( f"{data_dir}/{ev_type}/*.mseed")
    N_ev += len( l_files)

print( 'Ntot', N_ev)
dSpec['mSpec'] = np.zeros( (lPar['nx']*lPar['ny']+1, N_ev), dtype = float)
for i_ev, ev_type in enumerate( lPar['l_ev_type']):
    l_files = glob.glob( f"{data_dir}/{ev_type}/*.mseed")
    #===============================================================================
    #                       read mseed, preprocess
    #===============================================================================

    for i_fi, curr_file in enumerate(l_files):
        curr_basename = os.path.basename( curr_file).split('.')[0]
        print( 'nFile: ', i_fi+1, ' of ', len( l_files), curr_basename)
        ev_id, nfi, nn = curr_basename.split('_')[1], curr_basename.split('_')[2], curr_basename.split('_')[3]
        iCh = curr_basename.split('_')[5]

        # read
        SEIS    = read( curr_file)
        t0      = SEIS[0].stats.starttime
        delta_t = SEIS[0].stats.delta
        #detrend
        SEIS[0].detrend( 'linear')
        #===============================================================================
        #                  compute   spectrogram
        #===============================================================================
        # log spectrogram
        nperseg = int(.02*SEIS[0].stats.npts)
        print( 'npserseg', nperseg)
        a_f, a_t, Sxx = signal.spectrogram( SEIS[0].data, int(1./delta_t),
                                        scaling='spectrum',
                                        nfft = lPar['nfft'],
                                        nperseg=nperseg,
                                        noverlap=int( .5*nperseg))
        print( 'spectrogram dimensions: ', Sxx.shape, len( a_t), len( a_f))
        # setup matrix dimension during first run through files and event types
        if i_fi == 0 and ev_type == lPar['l_ev_type'][0]:
            print( '------------first run-----------------')
            ny,nx = Sxx.shape
            dSpec['a_f']     = a_f
            dSpec['a_t']     = a_t
            dSpec['nx']      = nx
            dSpec['ny']      = ny
            dSpec['delta_t'] = delta_t
            #dSpec['mSpec']   = np.zeros( (nx*ny+1, len(l_files)*len(lPar['l_ev_type'])), dtype = float)

        if lPar['logSpec'] == True:
            Sxx = np.log10( Sxx)

        a_spec = Sxx.flatten()
        Sxx = a_spec.reshape( ny,nx)
        #===============================================================================
        #                  test plot
        #===============================================================================
        if lPar['showPlot'] == True:
            fig, ax = plt.subplots( 2, 1, sharex=True)
            ax[0].set_title( f"sta={SEIS[0].stats.station}, cha={SEIS[0].stats.channel}, npts={SEIS[0].stats.npts}")
            ax[0].plot( np.arange( SEIS[0].stats.npts)*delta_t,
                        SEIS[0].data, 'k-')

            norm = mpl.colors.Normalize(vmin = 1, vmax=8)

            plot1 = ax[1].pcolormesh(a_t, a_f, Sxx,
                                     shading = 'nearest',
                                     #shading='gouraud',
                                     cmap = plt.cm.RdYlGn_r,
                                     # norm = norm
                                     )
            ax[1].set_ylabel('Frequency [Hz]')
            ax[1].set_xlabel('Time [sec]')
            plt.show()

        #===============================================================================
        #                       save data stack
        #===============================================================================
        # flatten
        a_spec = Sxx.flatten()
        # add label
        if ev_type == 'noise':
            i_label = 0
        elif ev_type == 'AE':
            i_label = 1
        elif ev_type == 'UP':
            i_label = 2
        a_spec = np.hstack(( a_spec, i_label))

        # store in large matrix
        i_tot = i_fi+len(l_files)*i_ev
        print( i_tot, 'event type', i_label)
        dSpec['mSpec'][:,i_tot] = a_spec

print( 'Ntot data features: ', dSpec['mSpec'].shape[0],dSpec['mSpec'].shape[1])
dSpec['mSpec'] = dSpec['mSpec'].T
#save as matlab binary
scipy.io.savemat(f"{data_dir}/{file_out}", dSpec, do_compression=True)

