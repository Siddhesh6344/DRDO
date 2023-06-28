from pylsl import StreamInlet, resolve_stream
from icecream import ic
import numpy as np
def main():

    fNIRS_data = np.array([])
    log_data = np.array([])
    # Resolving LSL Stream on Lab Network
    ic("looking for a NIRS stream...")
    nirs_streams = resolve_stream('type','NIRS')

    # Creating an inlet for the stream
    nirs_inlet = StreamInlet(nirs_streams[0])

    ic((nirs_inlet.info().name(),
        nirs_inlet.info().channel_count(),
        nirs_inlet.info().channel_format(),
        nirs_inlet.info().type(),
        nirs_inlet.info().nominal_srate(),
        nirs_inlet.info().source_id(),
        nirs_inlet.info().session_id()))

    while True:
         sample, timestamp = nirs_inlet.pull_sample()
         ic(timestamp,np.round(sample,2))

if __name__ =='__main__':
        main()