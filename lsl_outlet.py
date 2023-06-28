import sys
import time
from random import random as rand
from pylsl import StreamInfo, StreamOutlet, local_clock

def main(argv):
    srate = 280
    name = 'fNIRS'
    type = 'NIRS'
    n_channels = 256


    # Creating Stream Info
    info = StreamInfo(name,type,n_channels,srate,'float32','myuid34234')

    outlet = StreamOutlet(info)

    #  Stream start ..
    print("now sending data..")
    start_time= local_clock()
    sent_samples=0


    while True:
        elapsed_time = local_clock()-start_time
        required_samples= int(srate*elapsed_time)-sent_samples
        for sample_ix in range(required_samples):
            mysample = [rand() for _ in range(n_channels)]
            outlet.push_sample(mysample)

        sent_samples+=required_samples
        time.sleep(0.01)

if __name__ =='__main__':
    main(sys.argv[1:])

        
