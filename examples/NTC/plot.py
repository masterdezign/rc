#!/usr/bin/env python3

"""
This script visualizes prediction result from ./result.txt.
"""

import numpy as np
from matplotlib.pyplot import *
import os

def main(fname):
    # Load data
    xs = np.loadtxt(fname)
    tgt = xs[:,0]
    pre = xs[:,1]

    # Calculate normalized root mean square error
    err = nrmse(tgt, pre)

    # Plot
    title('Mackey-Glass prediction, error = %.3f' % err)
    plot(tgt, label='Target')
    plot(pre, 'x', label='Prediction')
    legend(loc='best')
    ylabel("Predicted value")
    xlabel("Sample, $n$")

    # Zoom-in
    # Samples per delay in predicted Mackey-Glass time series
    samples_per_delay = 17
    initial = 2425
    delays = 18
    xlim([initial, initial + delays * samples_per_delay])

    show()

def nrmse(targets, predictions):
    targetVariance = np.var(targets)
    return np.sqrt(np.mean((predictions-targets)**2) / targetVariance)

if __name__ == '__main__':
    localpath = os.path.dirname(os.path.realpath(__file__))
    main(os.path.join(localpath, "result.txt"))
