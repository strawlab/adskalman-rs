#!/usr/bin/env python

import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

fname = sys.argv[1]

if 1:
    df = pd.read_csv(fname)

    _, ax = plt.subplots(nrows=1,ncols=1)
    plt.plot( df['true_x'], df['true_y'], 'k-x', label='truth')
    # plt.plot( df['obs_x'], df['obs_y'], 'bx', label='observation')
    plt.plot( df['est_x'], df['est_y'], 'g-x', label='estimate')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    _, axs = plt.subplots(nrows=6,ncols=1, sharex=True)
    ax = axs[0]
    ax.plot( df['t'], df['true_x'], 'k-x', label='truth')
    ax.plot( df['t'], df['est_x'], 'g-x', label='estimate')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    ax = axs[1]
    ax.plot( df['t'], df['true_y'], 'k-x', label='truth')
    ax.plot( df['t'], df['est_y'], 'g-x', label='estimate')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('y')

    ax = axs[2]
    ax.plot( df['t'], df['true_xvel'], 'k-x', label='truth')
    ax.plot( df['t'], df['est_xvel'], 'g-x', label='estimate')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('xvel')

    ax = axs[3]
    ax.plot( df['t'], df['true_yvel'], 'k-x', label='truth')
    ax.plot( df['t'], df['est_yvel'], 'g-x', label='estimate')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('yvel')

    ax = axs[4]
    ax.plot( df['t'], df['obs_x'], 'bx', label='observation')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('obs x')

    ax = axs[5]
    ax.plot( df['t'], df['obs_y'], 'bx', label='observation')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('obs y')

    plt.show()
