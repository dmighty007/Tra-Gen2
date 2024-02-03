# The libraries...
import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.dihedrals import Ramachandran
import mdtraj
import pandas as pd
from scipy.signal import argrelextrema


class Analysis:
    def __init__(self, traj_file, top = None, selection = "resid 0-19"):
        mdtraj_u = mdtraj.load(traj_file, top = top)
        self.mda_u = mda.Universe(top, traj_file)
        self.ids = self.mda_u.select_atoms(selection).indices
        self.sliced_mdtraj = mdtraj_u.atom_slice(self.ids)
        self.atomgroup = self.mda_u.select_atoms(selection, updating = True)
        self.mda_u.trajectory[0]
        self.ref = self.mda_u.select_atoms(selection)
        ######### including the time information..
        time = []
        for frame in self.mda_u.trajectory:
            time.append(frame.time)
        self.time = time

    def RMSD(self, selection = "name CA"):
        """
        Returns the RMSD of the protein form refference structure
        """
        protein = self.atomgroup
        ref = self.ref
        R = RMSD(protein, ref, select = selection)
        R.run()
        return R.results.rmsd[:,2]

    def ROG(self):
        """
        Returns radius of gyration of the protein 
        """
        polymer = self.atomgroup
        #u.trajectory[0]
        #ref = u.select_atoms("resid 1-20")
        rog_list = []
        for frame in self.mda_u.trajectory:
            rog_list.append(polymer.radius_of_gyration())
        return np.array(rog_list, dtype = np.float32)

    def SASA(self):
        """
        Returns a 1D array of sasa value of the protein
        """
        sasa = mdtraj.shrake_rupley(self.sliced_mdtraj)
        return sasa.sum(axis = 1)
    
    def ramachandran(self):
        """
        Returns a 2D array, 1st column phi, 2nd column psi
        """
        r = Ramachandran(self.atomgroup).run()
        return r.results.angles
    
class Decision:

    def __init__(self):
        pass
    def slope_n_intercept(self, x, y):
        m,c = np.polyfit(x, y , deg = 1)
        return m, c 
    def decideROG(self, time, rog):
        m, _ = self.slope_n_intercept(time, rog)

        if m < 0:
            return "Rg is decreasing.. means the protein has folding kind of nature throughout the trajectory."
        elif m > 0:
            return "Rg is increasing.. means the protein has unfolding kind of nature throughout the trajectory."

        
    def decideSASA(self, time, sasa):
        m, _ = self.slope_n_intercept(time, sasa)

        if m < 0:
            return "SASA is decreasing.. means the protein is being buried from an exposed state."
        elif m > 0:
            return "SASA is increasing.. means the protein is being exposed from an unburied state."



    def local_minima(self, rmsd):
        s = pd.Series(rmsd)
        s = s.dropna()
        local_max_indices = argrelextrema(-s.values, np.greater, order = 10)
        return local_max_indices
    def local_maxima(self, rmsd):
        s = pd.Series(rmsd)
        s = s.dropna()
        local_max_indices = argrelextrema(s.values, np.greater, order = 10)
        return local_max_indices

    def number_of_transitions_from_rmsd(self, rmsd):
        minima = self.local_minima(rmsd)
        maxima = self.local_maxima(rmsd)
        num_transitions = 0
        if len(maxima[0]) > len(minima[0]):
            for mini in minima[0]:
                for i in range(len(maxima[0]) - 1):
                    if mini > maxima[0][i] and mini < maxima[0][i+1]:
                        num_transitions += 1
        else:
            for maxi in maxima[0]:
                for i in range(len(minima[0]) - 1):
                    if maxi > minima[0][i] and maxi < minima[0][i+1]:
                        num_transitions += 1
        return f"Throughout the simulation according to RMSD, {2*num_transitions + 1} conformational change have been observed."
    
class Plots:
    def __init__(self, out = "./"):
        self.setPlotParams()
        self.out = out 
    def setPlotParams(self):
        """
        Set plot parameters for matplotlib and seaborn.

        Returns:
        - None
        """
        # Set linewidth for axes in plots
        plt.rc("axes", linewidth=3)

        # Set font size and other parameters for legend
        plt.rc("legend", fontsize=24, frameon=False)

        # Disable the usage of TeX for text rendering
        plt.rcParams['text.usetex'] = False

        # Set font size and family for text in plots
        plt.rcParams['font.size'] = 24
        plt.rcParams['font.family'] = "serif"

        # Set tick direction and size for both major and minor ticks
        tick_direction = 'in'
        major_tick_size = 10.0
        minor_tick_size = 6.0

        plt.rcParams['xtick.direction'] = tick_direction
        plt.rcParams['ytick.direction'] = tick_direction
        plt.rcParams['xtick.major.size'] = major_tick_size
        plt.rcParams['xtick.minor.size'] = minor_tick_size
        plt.rcParams['ytick.major.size'] = major_tick_size
        plt.rcParams['ytick.minor.size'] = minor_tick_size
    def plotRMSD(self, time, rmsd, unit = "ps"):
        fig, ax = plt.subplots(figsize = [10, 6])
        ax.plot(time, rmsd, lw = 2)
        ax.set_xlabel(f"Time ({unit})", fontsize = 32)
        ax.set_ylabel("RMSD", fontsize = 32)
        fig.tight_layout()
        plt.savefig( self.out+"RMSD.png", dpi = 600)
        plt.close()
    def plotROG(self, time, rog, unit = "ps"):
        fig, ax = plt.subplots(figsize = [10, 6])
        ax.plot(time, rog, lw = 2)
        ax.set_xlabel(f"Time ({unit})", fontsize = 32)
        ax.set_ylabel("$R_g (\AA)$", fontsize = 32)
        fig.tight_layout()
        plt.savefig(self.out + "ROG.png", dpi = 600)
        plt.close()
    def plotSASA(self, time, sasa, unit = "ps"):
        fig, ax = plt.subplots(figsize = [10, 6])
        ax.plot(time, sasa, lw = 2)
        ax.set_xlabel(f"Time ({unit})", fontsize = 32)
        ax.set_ylabel("SASA", fontsize = 32)
        fig.tight_layout()
        plt.savefig(self.out + "sasa.png", dpi = 600)
        plt.close()

class printColor:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
