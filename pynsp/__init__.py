from .methods import tools, signal, stats, qc, rsfc
from .core.ui import QC, RSFC, EvokedFMRI
from .core.io import load, save
import os

__version__ = '0.2.2'
package_directory = os.path.dirname(os.path.abspath(__file__))

__all__ = ['QC', 'RSFC', 'EvokedFMRI',
           'signal', 'stats', 'qc', 'rsfc', 'tools',
           'load', 'save']
