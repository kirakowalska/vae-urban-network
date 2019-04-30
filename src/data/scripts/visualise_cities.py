from src.visualisation.visualise import plot_cities
from settings import DATAPATH
import pickle

xranges, yranges = plot_cities(DATAPATH,window_size=20000)

pickle.dump(xranges, open('xranges.pickle', 'wb'))
pickle.dump(yranges, open('yranges.pickle', 'wb'))
