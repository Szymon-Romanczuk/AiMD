from szymon.szymon import SR
from wojtek.train import get_scores
from jakub import JG_wywołaj
import pandas as pd

sz = SR()
wo = get_scores()
dane_jakub = pd.read_excel(r'jakub.xlsx')
ja = JG_wywołaj(dane_jakub)



