import numpy as np
import pandas as pd

class Interpreter():
    

    def mat_to_csv(name, lista):
        df = pd.DataFrame(data=mat.astype(float))
        df.to_csv(name, sep=' ', header=False, float_format='%.2f', index=False)