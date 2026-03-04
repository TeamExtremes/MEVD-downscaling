import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(OBS,TARGET):
    
    OBS = np.array(OBS)
    TARGET = np.array(TARGET)
    
    mask = ~np.isnan(OBS) & ~np.isnan(TARGET)
    obs_clean = OBS[mask].reshape(-1, 1) 
    down_clean = TARGET[mask]

    reg = LinearRegression()
    reg.fit(obs_clean, down_clean)

    # Obtener el slope (pendiente)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    x_line = np.linspace(np.min(obs_clean), np.max(obs_clean), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)

    return x_line, y_line, slope