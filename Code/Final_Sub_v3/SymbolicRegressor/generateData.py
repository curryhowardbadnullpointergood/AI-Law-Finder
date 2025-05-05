import numpy as np







def generate_force_data(num_samples=100, m_range=(1, 10), a_range=(0.1, 5)):
    m = np.random.uniform(m_range[0], m_range[1], size=(num_samples, 1))
    a = np.random.uniform(a_range[0], a_range[1], size=(num_samples, 1))
    x = np.hstack([m, a])
    y = m * a
    
    return x, y






