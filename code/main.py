import numpy as np
import pandas as pd
from pysr import PySR
from sklearn.model_selection import train_test_split

# Generate data
n_samples = 100
mass = np.random.uniform(1, 10, n_samples)  # Random masses between 1 and 10
acceleration = np.random.uniform(-5, 5, n_samples)  # Random accelerations between -5 and 5
force = mass * acceleration  # Calculate force using F = ma

# Create a Pandas DataFrame
data = pd.DataFrame({'mass': mass, 'acceleration': acceleration, 'force': force})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[['mass', 'acceleration']], data['force'], test_size=0.2, random_state=42
)

# Create and train the symbolic regression model
model = PySR(
    model_selection='best',  # Use the best model based on performance
    n_iterations=100,  # Number of iterations for the search
    population_size=100,  # Size of the population in the search
    max_tree_depth=5,  # Maximum depth of the expression trees
    verbosity=1,  # Verbosity level for output
)

model.fit(X_train, y_train)

# Evaluate the model on the test set
predictions = model.predict(X_test)

# Print the best equation found
print(model.get_best_equation())

# Evaluate the model's performance
print(f"MSE: {mean_squared_error(y_test, predictions)}")