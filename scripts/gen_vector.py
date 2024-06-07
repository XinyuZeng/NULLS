import numpy as np
np.random.seed(12161825)
# Generate 1024 random uint32 numbers in the range of 0 to 4096
random_values = np.random.randint(low=0, high=4097, size=1024, dtype='uint32')

# Print the array
print(list(random_values))