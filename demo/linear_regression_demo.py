import sys
import numpy as np
sys.path.insert(0, '../src')
from LinearRegression import LinearRegression

test = LinearRegression(0.001)
test.run(1)

