import sys
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
sys.path.insert(0, '../src')
from LinearRegression import LinearRegression
data = pd.read_csv('../data/beer_reviews.csv',
                   delimiter=',').head(2000)

test = LinearRegression(data[["review_overall",
                              "review_appearance",
                              "review_aroma"]],
                        "review_overall"
                        , 0.01)
print(test.total_error())
test.run(iterations=100)
print(test.total_error())
print(test.predict(np.array([5, 3.5])))


