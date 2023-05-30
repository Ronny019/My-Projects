# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#      'This is the first document.',
#      'This document is the second document.',
#     'And this is the third one.',
#      'Is this the first document?',
#  ]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# vectorizer.get_feature_names_out()
# print(X)

import numpy as np
from scipy.sparse import csr_matrix

indptr = np.array([0, 2, 3, 6])

indices = np.array([0, 2, 2, 0, 1, 2])

data = np.array([1, 2, 3, 4, 5, 6])

csr = csr_matrix((data, indices, indptr), shape=(3, 3))

# filter values greater than 2 and print indices
row, col = csr.nonzero()
values = csr[row, col]
filtered_indices = np.where(values > 2)
filtered_row_indices = row[filtered_indices]
filtered_col_indices = col[filtered_indices]
filtered_values = values[filtered_indices]
print("Filtered values:")
for i in range(len(filtered_values)):
    print(f"value: {filtered_values[i]}, row: {filtered_row_indices[i]}, column: {filtered_col_indices[i]}")
