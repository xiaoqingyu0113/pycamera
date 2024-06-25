def powerE(x, terms=20):
    result = 1.0  # Initialize result to 1 (the first term of the series)
    term = 1.0    # Initialize the first term of the series
    for n in range(1, terms):
        term *= x / n  # Compute the next term in the series
        result += term  # Add the term to the result
        # Break the loop if the term is small enough to not affect the result significantly
        if term < 1e-15:
            break
    return result


import numpy as np

def test_exp():
    x = 10
    # assert np.isclose(exp(x), np.exp(x))
    print(powerE(x))
    print(np.exp(x))

if __name__ == '__main__':
    test_exp()