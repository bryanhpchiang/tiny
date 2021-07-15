"""
Given a list of input vector arrays, and an AE model, do the following:

(1) Dump the list of input vector arrays as a float array declared in C.
(2) Dump the list of predicted model outputs as a float array in the same 

"""
import numpy as np


def _to_array_str(array):
    array_str = ", ".join([f"{x:.3f}F" for x in array])
    return f"{{ {array_str} }}"


if __name__ == "__main__":
    # d = np.array([1.123124124, 1, 4.0, 0, 6, 3])
    d = np.arange(100)
    print(to_float_array_str(d))
