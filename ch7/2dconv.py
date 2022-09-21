import numpy as np
import sys

FILTER_WIDTH = 3
filter = np.ones(shape=(FILTER_WIDTH, FILTER_WIDTH), dtype=np.float32)

def conv(X, filters, stride=1, pad=FILTER_WIDTH//2):
    h, w = X.shape
    filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, pad, 'constant')
    out = np.zeros((out_h, out_w), dtype=np.float32)

    for h in range(out_h): # slide the filter vertically.
        h_start = h * stride
        h_end = h_start + filter_h
        for w in range(out_w): # slide the filter horizontally.
            w_start = w * stride
            w_end = w_start + filter_w
            # Element-wise multiplication.
            out[h, w] = np.sum(in_X[h_start:h_end, w_start:w_end] * filter)

    return out

if __name__ == "__main__":
    width = int(sys.argv[1])

    inArr = np.arange(width * width).reshape((width,-1)).astype(np.float32)
    outArr = conv(inArr, filter)

    print("[inArr]\n", inArr)
    print("[outArr]\n", outArr)