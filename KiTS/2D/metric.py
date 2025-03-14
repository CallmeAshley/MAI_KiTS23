import numpy as np



"""From KiTS 공식 Github"""


def dice(prediction: np.ndarray, reference: np.ndarray):        ##   배경과 배경이 아닌 곳에 대한 dice score
    """
    Both predicion and reference have to be bool (!) numpy arrays. True is interpreted as foreground, False is background
    """
    intersection = np.count_nonzero(prediction & reference)
    numel_pred = np.count_nonzero(prediction)
    numel_ref = np.count_nonzero(reference)
    if numel_ref == 0 and numel_pred == 0:
        return np.nan
    else:
        return 2 * intersection / (numel_ref + numel_pred)
