import numpy as np
from typing import Tuple, List

def insertion(T: np.ndarray, x: int, r: int = 0) -> Tuple[np.ndarray, int, int]:
    """
    Perform row-insertion of integer x into a tableau T (2D numpy array).
    
    Rows are zero-padded to fixed width.
    
    Parameters
    ----------
    T : np.ndarray
        2D NumPy array representing the tableau.
    x : int
        The integer to insert.
    r : int, optional
        Current row index (0-based), default is 0.
    
    Returns
    -------
    Tuple[np.ndarray, int, int]
        (T, r, c) where T is the updated tableau, r and c are row and column 
        indices where insertion occurred.
    """
    
    # Get the r-th row
    row = T[r]
    
    # Find indices of non-zero elements
    nonzeros = np.nonzero(row)[0]

    if len(nonzeros) == 0:
        # Empty row: place x in the first slot
        T[r, 0] = x
        return (T, r, 0)

    # Scan row from left to right for bumping
    for j in nonzeros:
        if x < row[j]:
            # Bump the current element
            x_prime = row[j]
            T[r, j] = x

            # Recursive insertion into the next row
            return insertion(T, x_prime, r+1)

    # If no bump occurs, append at the first zero
    zero_positions = np.where(row == 0)[0]
    if len(zero_positions) == 0:
        raise ValueError("Row too short to append")
    
    T[r, zero_positions[0]] = x
    return (T, r, zero_positions[0])


def extraction(T: np.ndarray, r: int) -> Tuple[np.ndarray, int]:
    """
    Extract the rightmost element from row r and perform reverse bumping
    up the tableau.
    
    Parameters
    ----------
    T : np.ndarray
        2D NumPy array representing the tableau.
    r : int
        Row index from which to start extraction.
    
    Returns
    -------
    Tuple[np.ndarray, int]
        (T, y) where T is the updated tableau and y is the extracted element.
    """
    
    # Find length of non-zero entries in row r
    l = np.where(T[r] > 0)[0][-1] + 1
    y = T[r, l-1]        # rightmost element
    T[r, l-1] = 0        # remove it

    # Bubble the vacancy up through previous rows
    while r > 0:
        l = np.where(T[r-1] > 0)[0][-1] + 1
        # Scan row right-to-left for a bump
        for j in range(l-1, -1, -1):
            if T[r-1, j] < y:
                T[r-1, j], y = y, T[r-1, j]
                break
        r -= 1

    return T, y


def rsk_word_to_tableau(INP: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a word (sequence of integers) to its RSK insertion tableau and
    recording tableau.
    
    Parameters
    ----------
    INP : List[int]
        Word represented as integers in an alphabet [n].
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (P, Q) where P is the insertion tableau and Q is the recording tableau,
        both cropped to their minimal bounding box (non-zero entries only).
    """
    
    # Initialize empty tableaux
    t = np.zeros((len(INP), len(INP)))
    q = t.copy()

    # Insert each element of the word
    for i in range(len(INP)):
        t, r, c = insertion(t, INP[i])
        q[r, c] = i + 1  # record insertion order

    # Find non-zero rows and columns
    rows = np.any(t != 0, axis=1)
    cols = np.any(t != 0, axis=0)

    # Compute bounding box
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Crop tableaux to bounding box
    P = t[row_min:row_max+1, col_min:col_max+1]
    Q = q[row_min:row_max+1, col_min:col_max+1]
    
    return P, Q


def rsk_rebuild_word(P: np.ndarray, Q: np.ndarray) -> List[int]:
    """
    Reconstruct the original word from its RSK insertion tableau P and
    recording tableau Q.
    
    Parameters
    ----------
    P : np.ndarray
        Insertion tableau.
    Q : np.ndarray
        Recording tableau.
    
    Returns
    -------
    List[int]
        The original word as a list of integers.
    """
    
    # Determine extraction order (descending by Q values)
    order = [int(np.where(Q == i)[0][0]) for i in range(int(Q.max()), 0, -1)]

    w_0: List[int] = []
    P_copy = P.copy()
    
    # Extract elements in reverse order
    for i in order:
        P_copy, y = extraction(P_copy, i)
        w_0 = [int(y)] + w_0  # prepend to reconstruct word
    
    return w_0


def Arabic_reading_word(arr: np.ndarray) -> List[int]:
    """
    Convert a 2D tableau into a 1D "Arabic reading word":
    - read rows from bottom to top
    - flatten row-wise
    - remove all zeros
    
    Parameters
    ----------
    arr : np.ndarray
        2D NumPy array representing the tableau.
    
    Returns
    -------
    List[int]
        Flattened 1D list following the Arabic reading order.
    """
    
    flattened = arr[::-1].flatten()       # reverse rows and flatten row-wise
    result = flattened[flattened != 0]    # remove all zeros
    return result.tolist()
