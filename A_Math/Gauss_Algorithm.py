import sys
import math
import numpy as np
from colorama import init
from termcolor import colored


def end_program(reason):
    print("Execution has to be stopped.\n Reason:\n", reason)
    sys.exit()


def apply_gaussian(matrix, method):
    end_row = (np.size(matrix, 0) - 1)
    end_col = int((np.size(matrix, 1) - 1) if method != 'inverse' else (np.size(matrix, 1) / 2))
    current_row = 0
    current_col = 0
    while (current_col <= end_col) & (current_row <= end_row):
        next_element = matrix[current_row, current_col]
        if next_element == 0:
            for row2 in range(np.size(matrix, 0)):
                if row2 > current_row:
                    if matrix[row2, current_col] != 0:
                        matrix[[current_row, row2]] = matrix[[row2, current_row]]
                        print(colored("Switched line:", red), current_row, "with", row2, "\n", matrix, "\n")
                        current_col -= 1
                        break
            current_col += 1
        else:
            matrix[current_row, :] = matrix[current_row, :] / next_element
            print(colored("Convert Diagonal element to One:", "red"), "Column", current_col, ":  \n", matrix, "\n")
            for row2 in range(np.size(matrix, 0)):
                if row2 != current_row:
                    row_diag = matrix[row2, current_col]
                    if row_diag != 0:
                        op_sign = 1 if row_diag > 0 else -1
                        matrix[row2, :] = matrix[row2, :] - abs(row_diag) * op_sign * matrix[current_row, :]
            print(colored("Above & below current Diagonal element to Zero:", "red"), "Column", current_col, ":  \n", matrix, "\n")
            current_row += 1
            current_col += 1
    return matrix


# Start
init()
# test_matrix = [[1.0, 2.0, 0.0, 1.0], [1.0, 2.0, 2.0, 3.0], [4.0, 8.0, 2.0, 6.0], [3.0, 6.0, 4.0, 8.0]]
# test_matrix = [[0, 2.0, 2.0, 2.0], [2.0, 4.0, 8.0, 16.0], [3.0, 5.0, 7.0, 9.0]]
# test_matrix = [[1.0, 0.0, 2.0], [2.0, -1.0, 3.0], [4.0, 1.0, 8.0]]
print(colored("\nGaussian elimination calculator v1 (using reduced row-echelon form)", "yellow"))
print(colored("2021 by Patrick Blauth\n", "yellow"))
selection = input("Select (e)quation solving or (i)nverting\n")
method = 'x_values' if selection == 'e' else 'inverse'
raw_matrix = []

while input != "q":
    user_input = input("Enter matrix line (numbers with space-separated) or (q)uit for starting the calculation\n")
    if user_input == 'q':
        break
    x = list(map(float, user_input.split()))
    raw_matrix.append(x)

aug_A = np.array(raw_matrix) # raw_matrix
if method == 'inverse':
    identity_matrix = np.identity(np.size(aug_A, 1))
    aug_A = np.append(aug_A, identity_matrix, 1)
loop_end = (np.size(aug_A, 1) - 1) if method != 'inverse' else (np.size(aug_A, 1) / 2)
print(colored("\nStart matrix\n", "red"), aug_A, "\n")

aug_A = apply_gaussian(aug_A, method)

if method == 'x_values':
    # check for rows with koef.mat. only 0
    # if yes, check same row b vec value
        # != 0 -> no solution
        # == 0 -> for each row, select a variable for x-value and substitute
    print(colored("Final augmented matrix in reduced row echelon form:" , "red"), "\n", aug_A, "\n")
    for i in range(np.size(aug_A, 1) - 1):
        print("x", i, " = ", aug_A[i, np.size(aug_A, 1) - 1], sep='')
else:
    half_length = int(np.size(aug_A, 1) / 2)
    inv_matrix = aug_A[:, half_length:]
    print(colored("Inverted matrix:\n", "red"), inv_matrix, "\n")
