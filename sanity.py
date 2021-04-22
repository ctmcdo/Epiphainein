import concurrent.futures
import math
from ortools.sat.python import cp_model  # 8.2.8710
import time
from typing import Final


board_length: Final = 5


# n,    #unreachable,   time
# 3,    0,              < 1 second
# 4,    18,             < 1 second
# 5,    550,            ~ 3 seconds
# 6,    16398,          ~ 2 mins
# 7,    541782,         ~ 2 hours


infeasible_permutations = dict()
feasible_permutations = dict()


def diagram_is_reachable(squares, n):
    model = cp_model.CpModel()

    vars = list()
    top = -1
    left = n
    right = -1
    for s in squares:
        row = s // n
        if top < row:
            top = row

        file = s % n
        if file < left:
            left = file
        if right < file:
            right = file

        cutoff_file_l = file - row
        if cutoff_file_l < 0:
            cutoff_file_l = 0

        cutoff_file_r = file + row
        if (n - 1) < cutoff_file_r:
            cutoff_file_r = n - 1
        vars.append(model.NewIntVar(cutoff_file_l, cutoff_file_r, str(s)))

    model.AddAllDifferent(vars)
    model.Minimize(max(vars) - min(vars))  # minimize width

    solver = cp_model.CpSolver()
    cb = SquaresCallback(vars, n)
    status = solver.SolveWithSolutionCallback(model, cb)

    if status == cp_model.OPTIMAL:
        lmin = min(left, cb.left)
        rmax = max(right, cb.right)
        for j in range((n - 2) - top):
            for i in range(-lmin, n - rmax):
                transformed_squares = list()
                reflected_transformed_squares = list()  # horizontal symmetry
                for s in squares:
                    ts = s + i + (n * j)
                    transformed_squares.append(ts)
                    reflected_transformed_squares.append(
                        n * (ts // n) + ((n - 1) - (ts % n))
                    )
                feasible_permutations[tuple(sorted(transformed_squares))] = True
                feasible_permutations[
                    tuple(sorted(reflected_transformed_squares))
                ] = True
        return True
    if status == cp_model.FEASIBLE or status != cp_model.INFEASIBLE:
        exit(format("Unexpected status: {}", status))

    infeasible_permutations[tuple(squares)] = True
    infeasible_permutations[
        tuple(
            sorted([n * (s // n) + ((n - 1) - (s % n)) for s in squares])
        )  # horizontal symmetry
    ] = True
    return False


class SquaresCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, n):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.left = n
        self.right = -1

    def on_solution_callback(self):
        for v in self.__variables:
            if self.Value(v) < self.left:
                self.left = self.Value(v)
            if self.right < self.Value(v):
                self.right = self.Value(v)


def recurse_diagrams(squares, n):
    if len(squares) > n:
        return 0

    last_square = -1  # if remains -1, then we start at -1 + 1 = square 0
    if len(squares) > 0:
        last_square = squares[-1]

    reachable = 0
    for s in range(last_square + 1, n * (n - 2)):
        new_squares = squares + [s]

        new_squares_name = tuple(new_squares)
        if new_squares_name in infeasible_permutations:
            continue
        if new_squares_name in feasible_permutations or diagram_is_reachable(
            new_squares, n
        ):
            if len(new_squares) == 1:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    reachable += (
                        1 + executor.submit(recurse_diagrams, new_squares, n).result()
                    )
            else:
                reachable += 1 + recurse_diagrams(new_squares, n)
    return reachable


if __name__ == "__main__":
    start = time.time()
    total = 0
    for i_ in range(0, board_length + 1):
        total += math.comb((board_length * (board_length - 2)), i_)
    num_reachable = recurse_diagrams(list(), board_length) + 1  # +1 for empty board
    print("On {0}x{0} board, there are:".format(board_length))
    print("{} unreachable diagrams".format(total - num_reachable))
    print("{} reachable diagrams".format(num_reachable))
    print("The % unreachable is: {}".format(100 * ((total - num_reachable) / total)))
    print("Execution time: {:f}s".format(time.time() - start))
