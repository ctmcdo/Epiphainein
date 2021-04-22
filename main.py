from collections import defaultdict
import cython
import itertools
import math
import numpy as np
from operator import itemgetter
import re
import time
from typing import Dict, Final


# board length
n: Final = 8

# n     #unreachable    %unreachable    time
# 3     0               00.00%          < 1s
# 4     18              11.04%          < 1s
# 5     550             11.12%          < 1s
# 6     16398           08.63%          < 1s
# 7     541782          06.20%          < 1s
# 8     20217623        04.35%          ~ 3s
# 9     851074312       03.02%          ~ 3m
# 10    40168190051     02.10%          ~ 4h

max_num_intervals = int((n - 2) * ((n - 2) + 1) / 2 + 1)
ds_stack_size = max_num_intervals ** 2  # this is a bound
# but I don't think the below are
ds_solutions_singles_size = max_num_intervals ** 3
ds_solutions_pawns_size = max_num_intervals ** 4


def unreachable_pawn_diagram_count():
    if n < 4:
        return 0

    escc = SatisfiableComponentCollector(True)
    # wppsc = component width, #pawns, parity (of #intervals), #squares, #combinations
    edge_wppsc = escc.collect()

    non_edge_wppsc = recursive_defaultdict()
    if 4 < n:
        non_edge_wppsc = SatisfiableComponentCollector(
            False, unsatisfiable_cores=escc.unsatisfiable_cores
        ).collect()
        # there won't be additions to unsat cores for non_edge components

    d = np.uint64(0)
    scce = SatComponentCombinationEnumeration(non_edge_wppsc, edge_wppsc)
    for cs, ecs, p, permutation_factor in scce:
        (
            even_num_interval_diagrams,
            odd_num_interval_diagrams,
        ) = count_square_combinations(cs, ecs, non_edge_wppsc, edge_wppsc, p)
        even_num_interval_diagrams *= permutation_factor
        odd_num_interval_diagrams *= permutation_factor

        # inclusion-exclusion
        d += odd_num_interval_diagrams
        d -= even_num_interval_diagrams
    return d


Interval = cython.struct(a=cython.int, b=cython.int)
min_edge_interval_length: cython.int = 2
min_non_edge_interval_length: cython.int = 3

np_pair = np.dtype([("0", np.int32), ("1", np.int32)])
np_triple = np.dtype([("0", np.int32), ("1", np.int32), ("2", np.int32)])
np_quadruple = np.dtype(
    [("0", np.int32), ("1", np.uint64), ("2", np.int32), ("3", np.uint64)]
)


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


class SatisfiableComponentCollector:
    comma_prefix_re: Final = re.compile("^,*")
    dig_re: Final = re.compile("^\\d*:,*")
    comma_suffix_re: Final = re.compile(",*$")

    def __init__(self, edge, unsatisfiable_cores=None):
        self._edge: cython.bint = edge
        self.unsatisfiable_cores: Dict[str, bool] = unsatisfiable_cores or dict()
        self._max_component_width: cython.int = n if edge else n - 2
        self._initial_interval_endpoint: cython.int = (
            min_edge_interval_length if edge else min_non_edge_interval_length
        )
        self._end_interval: Interval = (
            (
                self._max_component_width - min_edge_interval_length,
                self._max_component_width,
            )
            if edge
            else (
                self._max_component_width - min_non_edge_interval_length,
                self._max_component_width,
            )
        )
        self._intervals = np.zeros(max_num_intervals, dtype=np_pair)
        self._intervals_len: cython.int = 0
        self._ds_stack = np.zeros(ds_stack_size, dtype=np_triple)  # disjoint sets stack
        self._ds_stack_indices = np.zeros(max_num_intervals + 2, np.int32)
        # (immutable_pawns, immutable_variations, total_pawns, total_variations)
        self._ds_solutions_singles = np.zeros(
            ds_solutions_singles_size, dtype=np_quadruple
        )
        self._ds_solutions_singles[0] = (0, 1, 0, 1)
        self._ds_solutions_singles_indices = np.zeros(max_num_intervals + 2, np.int32)
        self._ds_solutions_singles_indices[1] = 1
        self._ds_solutions_pawns = np.zeros(ds_solutions_pawns_size, dtype=np.int32)
        self._ds_solutions_pawns_indices = np.zeros(max_num_intervals + 2, np.int32)
        self._squares_stack = np.zeros(max_num_intervals + 1, np.int32)
        self._width_stack = np.zeros(max_num_intervals + 1, np.int32)
        self._initial_max_interval_width: Interval = (
            self._max_component_width - 1 if edge else self._max_component_width
        )  # resetting to n - 1 for edges prevents the always unsatisfiable (0, n) interval
        self._current_max_interval_width: cython.int = self._initial_max_interval_width
        self._wppsc = recursive_defaultdict()

    def collect(self):
        success: cython.bint = True
        success = self.__next(success)
        while success is not None:
            success = self.__next(success)

        return self._wppsc

    #
    # high level enumeration methods

    def __component_str(self, reverse: cython.bint = False):
        if not reverse:
            intervals = self._intervals[: self._intervals_len]
        else:
            r: cython.int = max(
                self._intervals[: self._intervals_len], key=itemgetter(1)
            )[1]
            intervals = [
                (r - ival[1], r - ival[0])
                for ival in reversed(self._intervals[: self._intervals_len])
            ]

        component_str_arr = [""] * n
        ival: Interval
        for ival in intervals:
            w: cython.int = str(ival[1] - ival[0])
            if component_str_arr[ival[0]] == "":  # nothing on this file yet
                component_str_arr[ival[0]] = w + ":"
            else:
                component_str_arr[ival[0]] += w + ":"
        return self.comma_suffix_re.sub("", ",".join(component_str_arr))

    def __component_contains_previous_unsatisfiable_core(self):
        component_str = self.comma_prefix_re.sub("", self.__component_str())
        while len(component_str) > 0:
            if self.unsatisfiable_cores.__contains__(component_str):
                return True
            component_str = self.dig_re.sub("", component_str)
        return False

    def __lexicographically_add_next_interval(self, pseudo_last=None):
        last: Interval
        if pseudo_last is None:
            last = self._intervals[self._intervals_len - 1].item()
        else:
            last = pseudo_last

        # if we can add one wider on same file, do so
        if (last[1] != self._max_component_width) and (
            (last[1] - last[0] + 1) <= self._current_max_interval_width
        ):
            self._intervals[self._intervals_len] = (last[0], last[1] + 1)
        else:  # else add one starting on file to the right
            r = last[0] + 1 + min_non_edge_interval_length
            if r > self._max_component_width:
                r = self._max_component_width
            self._intervals[self._intervals_len] = (last[0] + 1, r)
        self._intervals_len += 1

    def __add_component_to_wppsc(self):
        w: cython.int = self._width_stack[self._intervals_len]
        parity: cython.bint = self._intervals_len % 2
        s: cython.int = self._squares_stack[self._intervals_len]

        for si in range(
            self._ds_solutions_singles_indices[self._intervals_len],
            self._ds_solutions_singles_indices[self._intervals_len + 1],
        ):
            p: cython.int = self._ds_solutions_singles[si][2]
            c: cython.int = self._ds_solutions_singles[si][3]

            try:
                self._wppsc[w][p][parity][s] += c
            except TypeError:
                self._wppsc[w][p][parity][s] = c

    def __next(self, curr_is_feasible: cython.bint):
        if self._intervals_len == 0:  # initial values
            self._intervals[self._intervals_len] = (0, self._initial_interval_endpoint)
            self._intervals_len += 1
            self.__find_solutions()
            self.__add_component_to_wppsc()
            return True

        last_interval: Interval = self._intervals[self._intervals_len - 1].item()
        if not curr_is_feasible:
            self._current_max_interval_width = last_interval[1] - last_interval[0]

        if (last_interval == self._end_interval) or (  # can't add any other interval
            not curr_is_feasible
            and (
                (  # or replace [-1] with interval (l, r + 1), because it would also
                    # be infeasible, or replace with (l + 1, r) because (1) it would
                    # be too small
                    self._current_max_interval_width
                    <= min_non_edge_interval_length
                )
                or (  # or (2) it would result in disconnected intervals
                    last_interval[0]
                    == (
                        max(
                            self._intervals[: self._intervals_len - 1],
                            key=itemgetter(1),
                        )[1]
                        - 1
                    )
                )
            )
        ):
            if (
                self._intervals[0][0] == 0
                and self._intervals[0][1] == self._initial_max_interval_width
                and self._intervals_len <= 2
            ):
                return

            # so remove [-1]
            self._intervals_len -= 1
            self._current_max_interval_width = self._initial_max_interval_width

            # and replace [-2] with next in lexicographic sequence
            previous = self._intervals[self._intervals_len - 1].item()
            self._intervals_len -= 1
            self.__lexicographically_add_next_interval(previous)

        elif not curr_is_feasible:  # replace [-1] with next in lexicographic sequence
            self._intervals_len -= 1
            self.__lexicographically_add_next_interval(last_interval)

        else:  # add next
            self.__lexicographically_add_next_interval()

        if self.__component_contains_previous_unsatisfiable_core():
            return False

        found_solutions = self.__find_solutions()
        if not found_solutions:
            self.unsatisfiable_cores[self.__component_str()] = True
            self.unsatisfiable_cores[self.__component_str(reverse=True)] = True
        else:
            self.__add_component_to_wppsc()

        return found_solutions

    #
    # satisfiability related methods

    def __ival_num_squares(self, ival: Interval):
        ilen: cython.int = ival[1] - ival[0]
        if ilen >= n:
            exit("Received ival with length >= board length")

        num_squares: cython.int
        if self._edge and (ival[0] == 0 or ival[1] == n):
            num_squares = (ilen * (ilen + 1)) // 2  # sum from 1 to ilen
            if ilen == (n - 1):
                num_squares -= 1
        else:
            if ilen % 2 == 0:
                # sum of even numbers up to ilen
                num_squares = (ilen // 2) * ((ilen // 2) + 1)
            else:
                # sum of odd numbers up to ilen
                num_squares = ((ilen // 2) + 1) ** 2
        return num_squares

    def __segment_squares(self, a, b=None):
        squares: cython.int = self.__ival_num_squares(a)
        if not self._first_segment and b is not None:
            squares -= self.__ival_num_squares(b)

        self._first_segment = False
        return squares

    def __find_solutions(self):
        ival: Interval = self._intervals[self._intervals_len - 1].item()
        self._first_segment: cython.bint = True

        curr_segments_pointer: cython.int = self._ds_stack_indices[self._intervals_len]
        prev_segments_pointer: cython.int = self._ds_stack_indices[
            self._intervals_len - 1
        ]
        prev_pawn_segments_size: cython.int = (
            curr_segments_pointer - prev_segments_pointer
        )
        i = 0  # i is inclusive
        for i in range(0, prev_pawn_segments_size):
            if ival[0] < self._ds_stack[prev_segments_pointer + i][1]:
                break

        j: cython.int = -1  # j is exclusive
        clean_break: cython.bint = True
        for j in range(i, prev_pawn_segments_size):
            p1: cython.int = (
                -1
                if self._first_segment
                else self._ds_stack[prev_segments_pointer + j][0]
            )
            p2: cython.int = min(ival[1], self._ds_stack[prev_segments_pointer + j][1])
            squares = self.__segment_squares(
                (ival[0], p2), (ival[0], self._ds_stack[prev_segments_pointer + j][0])
            )
            self._ds_stack[curr_segments_pointer + j - i] = (p1, p2, squares)
            if ival[1] < self._ds_stack[prev_segments_pointer + j][1]:
                clean_break = False
                self._ds_stack[curr_segments_pointer + j - i + 1] = (
                    ival[1],
                    self._ds_stack[prev_segments_pointer + j][1],
                    self._ds_stack[prev_segments_pointer + j][2] - squares,
                )
            if ival[1] <= self._ds_stack[prev_segments_pointer + j][1]:
                break
        j += 1

        for k in range(j, prev_pawn_segments_size):
            self._ds_stack[
                curr_segments_pointer + (not clean_break) + k - i
            ] = self._ds_stack[prev_segments_pointer + k]

        new_segment_squares: cython.int = 0
        if ival[1] > self._width_stack[self._intervals_len - 1]:
            p1: cython.int = (
                -1
                if self._first_segment
                else self._width_stack[self._intervals_len - 1]
            )
            new_segment_squares: cython.int = self.__segment_squares(
                ival, (ival[0], self._width_stack[self._intervals_len - 1])
            )
            self._ds_stack[curr_segments_pointer + prev_pawn_segments_size - i] = (
                p1,
                ival[1],
                new_segment_squares,
            )

        growth: cython.bint = (not clean_break) + (new_segment_squares > 0)
        curr_pawn_segments_size: cython.int = prev_pawn_segments_size - i + growth
        self._ds_stack_indices[self._intervals_len + 1] = (
            curr_segments_pointer + curr_pawn_segments_size
        )

        prev_solutions_singles_pointer: cython.int = self._ds_solutions_singles_indices[
            self._intervals_len - 1
        ]
        prev_solutions_pawns_pointer: cython.int = self._ds_solutions_pawns_indices[
            self._intervals_len - 1
        ]
        self._soln_count: cython.int = 0
        for si in range(
            0,
            self._ds_solutions_singles_indices[self._intervals_len]
            - prev_solutions_singles_pointer,
        ):
            capacity: cython.int = 0
            for k in range(0, j - i):
                new_squares: cython.int = self._ds_stack[curr_segments_pointer + k][2]
                parent_pawns: cython.int = self._ds_solutions_pawns[
                    prev_solutions_pawns_pointer
                    + (si * prev_pawn_segments_size)
                    + i
                    + k
                ]
                capacity += min(new_squares, parent_pawns)

            it = self._ds_solutions_singles[prev_solutions_singles_pointer + si]
            capacity += min(new_segment_squares, n - it[2])

            ilen: cython.int = ival[1] - ival[0]
            if capacity <= ilen:
                continue

            immutable_pawns: cython.int = it[0]
            immutable_variations: cython.int = it[1]
            for k in range(0, i):
                p: cython.int = self._ds_solutions_pawns[
                    prev_solutions_pawns_pointer + (si * prev_pawn_segments_size) + k
                ]
                immutable_pawns += p
                immutable_variations *= binomial_coefficients[
                    self._ds_stack[prev_segments_pointer + k][2]
                ][p]

            partial_pawns_solution = np.zeros(curr_pawn_segments_size, np.int32)
            variations: cython.int = 1
            for k in range(j, prev_pawn_segments_size):
                p: cython.int = self._ds_solutions_pawns[
                    prev_solutions_pawns_pointer + (si * prev_pawn_segments_size) + k
                ]
                partial_pawns_solution[(not clean_break) + k - i] = p
                variations *= binomial_coefficients[
                    self._ds_stack[prev_segments_pointer + k][2]
                ][p]

            self.__recurse_solutions(
                immutable_pawns,
                immutable_variations,
                it[2],
                variations,
                i,
                j,
                clean_break,
                ilen + 1,
                si,
                capacity,
                partial_pawns_solution,
            )

        if self._soln_count == 0:
            return False

        self._ds_solutions_singles_indices[self._intervals_len + 1] = (
            self._ds_solutions_singles_indices[self._intervals_len] + self._soln_count
        )
        self._ds_solutions_pawns_indices[
            self._intervals_len + 1
        ] = self._ds_solutions_pawns_indices[self._intervals_len] + (
            self._soln_count * curr_pawn_segments_size
        )

        self._width_stack[self._intervals_len] = max(
            self._width_stack[self._intervals_len - 1], ival[1]
        )
        self._squares_stack[self._intervals_len] = (
            self._squares_stack[self._intervals_len - 1] + new_segment_squares
        )

        return True

    def __recurse_solutions(
        self,
        immutable_pawns,
        immutable_variations,
        total_pawns,
        variations,
        i,
        j,
        clean_break,
        still_required,
        si,
        still_available,
        partial_pawns_solution,
        k=0,
    ):
        parent_squares: cython.int = 0
        new_squares: cython.int = self._ds_stack[
            self._ds_stack_indices[self._intervals_len] + k
        ][2]

        num_refined_partitions: cython.int = j - i
        if k < num_refined_partitions:
            parent_squares: cython.int = self._ds_stack[
                self._ds_stack_indices[self._intervals_len - 1] + i + k
            ][2]
            prev_segment_size: cython.int = (
                self._ds_stack_indices[self._intervals_len]
                - self._ds_stack_indices[self._intervals_len - 1]
            )
            parent_pawns: cython.int = self._ds_solutions_pawns[
                self._ds_solutions_pawns_indices[self._intervals_len - 1]
                + (si * prev_segment_size)
                + i
                + k
            ]
            capacity: cython.int = min(new_squares, parent_pawns)

        else:
            parent_pawns: cython.int = 0
            capacity: cython.int = min(new_squares, n - total_pawns)

        squares_left_in_parent: cython.int = parent_squares - new_squares
        max_pawns_to_be_left_in_parent: cython.int = max(0, squares_left_in_parent)
        set_requirement: cython.int = max(
            0,
            parent_pawns - max_pawns_to_be_left_in_parent,
            still_required - (still_available - capacity),
        )
        for p in range(set_requirement, capacity + 1):
            partial_pawns_solution[k] = p

            _immutable_pawns: cython.int = immutable_pawns
            _immutable_variations: cython.int = immutable_variations
            _variations: cython.int = variations * binomial_coefficients[new_squares][p]
            if k < num_refined_partitions:
                pawns_left_in_parent: cython.int = parent_pawns - p
                c: cython.int = (
                    1
                    if squares_left_in_parent == 0
                    else binomial_coefficients[squares_left_in_parent][
                        pawns_left_in_parent
                    ]
                )
                if k == (num_refined_partitions - 1) and not clean_break:
                    partial_pawns_solution[k + 1] = pawns_left_in_parent
                    _variations *= c

                else:
                    _immutable_pawns += pawns_left_in_parent
                    _immutable_variations *= c

            if still_available - capacity > 0:
                self.__recurse_solutions(
                    _immutable_pawns,
                    _immutable_variations,
                    total_pawns,
                    _variations,
                    i,
                    j,
                    clean_break,
                    still_required - p,
                    si,
                    still_available - capacity,
                    partial_pawns_solution,
                    k + 1,
                )

            else:
                self._ds_solutions_singles[
                    self._ds_solutions_singles_indices[self._intervals_len]
                    + self._soln_count
                ] = (
                    _immutable_pawns,
                    _immutable_variations,
                    total_pawns + (p if k == num_refined_partitions else 0),
                    _immutable_variations * _variations,
                )

                for m in range(0, len(partial_pawns_solution)):
                    self._ds_solutions_pawns[
                        self._ds_solutions_pawns_indices[self._intervals_len]
                        + (self._soln_count * len(partial_pawns_solution))
                        + m
                    ] = partial_pawns_solution[m]

                self._soln_count += 1


# a component family is a multiset of components
class SatComponentCombinationEnumeration:
    def __iter__(self):
        return self

    def __init__(self, wppsc, wppsce):
        self._wp: Final = self.__parse_wp_(wppsc)
        self._ewp: Final = self.__parse_wp_(wppsce)

        self._curr_comps = []
        self._curr_edge_comps = []
        self._tw = 0
        self._tp = 0

    @staticmethod
    def __parse_wp_(wppsc):
        return [
            [p for p in sorted(wppsc[w].keys(), key=int)]
            for w in sorted(wppsc.keys(), key=int)
        ]

    def __factor(self):
        occupied_edge_squares = len(self._curr_edge_comps)
        # if spans board
        if (
            len(self._curr_edge_comps) == 1
            and (min_edge_interval_length + self._curr_edge_comps[0][0]) == n
        ):
            occupied_edge_squares = 2
            factor = 1
        elif len(self._curr_edge_comps) == 0 or (
            len(self._curr_edge_comps) == 2
            and self._curr_edge_comps[0] == self._curr_edge_comps[1]
        ):
            factor = 1
        else:
            factor = 2

        empty_inner_squares = n - self._tw - (2 - occupied_edge_squares)

        # each component considered distinct
        factor *= math.factorial(
            len(self._curr_comps) + empty_inner_squares
        ) // math.factorial(empty_inner_squares)

        last_comp = (-1, -1)
        streak = 1
        for c in sorted(self._curr_comps):
            if c == last_comp:
                streak += 1
            else:
                # account for indistinct components
                factor //= math.factorial(streak)
                streak = 1
            last_comp = c
        factor //= math.factorial(streak)

        return np.uint64(factor)

    def __next__(self):
        self.__next_helper(self._curr_comps, min_non_edge_interval_length, self._wp)
        if len(self._curr_comps) == 0:
            self.__next_helper(
                self._curr_edge_comps, min_edge_interval_length, self._ewp
            )
            if len(self._curr_edge_comps) == 0:
                raise StopIteration

        # (component family width, pawns within family domain)
        return (
            [
                (min_non_edge_interval_length + c[0], self._wp[c[0]][c[1]])
                for c in self._curr_comps
            ],
            [
                (min_edge_interval_length + c[0], self._ewp[c[0]][c[1]])
                for c in self._curr_edge_comps
            ],
            self._tp,
            self.__factor(),
        )

    def __next_helper(self, comps, mw, wp):
        if comps == self._curr_edge_comps and len(comps) == 2:
            added = False

        elif len(wp) > 0:  # try add
            next_comp = (0, 0)
            if len(comps) > 0:
                next_comp = comps[-1]

            added = self.__add_lexicographic_next_comp(comps, next_comp, mw, wp)

        else:  # no components in wp. Quite an edge case (for n == 4).
            return

        while added is False and len(comps) > 0:  # move,
            if comps[-1][1] < (len(wp[comps[-1][0]]) - 1):
                next_comp = (comps[-1][0], comps[-1][1] + 1)  # (up)
            elif comps[-1][0] < (len(wp) - 1):
                next_comp = (comps[-1][0] + 1, 0)  # (across)
            else:
                next_comp = None  # or finally backtrack

            p = comps.pop()
            self._tw -= mw + p[0]
            self._tp -= wp[p[0]][p[1]]

            if next_comp is not None:
                added = self.__add_lexicographic_next_comp(comps, next_comp, mw, wp)

    def __add_lexicographic_next_comp(self, comps, next_component, mw, wp):
        unoccupied_edges_given_acceptance = (
            2
            - len(self._curr_edge_comps)
            - (1 if comps == self._curr_edge_comps else 0)
        )

        # try add target
        if (
            next_component[0] + mw + self._tw <= n - unoccupied_edges_given_acceptance
        ) and (wp[next_component[0]][next_component[1]] + self._tp <= n):
            comps.append((next_component[0], next_component[1]))
            self._tw += mw + next_component[0]
            self._tp += wp[next_component[0]][next_component[1]]
        elif (  # or if too expensive
            (next_component[0] < (len(wp) - 1))
            # but there's room to move across
            and (
                next_component[0] + 1 + mw + self._tw
                <= n - unoccupied_edges_given_acceptance
            )
            and (wp[next_component[0] + 1][0] + self._tp <= n)  # and we can afford
        ):
            comps.append((next_component[0] + 1, 0))
            self._tw += mw + next_component[0] + 1
            self._tp += wp[next_component[0] + 1][0]

        else:
            return False

        return True


def count_square_combinations(components, edge_components, wppsc, wppsce, p):
    board = n * (n - 2)

    psc = [wppsc[wp[0]][wp[1]] for wp in components]
    psc.extend(wppsce[wp[0]][wp[1]] for wp in edge_components)

    even_combos = np.uint64(0)
    odd_combos = np.uint64(0)
    for psc_ in itertools.product(*psc):
        parity = sum(p for p in psc_) % 2  # odd or even number of intervals

        sc = [psc[i][psc_[i]] for i in range(0, len(psc_))]
        for sc_ in itertools.product(*sc):
            squares = 0
            prod = np.uint64(1)
            for i in range(0, len(sc_)):
                squares += sc_[i]
                prod *= sc[i][sc_[i]]

            combinations = np.uint64(0)
            for rp in range(0, n - p + 1):
                combinations += prod * binomial_coefficients[board - squares][rp]

            if parity == 0:
                even_combos += combinations
            else:
                odd_combos += combinations

    return even_combos, odd_combos


binomial_coefficients = None


def precompute_binomial_coefficients():
    global binomial_coefficients
    lim = n * (n - 2)
    binomial_coefficients = np.zeros(shape=(lim + 1, n + 1), dtype=np.uint64)
    for i in range(0, lim + 1):
        binomial_coefficients[i][0] = 1

    for j in range(1, n + 1):
        for i in range(j, lim + 1):
            binomial_coefficients[i][j] = (
                binomial_coefficients[i - 1][j - 1] + binomial_coefficients[i - 1][j]
            )


if __name__ == "__main__":
    start = time.time()
    precompute_binomial_coefficients()
    total = np.uint64(0)
    for i_ in range(0, n + 1):
        total += binomial_coefficients[(n * (n - 2))][i_]
    num_unreachable = unreachable_pawn_diagram_count()
    print("On {0}x{0} board, there are:".format(n))
    print("{} unreachable diagrams".format(num_unreachable))
    print("{} reachable diagrams".format(total - num_unreachable))
    print("The % unreachable is: {}".format(100 * (num_unreachable / total)))
    print("Execution time: {:f}s".format(time.time() - start))
