#################################
# Your name: Yefim Nudelman
#################################

import numpy as np
from intervals import *
import matplotlib.pyplot as plt
from builtins import min, max, sum
import random


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    P_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.random.rand(m, 1)
        y = np.empty((m, 1), dtype=int)

        i = 0
        for x_item in x:
            chance = 0.1 if 0.2 < x_item < 0.4 or 0.6 < x_item < 0.8 else 0.8
            y[i] = 1 if random.random() <= chance else 0
            i += 1

        sample = np.concatenate((x, y), axis=1)
        return sample[np.argsort(sample[:, 0])]

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        res = []
        m_values = range(m_first, m_last + 1, step)
        for m in m_values:
            emp_error_runs = []
            true_error_runs = []

            for _ in range(T):
                samples = self.sample_from_D(m)
                xs = samples[:, 0]
                ys = samples[:, 1]
                k_hypothesis, error_count = find_best_interval(xs, ys, k)
                empirical_error = error_count / m
                true_error = self.calculate_true_error(k_hypothesis)
                emp_error_runs.append(empirical_error)
                true_error_runs.append(true_error)

            res.append([np.mean(emp_error_runs), np.mean(true_error_runs)])
        res = np.array(res)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(m_values, res[:, 0], label="Empirical Error", marker='o')
        plt.plot(m_values, res[:, 1], label="True Error", marker='x')
        plt.xlabel("Sample Size (m)")
        plt.ylabel("Error")
        plt.title(f"Empirical and True Errors vs. Sample Size, hypothesis with at most (k={k}) intervals")
        plt.legend()
        plt.grid(True)
        plt.show()

        return res

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_values = range(k_first, k_last + 1, step)
        true_errors = []
        emp_errors = []
        samples = self.sample_from_D(m)
        xs = samples[:, 0]
        ys = samples[:, 1]
        for k in k_values:
            k_hypothesis, error_count = find_best_interval(xs, ys, k)
            empirical_error = error_count / m
            emp_errors.append(empirical_error)
            true_error = self.calculate_true_error(k_hypothesis)
            true_errors.append(true_error)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, emp_errors, label="Empirical Error", marker="o")
        plt.plot(k_values, true_errors, label="True Error", marker="x")
        plt.xlabel("Max Number of Intervals (k)")
        plt.ylabel("Error")
        plt.title("Empirical and True Errors vs. Max Number of Intervals (k)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return np.argmin(emp_errors) + 1

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_values = range(k_first, k_last + 1, step)
        true_errors = []
        total_costs = []
        emp_errors = []
        penalties = []
        samples = self.sample_from_D(m)
        xs = samples[:, 0]
        ys = samples[:, 1]
        for k in k_values:
            k_hypothesis, error_count = find_best_interval(xs, ys, k)
            empirical_error = error_count / m
            emp_errors.append(empirical_error)

            true_error = self.calculate_true_error(k_hypothesis)
            true_errors.append(true_error)

            penalty = self.calc_penalty(k, m)
            penalties.append(penalty)

            total_cost = empirical_error + penalty
            total_costs.append(total_cost)

        plt.figure(figsize=(12, 6))

        plt.plot(k_values, emp_errors, label="Empirical Error", marker="o")
        plt.plot(k_values, penalties, label="Penalty", marker="x")
        plt.plot(k_values, total_costs, label="Empirical Error + Penalty", marker="s")
        plt.plot(k_values, true_errors, label="True error")
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error / Penalty")
        plt.title("Empirical Error, Penalty, and Total Cost vs. Number of Intervals (k)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return np.argmin(total_costs) + 1

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        np.random.shuffle(samples)

        split_idx = int(0.8 * m)
        training_set = samples[:split_idx]
        holdout_set = samples[split_idx:]

        training_set = training_set[training_set[:, 0].argsort()]
        xs_train, ys_train = training_set[:, 0], training_set[:, 1]
        xs_hold, ys_hold = holdout_set[:, 0], holdout_set[:, 1]

        hold_size = len(xs_hold)

        validation_errors = []

        for k in range(1, 11):
            k_hypothesis, _ = find_best_interval(xs_train, ys_train, k)
            err_count = 0
            for x, y in zip(xs_hold, ys_hold):
                x = float(x)
                pred = 0
                for interval in k_hypothesis:
                    if interval[0] <= x <= interval[1]:
                        pred = 1
                        break
                if pred != y:
                    err_count += 1
            validation_error = err_count/hold_size
            validation_errors.append(validation_error)

        return np.argmin(validation_errors) + 1

    #################################
    # Place for additional methods

    def calculate_overlap(self, intervals1, intervals2):
        # calculates total overlap between intervals later use
        total_overlap = 0
        p1, p2 = 0, 0

        # Use two pointers to calculate the overlap
        while p1 < len(intervals1) and p2 < len(intervals2):
            # Find the overlapping range
            start = max(intervals1[p1][0], intervals2[p2][0])
            end = min(intervals1[p1][1], intervals2[p2][1])

            # If there's an overlap, add its length
            if start < end:
                total_overlap += end - start

            # Move the pointer of the interval that ends first
            if intervals1[p1][1] < intervals2[p2][1]:
                p1 += 1
            else:
                p2 += 1

        return total_overlap

    def calculate_true_error(self, intervals_h):
        '''
        We will partition the probability into 4 parts
        P(Y != h(X)) = P(Y != h(X)| X in intervals)P(X in intervals) + P(Y != h(X)|X not in P_intervals)P(X not in P_intervals)

        P(Y != h(X) | X in P_intervals) = P(h(X) = 1, Y = 0 | X in P_intervals) + P(h(X) = 0, Y = 1 | X in P_intervals)
        P(Y != h(X) | X not in P_intervals) = P(h(X) = 1, Y = 0 | X not in P_intervals) + P(h(X) = 0, Y = 1 | X not in P_intervals)

        P(h(X) = 1, Y = 0 | X in P_intervals)
        = P(h(X) = 1 | X in P_intervals) * P(Y = 0 | X in P_intervals) second part equal to 0.2 (I)

        P(h(X) = 0, Y = 1 | X in P_intervals)
        = P(h(X) = 0 | X in P_intervals) * P(Y = 1 | X in P_intervals) second part equal to 0.8 (II)

        P(h(X) = 1, Y = 0 | X not in P_intervals)
        = P(h(X) = 1 | X not in P_intervals) * P(Y = 0 | X not in P_intervals) second part equal to 0.9 (III)

        P(h(X) = 0, Y = 1 | X not in P_intervals)
        = P(h(X) = 0 | X not in P_intervals) * P(Y = 1 | X not in P_intervals) second part equal to 0.1 (IV)

        in total we will return
        (I + II)P(X in intervals) + (III + IV)P(X not in P_intervals)
        '''
        # calculate the total length of h
        h_length = sum(interval[1] - interval[0] for interval in intervals_h)

        # calculate the overlap where h(x) = 1 and x in P_intervals
        overlap_h1_in_P = self.calculate_overlap(intervals_h, self.P_intervals)

        # calculate the overlap where h(x) = 1 and x not in P_intervals
        overlap_h1_not_in_P = h_length - overlap_h1_in_P

        # P(h(x) = 1|X in P_intervals) = P(x in overlap)/P(x in P_intervals) = overlap/0.6
        prob_h1_given_P = overlap_h1_in_P / 0.6

        # P(h(x) = 0 | X in P_intervals) = 1 - P(h(x) = 1 | X in P_intervals)
        prob_h0_given_P = 1 - prob_h1_given_P

        # P(h(x) = 1 | X not in P_intervals) = P(x in overlap not in P with h(x) = 1) / P(X not in P_intervals) =
        # overlap/0.4
        prob_h1_given_not_P = overlap_h1_not_in_P / 0.4

        # P(h(x) = 0 | X not in P_intervals) = 1 - P(h(x) = 1 | X not in P_intervals)
        prob_h0_given_not_P = 1 - prob_h1_given_not_P

        return (prob_h1_given_P * 0.2 + prob_h0_given_P * 0.8) * 0.6 + (
                prob_h1_given_not_P * 0.9 + prob_h0_given_not_P * 0.1) * 0.4

    ################################

    def calc_penalty(self, k, n):
        VCdim = 2 * k
        delta = 0.1 / (k ** 2)
        expression = (VCdim + log(2 / delta)) / n
        return 2 * sqrt(expression)


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
