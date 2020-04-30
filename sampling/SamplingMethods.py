import random
import numpy as np
from scipy.spatial.distance import cdist
import cffi
import sys
import time
import math
import functools
from utils.thread import FuncThread


blue_noise_fail_rate = 0.1


class SamplingBase(object):
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        return

    def sample(self, data, category):
        raise NotImplementedError("Sampling not implemented")


class RandomSampling(SamplingBase):
    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        perm = np.random.permutation(n)
        selected_indexes = perm[:m]
        return selected_indexes


class DensityBiasedSampling(SamplingBase):
    def sample(self, data, category=None):
        k = 50
        X = np.array(data.tolist(), dtype=np.float64)
        n, d = X.shape
        m = round(n * self.sampling_rate)
        if k + 1 > n:
            k = int((n - 1) / 2)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        print(dist)
        radius_of_k_neighbor = dist[:, -1]
        for i in range(len(radius_of_k_neighbor)):
            radius_of_k_neighbor[i] = math.sqrt(radius_of_k_neighbor[i])
        maxD = np.max(radius_of_k_neighbor)
        minD = np.min(radius_of_k_neighbor)
        for i in range(len(radius_of_k_neighbor)):
            radius_of_k_neighbor[i] = ((radius_of_k_neighbor[i] - minD) * 1.0 / (maxD - minD)) * 0.5 + 0.5
        prob = radius_of_k_neighbor
        prob = prob / prob.sum()
        selected_indexes = np.random.choice(n, m, replace=False, p=prob)
        return selected_indexes


class BlueNoiseSampling(SamplingBase):
    def __init__(self, sampling_rate, failure_tolerance=1000):
        super(BlueNoiseSampling, self).__init__(sampling_rate)
        self.failure_tolerance = failure_tolerance
        return

    def sample(self, data, category=None):
        n, d = data.shape
        m = round(n * self.sampling_rate)
        selected_indexes = []

        k = int(1 / self.sampling_rate)
        X = np.array(data.tolist(), dtype=np.float64)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        radius = np.average(np.sqrt(dist[:, -1]))

        count = 0
        while count < m:
            failure_tolerance = min(5000, (n - m) * blue_noise_fail_rate)
            perm = np.random.permutation(n)
            fail = 0
            for idx in perm:
                if fail > failure_tolerance or count >= m:
                    break
                success = True
                for selected_id in selected_indexes:
                    if sum((data[idx] - data[selected_id])**2) < radius**2:
                        success = False
                        break
                if success:
                    count += 1
                    selected_indexes.append(idx)
                else:
                    fail += 1
                # print(count, fail)
            radius /= 2

        selected_indexes = np.array(selected_indexes)
        return selected_indexes


class OutlierBiasedRandomSampling(SamplingBase):
    def sample(self, data, category):
        k = 50
        X = np.array(data.tolist(), dtype=np.float64)
        n, d = X.shape
        m = round(n * self.sampling_rate)
        if k + 1 > n:
            k = int((n - 1) / 2)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        neighbor_labels = category[neighbor]
        outlier_score = [sum(neighbor_labels[i] != category[i]) for i in range(data.shape[0])]
        prob = np.array(outlier_score) / k + 1
        prob = prob / prob.sum()
        selected_indexes = np.random.choice(n, m, replace=False, p=prob)
        return selected_indexes


class ZOrderSampling(SamplingBase):
    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        # z_order_list = np.array(self._construct_z_order(range(n), 0, 1, 0, 1, data))
        z_order_list = self._construct_z_order(list(range(n)), data)
        sampled_indexes = (np.array(range(m)) / self.sampling_rate).astype(np.int)
        selected_indexes = z_order_list[sampled_indexes]
        return selected_indexes

    # def _construct_z_order(self, indexes, xl, xr, yl, yr, data):
    #     if len(indexes) <= 1:
    #         return indexes
    #     z_bin = [[] for _ in range(4)]
    #     xm = (xl + xr) / 2
    #     ym = (yl + yr) / 2
    #     for i in indexes:
    #         if xl <= data[i][0] < xm and yl <= data[i][1] < ym:
    #             z_bin[0].append(i)
    #         elif xm <= data[i][0] <= xr and yl <= data[i][1] < ym:
    #             z_bin[1].append(i)
    #         elif xl <= data[i][0] < xm and ym <= data[i][1] <= yr:
    #             z_bin[2].append(i)
    #         elif xm <= data[i][0] <= xr and ym <= data[i][1] <= yr:
    #             z_bin[3].append(i)
    #     result = []
    #     result.extend(self._construct_z_order(z_bin[0], xl, xm, yl, ym, data))
    #     result.extend(self._construct_z_order(z_bin[1], xm, xr, yl, ym, data))
    #     result.extend(self._construct_z_order(z_bin[2], xl, xm, ym, yr, data))
    #     result.extend(self._construct_z_order(z_bin[3], xm, xr, ym, yr, data))
    #     return result

    def _construct_z_order(self, indexes, data):
        def compare(i, j):
            if data[i][0] == data[j][0] and data[i][1] == data[j][1]:
                return 0
            xl, xr, yl, yr = 0, 1, 0, 1
            i_zone, j_zone = -1, -1
            while True:
                xm = (xl + xr) / 2
                ym = (yl + yr) / 2
                if xl <= data[i][0] < xm and yl <= data[i][1] < ym:
                    i_zone = 0
                elif xm <= data[i][0] <= xr and yl <= data[i][1] < ym:
                    i_zone = 1
                elif xl <= data[i][0] < xm and ym <= data[i][1] <= yr:
                    i_zone = 2
                elif xm <= data[i][0] <= xr and ym <= data[i][1] <= yr:
                    i_zone = 3
                if xl <= data[j][0] < xm and yl <= data[j][1] < ym:
                    j_zone = 0
                elif xm <= data[j][0] <= xr and yl <= data[j][1] < ym:
                    j_zone = 1
                elif xl <= data[j][0] < xm and ym <= data[j][1] <= yr:
                    j_zone = 2
                elif xm <= data[j][0] <= xr and ym <= data[j][1] <= yr:
                    j_zone = 3
                if i_zone != j_zone:
                    if i_zone < j_zone:
                        return -1
                    else:
                        return 1
                else:
                    if i_zone == 0:
                        xr, yr = xm, ym
                    elif i_zone == 1:
                        xl, yr = xm, ym
                    elif i_zone == 2:
                        xr, yl = xm, ym
                    elif i_zone == 3:
                        xl, yl = xm, ym

        indexes.sort(key=functools.cmp_to_key(compare))
        return np.array(indexes)


class NonUniformSampling(SamplingBase):
    def __init__(self, sampling_rate, grid_size=20):
        super(NonUniformSampling, self).__init__(sampling_rate)
        self.grid_size = grid_size
        return

    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        grid_count, grid_indexes = self._construct_grids(data, self.grid_size)
        sample_count = np.ceil(grid_count * self.sampling_rate).astype(np.int)
        non_empty_grids = np.sum(sample_count > 0)
        max_represneted_density = int(m * 2 / non_empty_grids)
        density_count = np.bincount(grid_count.reshape(-1))
        k = non_empty_grids / max_represneted_density
        tmp = 0
        next_milestone = k
        flag = True
        tmp_density = 1
        density_to_represent = [0]
        for grid_this_density in density_count[1:]:
            # if grid_this_density == 0:
            #     density_to_represent.append(-1)
            #     continue
            if tmp + grid_this_density <= next_milestone:
                density_to_represent.append(tmp_density)
                flag = False
            else:
                if next_milestone - tmp > tmp + grid_this_density - next_milestone or flag:
                    density_to_represent.append(tmp_density)
                    tmp_density += 1
                    flag = True
                else:
                    tmp_density += 1
                    density_to_represent.append(tmp_density)
                    flag = False
                next_milestone += k
            tmp += grid_this_density
        selected_indexes = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid_count[i][j] > 0:
                    this_select = np.random.choice(grid_count[i][j], density_to_represent[grid_count[i][j]], replace=False)
                    # this_select = np.random.choice(grid_count[i][j], sample_count[i][j], replace=False)
                    selected_indexes.append([grid_indexes[i][j][t] for t in this_select])
        selected_indexes = np.concatenate(selected_indexes)
        return selected_indexes

    def _construct_grids(self, data, grid_size):
        discrete_full_data = (data * grid_size).astype(np.int)
        discrete_full_data[np.where(discrete_full_data == grid_size)] -= 1
        grid_count = np.zeros((grid_size, grid_size)).astype(np.int)
        grid_indexes = [[[] for _ in range(grid_size)] for __ in range(grid_size)]
        for k, single_data in enumerate(discrete_full_data):
            grid_count[single_data[0], single_data[1]] += 1
            grid_indexes[single_data[0]][single_data[1]].append(k)
        return grid_count, grid_indexes


class HashmapBasedSampling(SamplingBase):
    # Stratified Sampling
    def __init__(self, sampling_rate, grid_size=20, threshold=10):
        super(HashmapBasedSampling, self).__init__(sampling_rate)
        self.grid_size = grid_size
        self.threshold = threshold
        return

    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        discrete_full_data = (data * self.grid_size).astype(np.int)
        discrete_full_data[np.where(discrete_full_data == self.grid_size)] -= 1
        frozen = np.zeros((self.grid_size, self.grid_size)).astype(np.int)
        counter = np.zeros((self.grid_size, self.grid_size)).astype(np.int)
        perm = np.random.permutation(n)
        count = 0
        selected_idx = []
        for idx in perm:
            i, j = discrete_full_data[idx]
            counter[i][j] += 1
            if not frozen[i][j]:
                selected_idx.append(idx)
                count += 1
                if i > 0 and j > 0:
                    frozen[i - 1][j - 1] = 1
                if i > 0 and j < self.grid_size - 1:
                    frozen[i - 1][j + 1] = 1
                if i < self.grid_size - 1 and j > 0:
                    frozen[i + 1][j - 1] = 1
                if i < self.grid_size - 1 and j < self.grid_size - 1:
                    frozen[i + 1][j + 1] = 1
                if counter[i][j] > self.threshold:
                    if i > 0 and j > 0:
                        frozen[i - 1][j - 1] = 0
                    if i > 0 and j < self.grid_size - 1:
                        frozen[i - 1][j + 1] = 0
                    if i < self.grid_size - 1 and j > 0:
                        frozen[i + 1][j - 1] = 0
                    if i < self.grid_size - 1 and j < self.grid_size - 1:
                        frozen[i + 1][j + 1] = 0
            if count == m:
                break
        selected_indexes = np.array(selected_idx)
        return selected_indexes


class SVDBasedSampling(SamplingBase):
    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        u, s, vt = np.linalg.svd(data.T, full_matrices=True)
        # corr = np.array([vt[0][j]**2 + vt[1][j]**2 for j in range(n)])
        corr = sum(vt**2)
        selected_indexes = corr.argsort()[-m:]
        return selected_indexes


class FarthestPointSampling(SamplingBase):
    def sample(self, data, category=None):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        selected_indexes = []
        first_choice = random.randint(0, n - 1)
        selected_indexes.append(first_choice)
        count = 1
        dist = cdist(np.array([data[first_choice]]), data, metric="euclidean").reshape(-1)
        while count < m:
            next_choice = dist.argmax()
            selected_indexes.append(next_choice)
            new_dist = cdist(np.array([data[next_choice]]), data, metric="euclidean").reshape(-1)
            dist = np.minimum(dist, new_dist)
            count += 1
        return np.array(selected_indexes)


# class DualSpaceSampling(SamplingBase):
#     def __init__(self, sampling_rate, beta=0.01):
#         super(DualSpaceSampling, self).__init__(sampling_rate)
#         self.beta = beta
#         return
#
#     def sample(self, data, category=None):
#         n = data.shape[0]
#         m = round(n * self.sampling_rate)
#         w = 1
#         gamma = 0.2
#         Smax = 2 ** int(math.log2(gamma * n))
#         S0 = 32
#         alpha = 0.066
#         knn_density = self._compute_density_field(data)
#         rbf_spread_param = self.beta * w / np.sqrt(knn_density)
#         self._construct_deformed_grids(data, knn_density, rbf_spread_param, S0, S0, alpha)
#         return None
#
#     def _compute_density_field(self, data):
#         X = np.array(data.tolist(), dtype=np.float64)
#         n, d = X.shape
#         k = int(0.01 * n)
#         if k + 1 > n:
#             k = int((n - 1) / 2)
#         neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
#         radius_of_k_neighbor = dist[:, -1]
#         for i in range(len(radius_of_k_neighbor)):
#             radius_of_k_neighbor[i] = math.sqrt(radius_of_k_neighbor[i])
#         knn_density = k / (n * math.pi * radius_of_k_neighbor**2)
#         return knn_density
#
#     def _construct_deformed_grids(self, data, knn_density, rbf_spread_param, S0, Smax, alpha):
#         S = S0
#         threshold = 1e-3
#         while S <= Smax:
#             # create mesh
#             init_L = 1 / S
#             cx = np.linspace(init_L / 2, 1 - init_L / 2, S)
#             cy = np.linspace(init_L / 2, 1 - init_L / 2, S)
#             mx, my = np.meshgrid(cx, cy)
#             d = np.zeros((S, S))
#             for i in range(S):
#                 for j in range(S):
#                     pos = np.array([mx[i][j], my[i][j]])
#                     d[i][j] = np.sum(np.exp(-np.sum((pos - data)**2, axis=1) / (2 * (rbf_spread_param**2))))
#             d_a = np.average(d)
#             A = d / d_a * (1 / S**2)
#             L = init_L * np.sqrt(A)
#             print(A)
#             gx = np.linspace(0, 1, S + 1)
#             gy = np.linspace(0, 1, S + 1)
#             grid_x, grid_y = np.meshgrid(gx, gy)
#             while True:
#                 A_D = np.zeros((S, S))
#                 for i in range(S):
#                     for j in range(S):
#                         pos = np.array([grid_x[i][j], grid_y[i][j]])
#                         if i < S - 1:
#                             neighbor = np.array([grid_x[i + 1][j], grid_y[i + 1][j]])
#                             this_edge = np.sum((neighbor - pos) ** 2)
#                             if j > 0:
#                                 expected_edge = (L[i][j] + L[i][j - 1]) / 2
#                             else:
#                                 expected_edge = L[i][j]
#                             pos += alpha * (expected_edge - this_edge) / 2 * (neighbor - pos) / this_edge
#                         if j < S - 1:
#                             neighbor = np.array([grid_x[i][j + 1], grid_y[i][j + 1]])
#                             this_edge = np.sum((neighbor - pos) ** 2)
#                             if i > 0:
#                                 expected_edge = (L[i][j] + L[i - 1][j]) / 2
#                             else:
#                                 expected_edge = L[i][j]
#                             pos += alpha * (expected_edge - this_edge) / 2 * (neighbor - pos) / this_edge
#                         grid_x[i][j], grid_y[i][j] = pos[0], pos[1]
#                 for i in range(S - 1):
#                     for j in range(S - 1):
#                         xl = [grid_x[i][j], grid_x[i + 1][j], grid_x[i + 1][j + 1], grid_x[i][j + 1]]
#                         yl = [grid_y[i][j], grid_y[i + 1][j], grid_y[i + 1][j + 1], grid_y[i][j + 1]]
#                         A_D[i][j] = self._polygon_area(xl, yl)
#                 sse = np.sum((A_D - A)**2)
#                 flag = False
#                 if sse < threshold:
#                     flag = True
#                 for i in range(S - 1):
#                     for j in range(S - 1):
#                         if grid_x[i][j] > grid_x[i][j + 1] or grid_y[i][j] > grid_y[i + 1][j]:
#                             flag = True
#                 if flag:
#                     break
#             S *= 2
#
#     def _polygon_area(self, x, y):
#         area = 0
#         for i in range(len(x) - 1):
#             area += (x[i] * y[i + 1] - x[i + 1] * y[i]) / 2
#         return abs(area)


class MultiViewZOrderSampling(SamplingBase):
    def sample(self, data, category):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        z_lists = []
        for i, _ in enumerate(np.bincount(category)):
            z_lists.append(self._construct_z_order(np.where(category == i)[0].tolist(), data))
        z_lists.append(self._construct_z_order(list(range(n)), data))
        last, curr = -1, 0
        to_sample = m
        try_list = []
        now_best = -1
        best_candidates = None
        while to_sample not in try_list:
            sr = to_sample / n
            try_list.append(to_sample)
            sets = self._construct_sets(z_lists, sr)
            selected_indexes = self._solve_set_cover(sets, n)
            print(sr, to_sample, selected_indexes.shape[0])
            if selected_indexes.shape[0] == m:
                print(selected_indexes.shape[0])
                return selected_indexes
            else:
                if last != selected_indexes.shape[0]:
                    to_sample = to_sample - (selected_indexes.shape[0] - m)
                else:
                    to_sample = to_sample - int((selected_indexes.shape[0] - m) / 2)
                if abs(now_best - m) > abs(selected_indexes.shape[0] - m):
                    now_best = selected_indexes.shape[0]
                    best_candidates = selected_indexes

        return best_candidates

        # sets = self._construct_sets(z_lists, self.sampling_rate)
        # selected_indexes = self._solve_set_cover(sets, n)
        # return selected_indexes

    # def _construct_z_order(self, indexes, xl, xr, yl, yr, data):
    #     if len(indexes) <= 1:
    #         return indexes
    #     z_bin = [[] for _ in range(4)]
    #     xm = (xl + xr) / 2
    #     ym = (yl + yr) / 2
    #     for i in indexes:
    #         if xl <= data[i][0] < xm and yl <= data[i][1] < ym:
    #             z_bin[0].append(i)
    #         elif xm <= data[i][0] <= xr and yl <= data[i][1] < ym:
    #             z_bin[1].append(i)
    #         elif xl <= data[i][0] < xm and ym <= data[i][1] <= yr:
    #             z_bin[2].append(i)
    #         elif xm <= data[i][0] <= xr and ym <= data[i][1] <= yr:
    #             z_bin[3].append(i)
    #     result = []
    #     for i in range(4):
    #         if i == 0:
    #             result.extend(self._construct_z_order(z_bin[i], xl, xm, yl, ym, data))
    #         elif i == 1:
    #             result.extend(self._construct_z_order(z_bin[i], xm, xr, yl, ym, data))
    #         elif i == 2:
    #             result.extend(self._construct_z_order(z_bin[i], xl, xm, ym, yr, data))
    #         elif i == 3:
    #             result.extend(self._construct_z_order(z_bin[i], xm, xr, ym, yr, data))
    #     return result

    def _construct_z_order(self, indexes, data):
        def compare(i, j):
            if data[i][0] == data[j][0] and data[i][1] == data[j][1]:
                return 0
            xl, xr, yl, yr = 0, 1, 0, 1
            i_zone, j_zone = -1, -1
            while True:
                xm = (xl + xr) / 2
                ym = (yl + yr) / 2
                if xl <= data[i][0] < xm and yl <= data[i][1] < ym:
                    i_zone = 0
                elif xm <= data[i][0] <= xr and yl <= data[i][1] < ym:
                    i_zone = 1
                elif xl <= data[i][0] < xm and ym <= data[i][1] <= yr:
                    i_zone = 2
                elif xm <= data[i][0] <= xr and ym <= data[i][1] <= yr:
                    i_zone = 3
                if xl <= data[j][0] < xm and yl <= data[j][1] < ym:
                    j_zone = 0
                elif xm <= data[j][0] <= xr and yl <= data[j][1] < ym:
                    j_zone = 1
                elif xl <= data[j][0] < xm and ym <= data[j][1] <= yr:
                    j_zone = 2
                elif xm <= data[j][0] <= xr and ym <= data[j][1] <= yr:
                    j_zone = 3
                if i_zone != j_zone:
                    if i_zone < j_zone:
                        return -1
                    else:
                        return 1
                else:
                    if i_zone == 0:
                        xr, yr = xm, ym
                    elif i_zone == 1:
                        xl, yr = xm, ym
                    elif i_zone == 2:
                        xr, yl = xm, ym
                    elif i_zone == 3:
                        xl, yl = xm, ym

        indexes.sort(key=functools.cmp_to_key(compare))
        return np.array(indexes)

    def _construct_sets(self, z_lists, rate):
        sets = []
        for z_list in z_lists:
            m = round(len(z_list) * rate)
            split_indexes = (np.array(range(m)) / m * len(z_list)).astype(np.int)
            tmp = np.split(np.array(z_list), split_indexes[1:])
            sets.extend(tmp)
        return sets

    def _solve_set_cover(self, sets, n):
        # index_occur_count = np.zeros(n)
        can_cover = np.ones(n) * 2
        set_cover_flag = np.zeros(len(sets))
        set_cover_num = 0
        index_to_set = [[] for _ in range(n)]
        for k, s in enumerate(sets):
            for idx in s:
                index_to_set[idx].append(k)
        selected_indexes = []
        while set_cover_num < len(sets):
            # choice = index_occur_count.argmin()
            choice = can_cover.argmax()
            selected_indexes.append(choice)
            for s in index_to_set[choice]:
                if not set_cover_flag[s]:
                    set_cover_flag[s] = True
                    set_cover_num += 1
                    # index_occur_count[sets[s]] += 1
                    for i in sets[s]:
                        can_cover[i] -= 1
        return np.array(selected_indexes)


# class MultiClassBlueNoiseSampling(SamplingBase):
#     def __init__(self, sampling_rate, failure_tolerance):
#         super(MultiClassBlueNoiseSampling, self).__init__(sampling_rate)
#         self.failure_tolerance = failure_tolerance
#         return
#
#     def sample(self, data, category=None):
#         n, d = data.shape
#         m = round(n * self.sampling_rate)
#         w = 100 / self.sampling_rate
#         selected_indexes = []
#         count = 0
#
#         # precomputed
#         class_num = np.max(category) + 1
#         data_by_category = []
#         for c in range(class_num):
#             cate_idx = np.where(category == c)
#             cate_data = data[cate_idx]
#             cate_size = cate_data.shape[0]
#             sigma_x = np.std(cate_data[:, 0]) * (cate_size ** (-1/6))
#             sigma_y = np.std(cate_data[:, 1]) * (cate_size ** (-1/6))
#             data_by_category.append((cate_idx, cate_data, cate_size, sigma_x, sigma_y))
#
#         cate_fill_rate = np.zeros(class_num)
#
#         while count < m:
#             flag = [True] * n
#             failure_tolerance = min(5000, (n - m) * blue_noise_fail_rate)
#             fail = 0
#             perm = [cate_idx[0][np.random.permutation(cate_size)] for (cate_idx, _, cate_size, _, _) in data_by_category]
#             pos = [0] * class_num
#             while count < m and fail < failure_tolerance:
#                 cate_sort = cate_fill_rate.argsort()
#                 for this_cate in cate_sort:
#                     if pos[this_cate] < data_by_category[this_cate][2]:
#                         break
#                 this_cate_size = data_by_category[this_cate][2]
#                 # idx = data_by_category[this_cate][0][0][random.randint(0, this_cate_size - 1)]
#                 # constraint_matrix = self._build_constraint_matrix(idx, data, data_by_category, class_num, w)
#                 idx = perm[this_cate][pos[this_cate]]
#                 pos[this_cate] += 1
#                 if flag[idx] and self._conflict_check(idx, selected_indexes, data, category,
#                                                       self._build_constraint_matrix(idx, data, data_by_category,
#                                                                                     class_num, w)):
#                     selected_indexes.append(idx)
#                     count += 1
#                     cate_fill_rate[category[idx]] += 1 / this_cate_size
#                 else:
#                     fail += 1
#                 flag[idx] = False
#                 print(count, fail, this_cate)
#             w /= 2
#             print("next")
#         selected_indexes = np.array(selected_indexes)
#         return np.array(selected_indexes)
#
#     def _conflict_check(self, idx, selected_idx, data, category, constraint_matrix):
#         dist = cdist(np.array([data[idx]]), data[selected_idx]).reshape(-1)
#         mindist = constraint_matrix[category[idx]][category[selected_idx]]
#         return np.sum(dist > mindist) == len(selected_idx)
#
#     def _build_constraint_matrix(self, idx, data, data_by_category, class_num, w):
#         r_matrix = np.zeros((class_num, class_num))
#         r_diag = np.zeros(class_num)
#         dist = data[idx] - data
#         for c, (cate_idx, cate_data, cate_size, sigma_x, sigma_y) in enumerate(data_by_category):
#             cate_dist = dist[cate_idx]
#             fc = np.sum([self._gaussian_kernel(d, sigma_x, sigma_y) for d in cate_dist])
#             r_matrix[c][c] = w / fc
#             r_diag[c] = r_matrix[c][c]
#         p = np.argsort(-r_diag)
#         C = []
#         D = 0
#         for k in range(class_num):
#             current_cate = p[k]
#             C.append(current_cate)
#             D += 1 / (r_diag[current_cate] ** 2)
#             for j in C:
#                 if current_cate != j:
#                     r_matrix[current_cate][j] = r_matrix[j][current_cate] = 1 / math.sqrt(D)
#         return r_matrix
#
#     def _gaussian_kernel(self, point, sigma_x, sigma_y):
#         return 1 / (2 * math.pi * sigma_x * sigma_y) * \
#                math.exp(-1 / 2 * ((point[0] / sigma_x) ** 2 + (point[1] / sigma_y) ** 2))


class MultiClassBlueNoiseSampling(SamplingBase):
    def __init__(self, sampling_rate, failure_tolerance=1000):
        super(MultiClassBlueNoiseSampling, self).__init__(sampling_rate)
        self.failure_tolerance = failure_tolerance
        return

    def sample(self, data, category=None):
        n, d = data.shape
        m = round(n * self.sampling_rate)
        w = 100 / self.sampling_rate
        selected_indexes = []
        count = 0

        # precomputed
        class_num = np.max(category) + 1
        data_by_category = []
        for c in range(class_num):
            cate_idx = np.where(category == c)
            cate_data = data[cate_idx]
            cate_size = cate_data.shape[0]
            data_by_category.append((cate_idx, cate_data, cate_size))

        cate_fill_rate = np.zeros(class_num)
        constraint_matrix = self._build_constraint_matrix(data_by_category, class_num)

        while count < m:
            flag = [True] * n
            failure_tolerance = min(5000, (n - m) * blue_noise_fail_rate)
            fail = 0
            perm = [cate_idx[0][np.random.permutation(cate_size)] for (cate_idx, _, cate_size) in data_by_category]
            pos = [0] * class_num
            while count < m and fail < failure_tolerance:
                cate_sort = cate_fill_rate.argsort()
                for this_cate in cate_sort:
                    if pos[this_cate] < data_by_category[this_cate][2]:
                        break
                this_cate_size = data_by_category[this_cate][2]
                idx = perm[this_cate][pos[this_cate]]
                pos[this_cate] += 1
                if flag[idx] and self._conflict_check(idx, selected_indexes, data, category, constraint_matrix):
                    selected_indexes.append(idx)
                    count += 1
                    cate_fill_rate[category[idx]] += 1 / this_cate_size
                else:
                    fail += 1
                flag[idx] = False
                print(count, fail, this_cate)
            constraint_matrix /= 2
            print("next")
        selected_indexes = np.array(selected_indexes)
        return np.array(selected_indexes)

    def _conflict_check(self, idx, selected_idx, data, category, constraint_matrix):
        dist = cdist(np.array([data[idx]]), data[selected_idx]).reshape(-1)
        mindist = constraint_matrix[category[idx]][category[selected_idx]]
        return np.sum(dist > mindist) == len(selected_idx)

    def _build_constraint_matrix(self, data_by_category, class_num):
        r_matrix = np.zeros((class_num, class_num))
        r_diag = np.zeros(class_num)
        for c, (cate_idx, cate_data, cate_size) in enumerate(data_by_category):
            k = int(1 / self.sampling_rate)
            if k + 1 > cate_size:
                k = cate_size - 1
            X = np.array(cate_data.tolist(), dtype=np.float64)
            neighbor, dist = Knn(X, cate_size, 2, k + 1, 1, 1, cate_size)
            radius = np.average(np.sqrt(dist[:, -1]))
            r_matrix[c][c] = radius
            r_diag[c] = r_matrix[c][c]
        p = np.argsort(-r_diag)
        C = []
        D = 0
        for k in range(class_num):
            current_cate = p[k]
            C.append(current_cate)
            D += 1 / (r_diag[current_cate] ** 2)
            for j in C:
                if current_cate != j:
                    r_matrix[current_cate][j] = r_matrix[j][current_cate] = 1 / math.sqrt(D)
        return r_matrix


class OutlierBiasedBlueNoiseSampling(SamplingBase):
    def __init__(self, sampling_rate, failure_tolerance=1000):
        super(OutlierBiasedBlueNoiseSampling, self).__init__(sampling_rate)
        self.failure_tolerance = failure_tolerance

    def sample(self, data, category):
        k = 50
        X = np.array(data.tolist(), dtype=np.float64)
        n, d = X.shape
        m = round(n * self.sampling_rate)
        if k + 1 > n:
            k = int((n - 1) / 2)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        neighbor_labels = category[neighbor]
        outlier_score = [sum(neighbor_labels[i] != category[i]) for i in range(data.shape[0])]
        prob = np.array(outlier_score) / k
        prob = prob / 2 + 0.5

        k = int(1 / self.sampling_rate)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        radius = np.average(np.sqrt(dist[:, -1]))

        selected_indexes = []

        count = 0
        while count < m:
            failure_tolerance = min(5000, (n - m) * blue_noise_fail_rate)
            perm = np.random.permutation(n)
            fail = 0
            for idx in perm:
                if fail > failure_tolerance or count >= m:
                    break
                success = True
                for selected_id in selected_indexes:
                    if sum((data[idx] - data[selected_id])**2) < radius**2:
                        success = False
                        break
                if success and np.random.rand() < prob[idx]:
                    count += 1
                    selected_indexes.append(idx)
                else:
                    fail += 1
                # print(count, fail)
            radius /= 2

        selected_indexes = np.array(selected_indexes)
        return selected_indexes


class OutlierBiasedDensityBasedSampling(SamplingBase):
    def __init__(self, sampling_rate, alpha=1, beta=1):
        super(OutlierBiasedDensityBasedSampling, self).__init__(sampling_rate)
        self.alpha = alpha
        self.beta = beta

    def sample(self, data, category):
        k = 50
        X = np.array(data.tolist(), dtype=np.float64)
        n, d = X.shape
        m = round(n * self.sampling_rate)
        if k + 1 > n:
            k = int((n - 1) / 2)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        neighbor_labels = category[neighbor]
        outlier_score = [sum(neighbor_labels[i] != category[i]) for i in range(data.shape[0])]
        outlier_score = np.array(outlier_score) / k

        radius_of_k_neighbor = dist[:, -1]
        for i in range(len(radius_of_k_neighbor)):
            radius_of_k_neighbor[i] = math.sqrt(radius_of_k_neighbor[i])
        maxD = np.max(radius_of_k_neighbor)
        minD = np.min(radius_of_k_neighbor)
        for i in range(len(radius_of_k_neighbor)):
            radius_of_k_neighbor[i] = ((radius_of_k_neighbor[i] - minD) * 1.0 / (maxD - minD)) * 0.5 + 0.5
        prob = self.alpha * radius_of_k_neighbor + self.beta * outlier_score
        prob = prob / prob.sum()
        selected_indexes = np.random.choice(n, m, replace=False, p=prob)
        return selected_indexes


# TODO Change C to Python / DLL
class RecursiveSubdivisionBasedSampling(SamplingBase):
    def sample(self, data, category):
        n = data.shape[0]
        m = round(n * self.sampling_rate)
        perm = np.random.permutation(n)
        selected_indexes = perm[:m]
        return selected_indexes


# Other functions
def Knn(X, N, D, n_neighbors, forest_size, subdivide_variance_size, leaf_number):
    ffi = cffi.FFI()
    ffi.cdef(
        """void knn(double* X, int N, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
            int subdivide_variance_size, int leaf_number);
         """)
    import os
    try:
        t1 = time.time()
        dllPath = os.path.join(config.scripts_root, 'knnDll.dll')
        C = ffi.dlopen(dllPath)
        cffi_X1 = ffi.cast('double*', X.ctypes.data)
        neighbors_nn = np.zeros((N, n_neighbors), dtype=np.int32)
        distances_nn = np.zeros((N, n_neighbors), dtype=np.float64)
        cffi_neighbors_nn = ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = ffi.cast('double*', distances_nn.ctypes.data)
        t = FuncThread(C.knn, cffi_X1, N, D, n_neighbors, cffi_neighbors_nn, cffi_distances_nn, forest_size,
                       subdivide_variance_size, leaf_number)
        t.daemon = True
        t.start()
        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        print("knn runtime = %f" % (time.time() - t1))
        return neighbors_nn, distances_nn
    except Exception as ex:
        print(ex)
    return [[], []]
