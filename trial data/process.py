import os
import json
import csv
import math
import numpy as np
from scipy import stats
import statsmodels
import scikit_posthocs as sp


dataset_names = ["synthetic", "mnist", "swiss_roll_2d", "swiss_roll_3d", "abalone",
                 "clothes", "crowdsourced_mapping", "epileptic_seizure", "condition_based_maintenance"]

sampling_methods = ["random sampling", "blue noise sampling", "density biased sampling",
                    "multi-view Z order sampling", "multi-class blue noise sampling",
                    "outlier biased density based sampling", "recursive subdivision based sampling"]

sampling_method_class_names = ["RandomSampling", "BlueNoiseSampling", "DensityBiasedSampling",
                    "MultiViewZOrderSampling", "MultiClassBlueNoiseSampling",
                    "OutlierBiasedDensityBasedSampling", "RecursiveSubdivisionBasedSampling"]


if __name__ == "__main__":
    print()
    root_path = "./"
    base_result_path = "publish"
    base_answer_path = "answer"
    
    res = []
    std = []
    nickname = []
    age = []
    gender = []
    degree = []
    filenames = []
    id_str = []
    
    problem_count = 115 + 115 + 59 + 9

    result_path = root_path + base_result_path
    answer_path = root_path + base_answer_path
    files = os.listdir(result_path)
    for file in files:
        if os.path.isdir(file) or file == ".DS_Store":
            continue
        with open(os.path.join(result_path, file), "r") as f:
            content = [line for line in f]
            if len(content) < problem_count:
                continue
            # From user
            d = [None for _ in range(problem_count)]
            flag = problem_count
            for i in range(problem_count):
                problem_info = json.loads(content[i][:-1])
                if d[problem_info['problem_id']] is None:
                    d[problem_info['problem_id']] = problem_info
                    flag -= 1
            if flag:
                continue
            i = problem_count
            okay = True
            if okay:
                filenames.append(file)
                id_str.append(file.split('.')[0])
                res.append(d)
                f.close()
                with open(os.path.join(answer_path, file), "r", encoding='gb18030') as ans_file:
                    # From answer
                    content = [line for line in ans_file]
                    std.append(content)
                    ans_file.close()

    total_obtain_answers = len(res)
    total_each_sampling = total_obtain_answers * 8 * 2
    total_each_dataset = total_obtain_answers * 7 * 2
    print("#Participants: ", total_obtain_answers)
    
    print()
    print("E1")
    # Task 1 - density_1
    task1_acc = np.zeros((total_obtain_answers, 8, 7, 2)).astype(np.int)
    task1_time = np.zeros((total_obtain_answers, 8, 7, 2)).astype(np.float)
    learning_effect_acc = np.zeros((total_obtain_answers, 8, 14)).astype(np.float)
    learning_effect_time = np.zeros((total_obtain_answers, 8, 14)).astype(np.float)
    task1_personal = np.zeros((total_obtain_answers)).astype(np.float)

    for i in range(3, 3 + 112):
        for k in range(total_obtain_answers):
            ans = res[k][i]['problem_answer']['sort_index'][0]
            dataset_index = dataset_names.index(res[k][i]['problem_dataset'])
            sampling_index = sampling_methods.index(res[k][i]['problem_sampling'])
            level = res[k][i]['level']
            ans_index = dataset_index * 7 * 2 + sampling_index * 2 + level + 1
            standard = int(std[k][ans_index][-2])
            assert std[k][ans_index].index(res[k][i]['problem_dataset'] + '-' + str(level) + ' ' + res[k][i]['problem_sampling']) == 0
            if res[k][i]['problem_timespend'] < 30000: # Filter invalid answers caused by network errors
                task1_time[k][dataset_index - 1][sampling_index][level] += res[k][i]['problem_timespend']
                learning_effect_time[k][dataset_index - 1][int((i - 3) / 8)] += res[k][i]['problem_timespend']
            if ans == standard :
                task1_acc[k][dataset_index - 1][sampling_index][level] += 1
                learning_effect_acc[k][dataset_index - 1][int((i - 3) / 8)] += 1
                task1_personal[k] += 1

    task1_acc_avg = [int(np.sum(task1_acc[:,:,i,:])) / total_each_sampling for i in range(7)]
    task1_acc_p = np.array([[np.average(task1_acc[k,:,i,:]) for i in range(7)] for k in range(total_obtain_answers)])
    task1_acc_ci = [np.std(task1_acc_p[:,i]) * 1.96 / math.sqrt(total_obtain_answers) for i in range(7)]
    task1_time_avg = [int(np.average(task1_time[:,:,i,:])) for i in range(7)]
    task1_time_ci = [np.std(task1_time[:,:,i,:]) * 1.96 / math.sqrt(total_obtain_answers * 8 * 2) for i in range(7)]

    print("Task1 Accuracy:\n", task1_acc_avg)
    print("Task1 Accuracy 95\% CI:\n", task1_acc_ci)
    print("Task1 Completion Time:\n", task1_time_avg)
    print("Task1 Completion Time 95\% CI:\n", task1_time_ci)

    task1_acc_detail = [[np.average(task1_acc[:,i,k,j]) for i in range(8) for j in range(2)] for k in range(7)]
    task1_time_detail = [[np.average(task1_time[:,i,k,j]) for i in range(8) for j in range(2)] for k in range(7)]

    acc_fm_res = stats.friedmanchisquare(*task1_acc_detail)
    time_fm_res = stats.friedmanchisquare(*task1_time_detail)

    print(acc_fm_res)
    print(sp.posthoc_conover_friedman(np.array(task1_acc_detail).T, p_adjust=None))
    print(time_fm_res)
    print(sp.posthoc_conover_friedman(np.array(task1_time_detail).T))

    print()
    print("E2")
    # Task 2 - density_2
    task2_acc = np.zeros((total_obtain_answers, 8, 7, 2)).astype(np.int)
    task2_time = np.zeros((total_obtain_answers, 8, 7, 2)).astype(np.float)
    learning_effect_acc = np.zeros((total_obtain_answers, 8, 14)).astype(np.float)
    learning_effect_time = np.zeros((total_obtain_answers, 8, 14)).astype(np.float)
    task2_personal = np.zeros((total_obtain_answers)).astype(np.float)
    offset = 2 + 9 * 2 * 7 + 20

    for i in range(118, 118 + 112):
        for k in range(total_obtain_answers):
            ans = res[k][i]['problem_answer']['sort_index'][0]
            dataset_index = dataset_names.index(res[k][i]['problem_dataset'])
            sampling_index = sampling_methods.index(res[k][i]['problem_sampling'])
            level = res[k][i]['level']
            ans_index = dataset_index * 7 * 2 + sampling_index * 2 + level + offset
            standard = int(std[k][ans_index][-2])
            if res[k][i]['problem_timespend'] < 30000: # Filter invalid answers caused by network errors
                task2_time[k][dataset_index - 1][sampling_index][level] += res[k][i]['problem_timespend']
                learning_effect_time[k][dataset_index - 1][int((i - 118) / 8)] += res[k][i]['problem_timespend']
            assert std[k][ans_index].index(res[k][i]['problem_dataset'] + '-' + str(level) + ' ' + res[k][i]['problem_sampling']) == 0
            if ans == standard:
                task2_acc[k][dataset_index - 1][sampling_index][level] += 1
                learning_effect_acc[k][dataset_index - 1][int((i - 118) / 8)] += 1
                task2_personal[k] += 1

    task2_acc_avg = [int(np.sum(task2_acc[:,:,i,:])) / total_each_sampling for i in range(7)]
    task2_acc_p = np.array([[np.average(task2_acc[k,:,i,:]) for i in range(7)] for k in range(total_obtain_answers)])
    task2_acc_ci = [np.std(task2_acc_p[:,i]) * 1.96 / math.sqrt(total_obtain_answers) for i in range(7)]
    task2_time_avg = [int(np.average(task2_time[:,:,i,:])) for i in range(7)]
    task2_time_ci = [np.std(task2_time[:,:,i,:]) * 1.96 / math.sqrt(total_obtain_answers * 8 * 2) for i in range(7)]

    print("Task2 Accuracy:\n", task2_acc_avg)
    print("Task2 Accuracy 95\% CI:\n", task2_acc_ci)
    print("Task2 Completion Time:\n", task2_time_avg)
    print("Task2 Completion Time 95\% CI:\n", task2_time_ci)

    task2_acc_detail = [[np.average(task2_acc[:,i,k,j]) for i in range(8) for j in range(2)] for k in range(7)]
    task2_time_detail = [[np.average(task2_time[:,i,k,j]) for i in range(8) for j in range(2)] for k in range(7)]

    acc_fm_res = stats.friedmanchisquare(*task2_acc_detail)
    time_fm_res = stats.friedmanchisquare(*task2_time_detail)

    print("Friedman Test for Task2 Accuracy:\n", acc_fm_res)
    # print("Conover Test for Task2 Accuracy:\n", sp.posthoc_conover_friedman(np.array(task2_acc_detail).T))
    print("Friedman Test for Task2 Completion Time:\n", time_fm_res)
    print("Conover Test for Task2 Completion Time:\n", sp.posthoc_conover_friedman(np.array(task2_time_detail).T))

    print()
    print("E3")
    # Task 3 - outlier 1
    full_outlier_idx = {
        name: set()
    for name in dataset_names}
    with open("sampling_index/outlier.txt") as f:
        content = [line for line in f]
        for i in range(len(dataset_names)):
            name = content[3 * i][:-1]
            [x, y, w, h] = list(map(lambda s: float(s), content[3 * i + 1][:-1].split(" ")))
            outlier_indices = content[3 * i + 2][:-1].split(" ")[:-1]

            npz_path = os.path.join("./sampling_index/" + name + '.npz')
            all_data = np.load(npz_path)
            position = all_data["positions"]
            xmin, xmax = min(position[:, 0]), max(position[:, 0])
            position[:, 0] = (position[:, 0] - xmin) / (xmax - xmin)
            ymin, ymax = min(position[:, 1]), max(position[:, 1])
            position[:, 1] = (position[:, 1] - ymin) / (ymax - ymin)

            for idx in outlier_indices:
                ii = int(idx)
                if x <= position[ii][0] <= x + w and y <=position[ii][1] <= y + h:
                    full_outlier_idx[name].add(int(ii))
        f.close()
    
    partial_outlier_idx = [[set() for method in sampling_methods] for name in dataset_names[1:]]
    for i, name in enumerate(dataset_names[1:]):
        for j, method in enumerate(sampling_methods):
            datapath = "./sampling_index/" + name + "_" + sampling_method_class_names[j] + ".csv"
            selected_indexes = []
            with open(datapath, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if len(row) > 0:
                        selected_indexes.append(int(row[0]))
            for idx in selected_indexes:
                if idx in full_outlier_idx[name]:
                    partial_outlier_idx[i][j].add(idx)

    upper_bound_outlier = np.zeros(8).astype(np.float)
    for i, name in enumerate(dataset_names[1:]):
        upper_bound_outlier[i] = max([len(partial_outlier_idx[i][j]) for j in range(7)]) / len(full_outlier_idx[name])

    task3_prec = np.zeros((total_obtain_answers, 8, 7)).astype(np.float)
    task3_recl = np.zeros((total_obtain_answers, 8, 7)).astype(np.float)

    task3_correct_find = np.zeros((total_obtain_answers, 8, 7)).astype(np.int)
    task3_total_find = np.zeros((total_obtain_answers, 8, 7)).astype(np.int)

    for i in range(233, 233 + 56):
        for k in range(total_obtain_answers):
            ans = set(res[k][i]['problem_answer']['outliers'])
            dataset_index = dataset_names.index(res[k][i]['problem_dataset'])
            sampling_index = sampling_methods.index(res[k][i]['problem_sampling'])
            standard = partial_outlier_idx[dataset_index - 1][sampling_index]
            correct_idx = ans & standard
            precision = len(correct_idx) / len(ans) if len(ans) > 0 else 1
            recall = len(correct_idx) / len(full_outlier_idx[res[k][i]['problem_dataset']]) / upper_bound_outlier[dataset_index - 1]
            task3_prec[k][dataset_index - 1][sampling_index] = precision
            task3_recl[k][dataset_index - 1][sampling_index] = recall
            task3_correct_find[k][dataset_index - 1][sampling_index] = len(correct_idx)
            task3_total_find[k][dataset_index - 1][sampling_index] = len(ans)

    task3_prec_avg = [np.average(task3_prec[:,:,i])for i in range(7)]
    task3_recl_avg = [np.average(task3_recl[:,:,i])for i in range(7)]

    sampled_outlier_ratio = [[len(partial_outlier_idx[i][j]) / len(full_outlier_idx[name]) for j in range(7)] for i, name in enumerate(dataset_names[1:])] 

    print("Task3 Precision:\n", task3_prec_avg)
    print("Task3 Recall:\n", task3_recl_avg)

    task3_prec_ci = [np.std(task3_prec[:,:,i]) * 1.96 / math.sqrt(total_obtain_answers * 8) for i in range(7)]
    task3_recl_ci = [np.std(task3_recl[:,:,i]) * 1.96 / math.sqrt(total_obtain_answers * 8) for i in range(7)]

    print("Task3 Precision 95% CI:\n", task3_prec_ci)
    print("Task3 Recall 95% CI:\n", task3_recl_ci)


    task3_prec_detail = [[np.average(task3_prec[:,i,k]) for i in range(7)] for k in range(7)]
    task3_recl_detail = [[np.average(task3_recl[:,i,k]) for i in range(7)] for k in range(7)]

    prec_fm_res = stats.friedmanchisquare(*task3_prec_detail)
    recl_fm_res = stats.friedmanchisquare(*task3_recl_detail)

    print("Friedman Test for Task3 Precision:\n", prec_fm_res)
    # print("Conover Test for Task3 Precision:\n", sp.posthoc_conover_friedman(np.array(task3_prec_detail).T))
    print("Friedman Test for Task3 Recall:\n", recl_fm_res)
    print("Conover Test for Task3 Precision:\n", sp.posthoc_conover_friedman(np.array(task3_recl_detail).T))

    print()
    print("E4")
    # Task 4 - shape_1
    task4_points = np.zeros((total_obtain_answers, 8, 7)).astype(np.float)
    task4_points_dataset = np.zeros((total_obtain_answers, 8, 7)).astype(np.float)

    for i in range(problem_count - 8, problem_count):
        j = i - (problem_count - 8)
        for k in range(total_obtain_answers):
            dataset_index = dataset_names.index(res[k][i]['problem_dataset']) - 1
            ans = res[k][i]['problem_answer']['sort_index']
            rel = [int(idx) for idx in res[k][i]['problem_answer']['relation_type']]
            pos = 7
            total = 0
            count = 0
            curr = []
            while pos > 0:
                total += pos
                count += 1
                curr.append(ans[7 - pos])
                if ans[7 - pos] not in rel:
                    for idx in curr:
                        task4_points[k][j][idx] += (total / count)
                        task4_points_dataset[k][dataset_index][idx] += (total / count)
                    curr = []
                    total, count = 0, 0
                pos -= 1

    task4_points_avg = [np.average(task4_points[:,:,i]) for i in range(7)]
    print("Task4 Score:\n", task4_points_avg)

    task4_points_ci = [np.std(task4_points[:,:,i]) * 1.96 / math.sqrt(total_obtain_answers * 8) for i in range(7)]
    print("Task4 Score 95\% CI:\n", task4_points_ci)

    task4_points_avg_d = [[np.average(task4_points[:,j,i]) for i in range(7)] for j in range(8)]
    print("Task4 Score for each dataset:\n", task4_points_avg_d)

    task4_points_ci_d = [[np.std(task4_points[:,j,i]) * 1.96 / math.sqrt(total_obtain_answers) for i in range(7)] for j in range(8)]
    print("Task4 Score 95\% CI for each dataset:\n", task4_points_ci_d)
