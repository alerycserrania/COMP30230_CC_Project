import os
import multiprocessing


datafiles = [
    'examples.data', 'examples2.data',  'examples3.data'
]

mlp_type = [
    'r', 'r', 'c'
]

learning_rates = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
]

ratios_trs_tests = [
    0.5,
    0.8,
    0.8,
]

nb_hidden_units = [
    [2, 3, 4, 5, 8, 10, 15, 20],
    [4, 8, 10, 15, 20, 25, 40],
    [10, 20, 40, 75, 100],
]

activations = [
    ['sigmoid', 'sigmoid'],
    ['sigmoid', 'linear'],
    ['sigmoid', 'sigmoid'],
]

max_epochs = [
    [100, 200, 500, 1000],
    [100, 200, 500, 1000],
    [100, 200, 500, 1000],
]

update_periods = [
    [1, 2, 3, 4],
    [1, 2, 4, 8, 12],
    [2, 4, 8, 16, 32],
]


def run_parallel(lr, i):
    datafile = datafiles[i]
    total = len(nb_hidden_units[i]) * len(max_epochs[i]) * len(update_periods[i])
    count = 0
    for nb_hidden_unit in nb_hidden_units[i]:
        for max_epoch in max_epochs[i]:
            for update_period in update_periods[i]:
                filename = f'{ratios_trs_tests[i]}_{nb_hidden_unit}' \
                           f'_{lr}_{max_epoch}_{update_period}'
                cmd = f'cmake-build-release\\ccproject.exe ' \
                      f'{datafile} {mlp_type[i]} {ratios_trs_tests[i]} ' \
                      f'{nb_hidden_unit} {activations[i][0]} {activations[i][1]} ' \
                      f'{lr} {max_epoch} {update_period} ' \
                      f'> outdata\\{datafile}\\{filename}'
                os.system(cmd)
                count += 1
                print(f'{datafiles[i]} {lr}: {100 * count / total}%')


if __name__ == '__main__':
    processes = []
    for i in range(len(datafiles)):
        for learning_rate in learning_rates:
            p = multiprocessing.Process(
                target=run_parallel,
                args=(learning_rate, i, )
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()


