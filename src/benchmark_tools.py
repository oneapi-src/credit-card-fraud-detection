import os
import json
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

def init_json(jsonfile):
    dict_list = {}
    os.makedirs(os.path.dirname(jsonfile), exist_ok=True)

    if os.path.exists(jsonfile):
        os.remove(jsonfile)
        
    with open(jsonfile, 'w+', encoding='utf-8') as jf:
            json.dump(dict_list, jf)


def parse_logs(logfile, jsonfile):
    json_dict = {}
    logname = logfile.split('_')[1]
    with open(logfile, 'r', encoding='utf-8') as log:
        for line in log:
            if 'time' in line:
                test = line.split(' ', 1)[1].split('=')
                
                if ')' in test[1]:
                    test[1] = test[1].split()[1][:-1]
                
                json_dict[test[0]] = float(test[1])
                #json.dump(dict_list, jf)
                
    with open(jsonfile,'r') as file:
        json_data = json.load(file)

    json_data[logname] = json_dict

    with open(jsonfile,'w') as file:
        json.dump(json_data, file, indent = 4)
    

def get_json(json_file):
    with open(json_file, "r") as read_file:
        json_data = json.load(read_file)

    return json_data

def plot_log_times(stock_json, intel_json):

    if not os.path.exists(stock_json):
        print("File not found: {}".format(stock_json))
        return

    if not os.path.exists(intel_json):
        print("File not found: {}".format(intel_json))
        return

    j_stock = get_json(stock_json)
    j_intel = get_json(intel_json)

    stock_keys = j_stock.keys()
    intel_keys = j_intel.keys()

    no_pairs = list(set(stock_keys).symmetric_difference(set(intel_keys)))
    pairs = list(set(stock_keys) & set(intel_keys))

    missing = []
    for elem in no_pairs:
        if elem in intel_keys:
            missing.append("stock_"+elem)
        elif elem in stock_keys:
            missing.append("intel_"+elem)

    if missing != []:
        print("Can't find pair for {} and won't be plotted.".format(missing))

    bar_width = 0.35
    opacity = 0.8

    for elem in pairs:

        keys = j_intel[elem].keys()
        index = np.arange(len(keys))

        if len(keys) > 5:
            divide = 1
        else:
            divide = 0

        fig = plt.figure()
        #ax = fig.add_axes([0,0,1,1])

        stock_data = []
        intel_data = []
        for key in keys:
            stock_data.append(1)
            intel_data.append(j_stock[elem][key]/j_intel[elem][key])
            
        if divide == 0:
            rects1 = plt.bar(index, stock_data, bar_width, alpha=opacity, color='b', label='Stock')
            rects2 = plt.bar(index + bar_width, intel_data, bar_width, alpha=opacity, color='c', label='Intel')

            # Add counts above the two bar graphs
            for rect, value in zip(rects1 + rects2, stock_data + intel_data):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{value:.2f}X', ha='center', va='bottom', weight='bold')
                
            #plt.xlabel('Person')
            plt.ylabel('Relative Speedup to Stock\n Higher is Better')
            plt.title(elem)
            labels = ['\n'.join(wrap(l,24)) for l in keys]
            plt.xticks(index + bar_width/2, labels)
            plt.legend()
            
        else:
            _, axs = plt.subplots(2,1, figsize=(10, 10))

            for i in [0,1]:
                rects1 = axs[i].bar(index[i::2], stock_data[i::2], bar_width, alpha=opacity, color='b', label='Stock')
                rects2 = axs[i].bar(index[i::2] + bar_width, intel_data[i::2], bar_width, alpha=opacity, color='c', label='Intel')


                for rect, value in zip(rects1 + rects2, stock_data[i::2] + intel_data[i::2]):
                    height = rect.get_height()
                    axs[i].text(rect.get_x() + rect.get_width() / 2.0, height, f'{value:.2f}X', ha='center', va='bottom', weight='bold')

                axs[i].set_ylabel('Relative Speedup to Stock\n Higher is Better')
                #axs[i].title(elem)
                labels = ['\n'.join(wrap(l,24)) for l in list(keys)[i::2]]
                axs[i].set_xticks(index[i::2] + bar_width/2, labels)
                axs[i].legend()
                plt.suptitle(elem)

        plt.tight_layout()
        plt.show()