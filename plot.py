import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from datetime import datetime

class entity():
    def __init__(self): 
        self.lot_info = [] 
        self.entity_name = ''
        self.part_no = ''
        self.part_id = ''
        self.bd_id = ''
        
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory) 

# dictionary find_index : entity_name ==> index in list entity
# dictionary lot_info : lot_number, start time, end_time

if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    ety = []
    ety_index = -1
    find_index = {}
    for line in f.readlines():
        tmp = line.split(', ')
        entity_name = tmp[4]
        ety_index = find_index.get(entity_name)
        if ety_index == None:
            ety_index = len(ety)
            find_index[entity_name] = len(ety)
            ety.append(entity())
        lot_info = {'lot_number' : tmp[0],
                    'start' : float(tmp[5]),
                    'end' : float(tmp[6])}
        ety[ety_index].entity_name = entity_name
        ety[ety_index].part_no = tmp[2]
        ety[ety_index].part_id = tmp[3]
        ety[ety_index].bd_id = tmp[1]
        ety[ety_index].lot_info.append(lot_info)
    
    foldername = './result' + datetime.now().strftime('-%Y%m%d%H%M') + '/' # folder which store result picture
    createFolder(foldername)
    
    #dictionary color_map : part_no/part_id ==> color
    color_barh = ['navy','m','y','c','coral', 'brown', 'orange', 'purple','gold','seagreen','darkslategray','olive','chocolate','deeppink','indigo','forestgreen','steelblue','violet','cornflowerblue','darkorange','darkgreen','dimgray','r','g','b']
    color_map = {}

    plot_num = 25 # how many entity in one plot
    etytmp = []
    plt.rcParams["figure.figsize"] = (20, 10)
    count_pic = 0
    for i in range(len(find_index)):
        # id = ety[i].part_no + '/' + ety[i].part_id
        id = ety[i].bd_id
        colorsave = color_map.get(id)
        if colorsave == None:
            color_map[id] = color_barh[i % plot_num]
            colorsave = color_barh[i % plot_num]

        for j in range(len(ety[i].lot_info)):
            end = ety[i].lot_info[j]['end']
            start = ety[i].lot_info[j]['start']
            plt.barh(i%plot_num, end-start,left=start, color=colorsave)
            plt.text(start,i%plot_num,'%s'%(ety[i].lot_info[j]['lot_number']),color="white")
        etytmp.append(ety[i].entity_name)
        if (i % plot_num == plot_num-1) or (i == len(find_index)-1):
            plt.yticks(np.arange(len(etytmp)),etytmp)
            patches = [mpatches.Patch(color="{:s}".format(value), label="{:s}".format(key) ) for key,value in color_map.items()]
            plt.xlabel("Time(min)")
            plt.ylabel("entity_name")
            plt.legend(handles=patches)
            plt.plot()
            count_pic = count_pic + 1
            picname = foldername + str(count_pic) + '.png'
            plt.savefig(picname)
            plt.clf()
            etytmp.clear()
            color_map.clear()
    f.close()
    
