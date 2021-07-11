import plotly.express as px
import pandas as pd
import datetime
import time
import os 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory) 

foldername = './result' + datetime.datetime.now().strftime('-%Y%m%d%H%M') + '/' # folder which store result picture
createFolder(foldername)  

# 
result = pd.read_csv("result(0707_2).csv") #//改
entities= result.groupby("entity") # group by entity
entity_key_list= list(entities.groups.keys()) # key list

# 2021/4/17 08:30 (change to var)
ini_time_stamp = 1618619400 #//改
plot_num = 50 #each time plot 50


df=[]
count=0
for i in range(len(entity_key_list)): 
    key= entity_key_list[i]
    count+=1
    for j in range(len(entities.groups[key])):
        row_index=entities.groups[key][j]

        Lot_id =result.iloc[row_index].at['lot_number'].strip() # remove space
        Entity =result.iloc[row_index].at['entity'].strip()
        start_ts =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result.iloc[row_index].at['start_time']*60)) 
        end_ts =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result.iloc[row_index].at['end_time']*60)) 
        Sets = result.iloc[row_index].at['part_no'].strip()+ "/"+ result.iloc[row_index].at['part_id'].strip()
        Bd_id= result.iloc[row_index].at['bd_id'].strip()
    

        df.append(dict(Lot_ID= Lot_id, Start=start_ts,Finish=end_ts,Entity=Entity,Sets=Sets,Bd_Id=Bd_id))
    if(count==plot_num) or i==len(entity_key_list)-1: # consider final part
       #show figure
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Entity", color="Bd_Id",text="Lot_ID",hover_name="Sets")   # hover_name
        fig.update_yaxes(categoryorder="category descending") # sort by
        fig.update_traces(textposition='inside',marker_line_color='rgb(8,48,107)')
        fig.write_html(foldername +df[0]['Entity']+"_" +df[-1]['Entity']+".html")
        df=[]
        count=0
 