import csv
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os 


def plot_gantt_by_order(result_df):
    # 1. Group the result.csv by entity 
    # 2. Use group key(entity) to find which lot processing on this entity, and plot each entity's lots by order.
    # 3. Using plotly.express.timeline to plot gantt, which need the information of below data.(need using time stamp to convert date )
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory) 

    # initizlize
    plot_num = 50 #each time plot 50 entities
    entities= result_df.groupby("entity") # group by entity
    entity_key_list= list(entities.groups.keys()) # key list
    color_discrete_sequence_list=['#F0F8FF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4',    # add color cycle
                '#000000','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0',
                '#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC','#DC143C','#00FFFF',
                '#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B',
                '#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B',
                '#2F4F4F','#00CED1','#9400D3','#FF1493','#00BFFF','#696969','#1E90FF',
                '#B22222','#FFFAF0','#228B22','#FF00FF','#DCDCDC','#F8F8FF','#FFD700',
                '#DAA520','#808080','#008000','#ADFF2F','#F0FFF0','#FF69B4','#CD5C5C',
                '#4B0082','#FFFFF0','#F0E68C','#E6E6FA','#FFF0F5','#7CFC00','#FFFACD',
                '#ADD8E6','#F08080','#E0FFFF','#FAFAD2','#90EE90','#D3D3D3','#FFB6C1',
                '#FFA07A','#20B2AA','#87CEFA','#778899','#B0C4DE','#FFFFE0','#00FF00',
                '#32CD32','#FAF0E6','#FF00FF','#800000','#66CDAA','#0000CD','#BA55D3',
                '#9370DB','#3CB371','#7B68EE','#00FA9A','#48D1CC','#C71585','#191970',
                '#F5FFFA','#FFE4E1','#FFE4B5','#FFDEAD','#000080','#FDF5E6','#808000',
                '#6B8E23','#FFA500','#FF4500','#DA70D6','#EEE8AA','#98FB98','#AFEEEE',
                '#DB7093','#FFEFD5','#FFDAB9','#CD853F','#FFC0CB','#DDA0DD','#B0E0E6',
                '#800080','#FF0000','#BC8F8F','#4169E1','#8B4513','#FA8072','#FAA460',
                '#2E8B57','#FFF5EE','#A0522D','#C0C0C0','#87CEEB','#6A5ACD','#708090',
                '#FFFAFA','#00FF7F','#4682B4','#D2B48C','#008080','#D8BFD8','#FF6347',
                '#40E0D0','#EE82EE','#F5DEB3','#FFFFFF','#F5F5F5','#FFFF00','#9ACD32']  
    df=[] #plot need dataframe
    plt_count=0 # record the times current plot
    foldername = './output/plot_by_order' + '/' # folder which store result picture
    createFolder(foldername)  
    #plot
    for i in range(len(entity_key_list)): 
        key= entity_key_list[i]
        plt_count+=1
        for j in range(len(entities.groups[key])):
            row_index=entities.groups[key][j]
            Lot_id =result_df.iloc[row_index].at['lot_number'].strip() # remove space
            Entity =result_df.iloc[row_index].at['entity'].strip()
            start_ts =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result_df.iloc[row_index].at['start_time']*60)) 
            end_ts =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result_df.iloc[row_index].at['end_time']*60)) 
            Sets = result_df.iloc[row_index].at['part_no'].strip()+ "/"+ result_df.iloc[row_index].at['part_id'].strip()
            Bd_id= result_df.iloc[row_index].at['bd_id'].strip()
            # create plot needed dateframe
            df.append(dict(Lot_ID= Lot_id, Start=start_ts,Finish=end_ts,Entity=Entity,Sets=Sets,Bd_Id=Bd_id))
        if(plt_count==plot_num) or i==len(entity_key_list)-1: # consider final part
        #show figure
            fig = px.timeline(df, x_start="Start", x_end="Finish", y="Entity", color="Bd_Id",color_discrete_sequence=color_discrete_sequence_list,text="Lot_ID",hover_name="Sets")   # hover_name
            fig.update_yaxes(categoryorder="category descending") # sort by
            fig.update_traces(textposition='inside',marker_line_color='rgb(8,48,107)')
            fig.write_html(foldername +df[0]['Entity']+"_" +df[-1]['Entity']+".html")
            # reset
            df=[]
            plt_count=0

    pass

def output_simulation(result_df,machine_area_df): 

    #  Get the filetered dataframe by judging whether proceesing through specific time
    def get_result_byFilter(merge_result,relative_time):
        filter_strt = merge_result["start_time"] <= relative_time  # retuen true or false of each record on dataframe
        filter_end = merge_result["end_time"] >= relative_time
        filter_result = merge_result[filter_strt & filter_end]     # if satisfying both condition, it will get this record(through all)  

        return filter_result

    # Get entity amount(dataframe) on specific time
    # input: filetered dataframe & new column name
    # use two times group by to get amount
    def get_ENT_df(merge_result,ent_colname): 

        temp_group= merge_result.groupby(['cust','pin_pkg','prod_id','bd_id','oper','Location','entity']) # to know entity amount
        df_grp_entity = pd.DataFrame(temp_group.groups.keys())
        grp_basis = df_grp_entity.groupby([0,1,2,3,4,5])  # from cust to Location(index=0~5) #second groupby to build new dataframe format
        df_keys= pd.DataFrame(grp_basis.groups.keys())
        df_Ent_amount= pd.DataFrame(list(grp_basis.size()))
        ENTdf= pd.concat([df_keys, df_Ent_amount], axis=1)
        ENTdf.columns=['cust','pin_pkg','prod_id','bd_id','oper','Location',ent_colname]  


        return ENTdf
    
    #Use group by to get the amount of quantity of each sets and make it to dataframe(list>df)
    #Find the lots using same group by set and accumulate their die quantity
    def get_qty_df(merge_result):

        qty_list=[]
        original_group=merge_result.groupby(['cust','pin_pkg','prod_id','bd_id','oper','Location']) 
        group_key_list= list(original_group.groups.keys())
        
        for i in range(len(group_key_list)): 
            key= group_key_list[i]
            qty_sum=0
            for j in range(len(original_group.groups[key])):
                row_index=original_group.groups[key][j]
                qty_sum+=merge_result.iloc[row_index].at['qty']
            qty_list.append(qty_sum)

        df_keys= pd.DataFrame(original_group.groups.keys())
        qty_df = pd.DataFrame(qty_list)
        outplan_df= pd.concat([df_keys, qty_df], axis=1)
        outplan_df.columns=['cust','pin_pkg','prod_id','bd_id','oper','Location','Output plan'] 

        return outplan_df

    # initialize
    merge_result = pd.merge(result_df, machine_area_df, left_on ='entity', right_on ='Entity', how='left') 
    oneday_result = pd.merge(result_df[result_df["end_time"]<= 1440], machine_area_df, left_on ='entity', right_on ='Entity', how='left') #(total 27.5hrs)

    grp_basis = merge_result.groupby(['cust','pin_pkg','prod_id','bd_id','oper','Location']) 
    df_frame= pd.DataFrame(grp_basis.groups.keys())
    df_frame.columns=['cust','pin_pkg','prod_id','bd_id','oper','Location']  

    ENT_1 = get_ENT_df(get_result_byFilter(merge_result,0),'Original ENT(+0)')      #relative to 07:30(standard time) is 0 minute
    ENT_2 = get_ENT_df(get_result_byFilter(merge_result,210),'Original ENT(+210)')    #relative to 11:00 is need to plus 210 minute
    ENT_3 = get_ENT_df(get_result_byFilter(merge_result,1650),'Allocate ENT')
    outplan_df = get_qty_df(oneday_result)                                          # one day

    # merge and concat all data to one dataframe
    merge_df1 = pd.merge(df_frame, ENT_1, on =(['cust','pin_pkg','prod_id','bd_id','oper','Location']), how ='left')
    merge_df2 = pd.merge(merge_df1, ENT_2, on =(['cust','pin_pkg','prod_id','bd_id','oper','Location']), how ='left')
    merge_df3 = pd.merge(merge_df2, ENT_3, on =(['cust','pin_pkg','prod_id','bd_id','oper','Location']), how ='left')
    simulation_output_df = pd.merge(merge_df3, outplan_df, on =(['cust','pin_pkg','prod_id','bd_id','oper','Location']), how ='left')
    
    # write to csv
    simulation_output_df.to_csv("output/simulation_output.csv", index=False ,na_rep=0) 
    
    pass

def output_setup_record(result_df):
    sorted_result_df=result_df[result_df["start_time"]<= 1440].sort_values(by=['entity', 'start_time']) #within one day
    setup_dict ={}
    times_info_list=[]
    entity_name= sorted_result_df['entity'].iloc[0] # default
    bd_id = sorted_result_df['bd_id'].iloc[0]       # default
    total_setup_times=0

    for i in range(len(sorted_result_df.index)): # serach for all data
        if entity_name == sorted_result_df['entity'].iloc[i]:  # If entity is same, need to judge whether bd_id is same. Else, record new entity(key)

            if bd_id != sorted_result_df['bd_id'].iloc[i]: # if bd_id is different need to record
                setup_start=time.strftime("%m-%d %H:%M:%S", time.localtime(ini_time_stamp+sorted_result_df['end_time'].iloc[i-1]*60))  #the endT of previous and convert to date
                setup_end=time.strftime("%m-%d %H:%M:%S", time.localtime(ini_time_stamp+sorted_result_df['start_time'].iloc[i]*60))   #the startT of current and convert to date
                # change to date
                times_info_list.append(setup_start+' > '+setup_end) 
                total_setup_times +=1
                bd_id = sorted_result_df['bd_id'].iloc[i]

        else: 
            #add to dict
            setup_dict[entity_name]=times_info_list

            #next entitys
            entity_name = sorted_result_df['entity'].iloc[i]
            bd_id = sorted_result_df['bd_id'].iloc[i]
            times_info_list=[]

        #final one
        if i ==len(sorted_result_df.index)-1: 
            setup_dict[entity_name]=times_info_list

    # write file
    with open("output/setup_record_0901.csv", "w", newline="") as f_output:
        csv_output = csv.writer(f_output)
        csv_output.writerow(['一日改機次數',total_setup_times])
        for key in sorted(setup_dict.keys()):
            csv_output.writerow([key] + setup_dict[key])

    pass

def output_result_to_date(result_df):
    # 1. Convert relative time to absulute time(date) (start time and end time)
    # 2. Change the column order
    new_result_df = result_df.copy(deep=True)
    for i in range(len(result_df.index)):
        new_result_df['start_time'].iloc[i] =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result_df['start_time'].iloc[i]*60))
        new_result_df['end_time'].iloc[i] =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ini_time_stamp+result_df['end_time'].iloc[i]*60)) 
    new_result_df= new_result_df[['lot_number','cust','pin_pkg','prod_id','qty','part_id','part_no','bd_id','entity','start_time','end_time']]

    # write file 
    new_result_df.to_csv("output/new_result.csv", index=False ,na_rep=0)

    pass

# read file and initialize
result_df = pd.read_csv("output/result.csv")
machine_area_df = pd.read_csv("input/locations.csv" )[['Entity','Location']]
config_df = pd.read_csv("input/config.csv")
ini_time_stamp = int(time.mktime(time.strptime(config_df.at[0,'std_time'], "%Y/%m/%d %H:%M"))) # get standard time

# execute
plot_gantt_by_order(result_df)
output_simulation(result_df,machine_area_df)
output_setup_record(result_df)
output_result_to_date(result_df)