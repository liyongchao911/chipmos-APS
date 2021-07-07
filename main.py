import json
import os
import re
import pandas as pd
from datetime import datetime


def filePreprocessing(path, files:dict, csv_config):

    for file in files:
        file_path = os.path.join(path, file["filename"])
        file_function = file["function"]

        if(not file_function in csv_config):
            csv_config[file_function] = []
        try:

            if(re.search(r'.csv', file_path) != None):
                df = pd.read_csv(file_path, dtype=str)
                file_path = file_path.replace(r'.csv', '')
            elif(re.search(r'.xlsx', file_path) != None):
                df = pd.read_excel(file_path, dtype=str)
                file_path = file_path.replace(r'.xlsx', '')
        except ValueError as e:
            print(e)
            print(file_path)
            exit(-1)


        df = df.rename(columns=lambda x : x.strip())

        if "columns" in file:
            df = df[file["columns"]]
            for col in file["columns"]:
                df[col] = df[col].str.strip()

        df.to_csv(file_path + ".csv", index=False)
        csv_config[file_function].append(file_path)
    return csv_config

def preprocessing(conf:dict):
    csv_config = {}
    path = conf["file_path"]
    csv_config_file_path = os.path.join(path, "config.csv")
    csv_config = filePreprocessing(path, conf["preprocess_files"], csv_config)
    csv_config = filePreprocessing(path, conf["nopreprocess_files"], csv_config)

    wip_filename = csv_config["wip"][0]
    time = re.split(r"_|\.csv", wip_filename)[1]
    dt = datetime.strptime(time, "%Y%m%d%H%M%S")

    csv_config["std_time"] = [ dt.strftime("%Y/%m/%d %H:%M")]
    df = pd.DataFrame(csv_config)
    df.to_csv(csv_config_file_path, index=False)
    # print(json.dumps(csv_config, indent=4))

    return csv_config_file_path


if __name__ == '__main__':
    if(len(os.sys.argv) < 2):
        print("Please specify the config file.")
        exit(-1)
    config = os.sys.argv[1]
    csv_config_file_path = preprocessing(json.load(open(config)))
