# ChipMOS WB station

This is a software used to solve scheduling problem in ChipMOS Co. The software is mainly developed by **C++** and includes the raw data processing and the algorithm. The algorithm used in the program is the genetic algorithm. Now the software is used in the factory everyday.

The data processing in the programs has 2 stages. One focuses on the valid data and the other focuses on data binding like the route, the tooling, wire and something about the lot in factory. The program originally was designed to use GPU and was suppose to be developed on GPU, but the program needs to finish the new specification from the operators in factory **every week**. So, it is efficient to use CPU to develop the stable version and then if the time and the money is sufficient, it would be possible to be accelerated by using GPU.


## Setup virtual environment

```shell=
python3 -m venv venv
source venv/bin/activate # for Unix-like
python3 -m pip install -r requirements.txt
```

## Compile c++ program

```shell=
cmake -B build -S . (in Unix-like)
cmake -B build -S . -G "MinGW Makefiles" (in Windows)
cmake --build build
```


## Execute

* Step 1 : Edit the files' path in `config.json`
* Step 2 : Ensure the `config.json` file, `main.py` and `main` are in the same directory
* Step 3 : Data preprocessing to generate the `config.csv` file.
```shell=
python3 main.py config.json
```
* Step 4: Execute
```shell=
./main [-f=/--file=][config.csv] [-p] [-r]
```

## Testing

Don't forget to clone the testing data repo and put into the build. The directory tree should look like
```
.
├── Makefile
├── main
├── test
├── test_data
    ├── cure_time.csv
    ├── ent_limit.csv
    ├── process_find_lot_size_and_entity.csv
    ├── queuetime.csv
    ├── route_list.csv
    ├── wrong_cure_time
    │   ├── cure_time0.csv
    │   ├── cure_time1.csv
    │   ├── cure_time2.csv
    │   ├── cure_time3.csv
    │   ├── cure_time4.csv
    │   ├── cure_time5.csv
    │   ├── cure_time6.csv
    │   ├── cure_time7.csv
    │   ├── cure_time8.csv
    │   └── cure_time9.csv
    └── wrong_queue_time
        ├── queuetime0.csv
        ├── queuetime1.csv
        ├── queuetime2.csv
        ├── queuetime3.csv
        ├── queuetime4.csv
        ├── queuetime5.csv
        └── queuetime6.csv
```
