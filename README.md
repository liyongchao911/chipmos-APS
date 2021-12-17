# ChipMos WB station

## Setup virtual environment

```shell=
python3 -m venv venv
source venv/bin/activate # for Unix-like
python3 -m pip install -r requirements.txt
```

## Compile c++ program

```shell=
mkdir build
cd build/
cmake .. (in Unix)
cmake -G "MinGW Makefiles" .. (in Windows)
make
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


## TODO

### Testing

- _chooseMachinesForAGroup
- _initializeNumberOfExpectedMachines
- _setupContainerForMachines
- _setupResources
- _loadResource
- _loadResourcesOnTheMachine
- prepareMachines
- _linkMachineToAJob
- prepareJobs
