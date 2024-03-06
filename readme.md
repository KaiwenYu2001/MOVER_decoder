# Extract waveforms from `.xml`

`./Extract_waveform`

## Instructions:

1. The scripts executing `SIS` and `EPIC` data are different, due to different data structure.
2. `BernoulliXML_Tools_Only.py ` is written by Joe Rinehart, MD, 
    I kept as most code originally as I can.
3. `Example_Of_Execution.py` is also written by MD Rinehart. 
    I made quite a lot modification here and this is 
    the main script that do data extraction.
4. `Help_Function.py` provides function that can check data
    in xml files to see is there is actual data in it.
5. `main.py` can handle all xml files in a single directory,
    for example, it can process all patient data for patientID
    begin with '01'.
6. `multi_patient.py` processes entire patient database.

## Steps to run scripts:

1. Install the dependencies: `pip install -r requirements.txt`

2. Before running them, change the directory in the beginning lines of scripts:

  ```python
  '''
  1. Before executing, all files in the output directory will be removed. So double check the OUTPUT_DIR before you run it. 
  2. Do not contain `/.` in your directory path. 
  '''
  USE_MULTITHREADING # if you want to apply multithreading to speed up
  MAX_POOL_SIZE # pool size depends on your CPU capability, usually equals to the number of cores in computer
  INPUT_DIR # the directory that stores all xml files
  OUTPUT_DIR # the directory that you want to store your output.
  ```

3. Run `main.py` or `multi_patient.py`
4. If the script returns with error,  check the dependences and try install dependences manually.

## Contributors

- Joseph Rinehart, MD
- Kaiwen Yu
- Mirana C. Angel
