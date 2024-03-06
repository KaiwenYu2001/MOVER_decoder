import os
import math
import Help_Function as HF
import BernoulliXML_Tools_Only as BT
import sys
import matplotlib.pyplot as plt
import csv


def Execute(patient_dir: str, patient_id: str, output_dir: str, debug=False):
    '''
    Execute an single patient in the input_dir.
    '''
    # =================================================================
    input_dir = patient_dir

    # Before execution, check the type of waveforms(IP, CB) and then identify if there does exist data.
    keep = []
    IP_type = True
    Found_data = False
    for IP_type in [True, False]:
        flist_tmp = BT.Find_XML_in_DIR(input_dir, "sampleID", "2025-01-10", debug=debug, IP=IP_type)
        flist = []
        for x in flist_tmp:
            if x.split('/')[-1].split('-')[0] == patient_id:
                flist.append(x)
        if HF.check_data_in_filelist(flist):
            Found_data = True
            break

    if Found_data == False:
        raise ValueError("Found no data for patient.")

    # Execution and print length and frequency of waves.
    Wave = BT.load_xml_into_wave_assembly(flist)
    if debug:
        print(f"\tcb_inv: {len(Wave.cb_inv)}, art: {len(Wave.art)}, cb_ihz: {Wave.cb_ihz}")
        print(f"\tcb_ecg: {len(Wave.cb_ecg)}, ecg: {len(Wave.ecg)}, cb_ehz: {Wave.cb_ehz}")
        print(f"\tcb_pleth: {len(Wave.cb_pleth)}")

    waveInfoLib = {"cb_inv": len(Wave.cb_inv), "art": len(Wave.art), "cb_ecg": len(Wave.cb_ecg), \
                        "ecg": len(Wave.ecg), "cb_pleth": len(Wave.cb_pleth)}
    flag = 0
    for waveType in waveInfoLib:
        if waveInfoLib[waveType] != 0:
            flag += 1
            if waveType == "cb_inv":
                WaveTarg = Wave.cb_inv
            elif waveType == "art":
                WaveTarg = Wave.art
            elif waveType == "cb_ecg":
                WaveTarg = Wave.cb_ecg
            elif waveType == "ecg":
                WaveTarg = Wave.ecg
            elif waveType == "cb_pleth":
                WaveTarg = Wave.cb_pleth

            # Now find the segments I need
            keep = []
            # Now find the segments I need
            for j in range(len(WaveTarg)):
                keep.append(WaveTarg[j])

            (Fwave, true_hz, windex, wave_start_time, wave_end_time, _proc_notes) = BT.build_wave(keep, 100, 1)
            # IP_type = "IP" if IP_type else "CB"
            output_filename = os.path.join(output_dir, f"{patient_id}.csv")
            if flag == 1: # This means this is the first time of creating csv.
                with open(output_filename, 'w', newline='', encoding="utf8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['type', 'start_time', 'end_time', 'true_hz_rate', 'datapoints'])
                    writer.writerow(
                        [waveType, wave_start_time.timestamp(), wave_end_time.timestamp(), true_hz, '|'.join(map(str, Fwave))])
            else: # if not, just append a row to it. 
                with open(output_filename, 'a+', newline='', encoding="utf8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [waveType, wave_start_time.timestamp(), wave_end_time.timestamp(), true_hz, '|'.join(map(str, Fwave))])
    if not flag:
        raise ValueError("Found no waveform for patient.")


def Plot(Fwave):
    # plot the waveforms with pyplotlib
    plt.figure()
    sub1 = plt.subplot(1, 2, 1)
    sub2 = plt.subplot(1, 2, 2)
    sub1.plot(Fwave)
    wavelen = len(Fwave)
    sub2.plot(Fwave[math.ceil(wavelen / 2): math.ceil(wavelen / 2) + 300])
    # plt.ylim(-1000, 1000)
    plt.show()


if __name__ == '__main__':
    patient_id = "0c0b5463848f9505IP"
    input_dir = "/Users/kevinyu/Desktop/test_env/test_case/epic_test"
    output_dir = "/Users/kevinyu/Desktop/test_env/output"
    Execute(patient_id, input_dir, output_dir)
