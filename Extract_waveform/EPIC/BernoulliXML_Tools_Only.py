# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:18:27 2019
Last edits Sat Mar 03
@author: Joe Rinehart

        This program will do several things on execution:

        Convert Bernoulli/CPC XML waveform files into binary numpy files
        (basically binary Int16 files). The files are saved as .npy but they
        can be read by any binary reader as Int16 encoding. The output files
        are ONLY the waveform data (no time data).

        Each wave file is also cleaned up during conversion: 'Noise' before or
        after the actual waveform (generally identifiable by lack of data in
        appropriate value ranges) is stripped out, and waveforms that are
        pre-determined to be entirely non-patient data are ignored and not
        converted.  In general this saves around 20-40% add'l disk space in the
        final saved binary files and reduces future processing time on clearly
        bad data.  Missing time gaps (the 33 sequence-offset 20-25 sec gaps)
        are filled in with blanks/silence to ensure times line up correctly.

        Cases that go over midnight are automatically sequenced correctly
        during reassembly (since the XML files end up out of order in these
        cases).

    This module is fully coded to handle 3 waveforms from the IP directories
    (EKG, Art, Pleth), as well as 7 CB waves (EKG, INV, PLT, RES, AWP, FLO,
    and CO2). After scanning dozens of patients, however, there doesn't seem
    to ever be any data in the IP-ART or IP-PLETH, or in RES, AWP, FLO, or CO2,
    so all of these have been deactivated in the code for time efficiency.

All you should need to do is set the directories immediately below to the
appropriate targets on your system and run the script

"""

# =============================================================================

_wave_output_root = "U:/EWbinwaves/"
_XML_root_dir = "U:/emr/"
_master_output_dir = "/Users/kevinyu/Desktop/UCIResearch/june12/WorkingExample"

# =============================================================================

import sys
import os
import datetime
import base64
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import gc
import math

# Module Variables
_logfile = (_master_output_dir + "XML_to_Bin_" +
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".log")


# Convenience class pulled from StackOverflow
def set_bit(v, index, x):
    """
        Set the index:th bit of v to 1 if x is truthy,
        else to 0, and return the new value.
    """
    mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
    v &= ~mask  # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask  # If x was True, set the bit indicated by the mask.
    return v  # Return the result, we're done.


# Class that holds a single "measurement" unit of a wave from the XML
# (conveience class to hold sequence IDs & time with the data)
# timestamp = python datetime
# cpc_sequence, device_sequence = int
# wave = list of binary waveform values
class wave_chunk:
    def __init__(self, timestamp, cpc_sequence, device_sequence, wave, points):
        self.cpc_sequence = int(cpc_sequence)
        self.device_sequence = int(device_sequence)
        self.wave = wave
        self.timestamp = timestamp
        self.points = int(points)


# Class to hold all the wave_chunks for the whole wave prior to assembly
# Restructure the waves to hold their own start timestamp
# pull the methods that follow into the assembly, too!
class wave_assembly:
    def __init__(self, ptid):
        if (ptid == "None"):
            raise Exception("ERROR: 'NONE' PTID PASSED")
        self.ptid = ptid
        self.pdate = ''

        # Each of the following are lists of wave_chunk objects
        self.ecg = []
        self.art = []
        # self.pleth = []
        self.cb_ecg = []
        self.cb_inv = []
        self.cb_pleth = []
        # self.cb_co2 = []
        # self.cb_awp = []
        # self.cb_flow = []
        # self.cb_resp = []

        # hz rate of the captured waves
        self.ecghz = 0
        self.arthz = 0
        # self.plethz = 0
        self.cb_ehz = 0
        self.cb_ihz = 0
        self.cb_phz = 0
        # self.cb_chz = 0
        # self.cb_ahz = 0
        # self.cb_fhz = 0
        # self.cb_rhz = 0
        self.ip_code = ''
        self.cb_code = ''


def insert_wave_chunk_by_time(chunk_array, new_chunk, next_insert):
    """
        Since they are saved only by case time in 24 hour format,
        the xml files might not be in the right order if the case went longer
        than 24 hours.  I'm assuming there are other ways it can be wrong.
        Rather than append, we'll put the chunks in correct chronological
        order as we add them to the array.

        All we need to do is start at the beginning and check to be sure the
        new_chunk is < existing. So long as all chunks get added in this
        manner they'll end up in sequence

        Since we assume an entire XML file is always going to be in sequence
        for itself, we can optimize by returning the last insert index when a
        chunk is added, then cycle back and auto-append the rest of the
        current XML at the same location until a new data file is opened
        in the directory being read.  This reduces searches from 1 per chunk
        to 1 per XML file. Nice.

        Unfortunately, the assumption that the XML is in order WITHIN a file
        proved wrong - see fxn below. Nevertheless, this function is now fast
        and gets the majority of the work done, so we'll keep it as is rather
        than forcing each chunk to be inserted by timestamp individually which
        adds a LOT of time to the process.

        When calling the function over a single XML file, the function returns
        'next_insert'. Pass this value back IN to next_insert EXCEPT when
        a new XML file is loaded; in that case pass back in 0:

            next_insert == 0: Must search for chunk position
            next_insert ==-1: Okay to append at end
            next_insert >  0: Okay to append at that position

    """

    # Append if FIRST chunk or if next_insert is == -1 (the "end" in python)
    if len(chunk_array) == 0 or next_insert == -1:
        chunk_array.append(new_chunk)
        return -1

    # If next_insert is > 0, we insert there
    if next_insert > 0:
        chunk_array.insert(next_insert, new_chunk)
        next_insert += 1
        return next_insert

    # must search for position
    for i in range(len(chunk_array)):
        if (new_chunk.timestamp < chunk_array[i].timestamp):
            chunk_array.insert(i, new_chunk)
            return i + 1

    # We didn't find a later chunk, so this new chunk goes at the end
    chunk_array.append(new_chunk)
    return -1


""" loads XML files from dir into wave_holder (a 'wave_assembly' object)"""


def Find_XML_in_DIR(dirname, patient_ID, patient_date, debug=0, mask="", IP=False):
    ptid = patient_ID
    pdate = patient_date
    flist = []

    # Create the waveholder
    wave_holder = wave_assembly(ptid)
    wave_holder.pdate = pdate

    if debug and 0:
        print("\tProcessing XML for PID " + ptid + " on " + pdate)

    # Scan the directories and find the subfiles
    for fname in os.listdir(dirname):
        if debug and 0:
            print(("check:", fname, IP))
        # We can be very specific about what we're looking for
        if (not '000Z' in fname) and (IP == True):
            flist.append(dirname + "/" + fname)
        elif ('000Z' in fname) and (IP == False):
            flist.append(dirname + "/" + fname)
    if (debug):
        print(("\t" + str(len(flist)) + " total xml files found"))

    return flist


def load_xml_into_wave_assembly(flist, debug=False):
    if debug and 0:
        print(flist)

    # Open each XML file found in turn. (doesn't matter CB or IP dir, since
    # each chunk is identified individually by name code and sorted by time
    # stamp into the chunk arrays).  Load each chunk into our wave_assembly
    wave_holder = wave_assembly("")
    wave_holder.pdate = ""

    for fname in flist:
        if debug:
            print("\t", fname)
        if fname.find("filepart") > -1:
            # only a piece of the file; parser will fail
            continue

        # Reset the next_insert positions so a new search will occur
        # Speeds up insertions for out-of-sequence XML
        IE_next_insert = 0
        IA_next_insert = 0
        # IP_next_insert = 0
        BE_next_insert = 0
        BI_next_insert = 0
        BP_next_insert = 0
        # BC_next_insert = 0
        # BA_next_insert = 0
        # BF_next_insert = 0
        # BR_next_insert = 0

        # Check for parser working files and skip
        if (fname.find('/.') > -1):
            if debug:
                print("\t\tParser working file...skipping")
            continue

        filename = fname
        # print ("Parsing '"+filename+"'")

        if not os.path.isfile(filename):
            raise Exception("File Not Found: " + filename)
        tree = ET.parse(filename)
        cpcArchive = tree.getroot()

        # Walk the XML tree - grab seq, datetime, txoffset
        for cpc in cpcArchive:
            offset = cpc.get('tzoffset')
            cpc_seq = cpc.get('seq')

            # get the timestamp and convert to python time
            # These are stored in PST (local time)
            timestamp = ''
            if (cpc.get('datetime').find(".") > -1):
                timestamp = (datetime.datetime.strptime
                             (cpc.get('datetime'), '%Y-%m-%dT%H:%M:%S.%fZ'))
            else:
                timestamp = (datetime.datetime.strptime
                             (cpc.get('datetime'), '%Y-%m-%dT%H:%M:%SZ'))

            # Get offset hours, convert time to UTC
            toff = cpc.get('tzoffset')
            toffstart = 0
            if toff.find("UTC") > -1:
                toffstart = 4
            toff = int(toff[toffstart:toff.find(':')])
            timestamp = timestamp - datetime.timedelta(hours=toff)

            # There should only be one 'device' under cpc; if >1 this will fail
            # (it will silently miss all devices past the first I think)
            device = cpc.find('device')
            device_seq = device.get('seq')

            # Check the status of the device
            if (device.find('status').text == "DATADOWN"):
                # No data...nothing more to do here
                continue

            # Each 'cpc' contains many 'measurements' types.
            for mg in device.find('measurements'):
                if (mg.tag == "m"):
                    # These are single values, we want the groups ("mg")
                    continue

                wave = ''
                offset = 0
                gain = 0
                hz = 0
                points = 0
                binwave = []

                # Now we can pull the 'm' values in
                for m in mg:
                    if (m.attrib["name"] == 'Offset'):
                        offset = int(m.text)
                    elif (m.attrib["name"] == 'Gain'):
                        # GAIN is not correct in the XML for pressures
                        if (mg.get('name') == 'GE_ART'):
                            gain = 0.25
                        elif (mg.get('name') == 'INVP1'):
                            gain = 0.01
                        else:
                            gain = float(m.text)
                    elif (m.attrib["name"] == 'Wave'):
                        wave = m.text
                    elif (m.attrib["name"] == 'Hz'):
                        hz = int(m.text)
                    elif (m.attrib["name"] == 'Points'):
                        points = int(m.text)

                # Convert the base64 char string from the XML into
                # SmallInt (0-255) array
                wave = base64.b64decode(wave)

                # Convert the SmallInt array into Int values
                # pairs of the wave array --> single int value
                for i in range(0, len(wave) - 1, 2):
                    t = (wave[i]) + wave[i + 1] * 256
                    # This is dense: left side CLEARS the 15th bit. Right side
                    #    substracts -32768 from the number if that bit was '1'
                    #    before it was cleared
                    # (t >> 15) grabs the last bit (shifts), leaving 1 or 0
                    t = set_bit(t, 15, 0) + (-32768) * (t >> 15)
                    # Adjust by gain & offset then add to bin array
                    t = t * gain + offset
                    binwave.append(t)

                # Add the wave_chunk to the appropriate wave_assembly
                # Hz gets set here; it happens repetitively.  This is cleaner,
                #   though, than peeking from above since each waveform may have
                #   a different Hz rate (art vs. ecg vs. pleth)
                w = wave_chunk(timestamp, cpc_seq, device_seq, binwave, points)
                if (mg.get('name') == 'GE_ECG'):
                    IE_next_insert = insert_wave_chunk_by_time(wave_holder.ecg,w,IE_next_insert)
                    wave_holder.ecghz = hz
                    pass
                elif (mg.get('name') == 'GE_ART'):
                    IA_next_insert = insert_wave_chunk_by_time(wave_holder.art, w, IA_next_insert)
                    wave_holder.arthz = hz
                    pass
                elif (mg.get('name') == 'GE_PLETH'):
                    IP_next_insert = insert_wave_chunk_by_time(wave_holder.pleth,w,IP_next_insert)
                    wave_holder.plethhz = hz
                    pass
                elif (mg.get('name') == 'ECG1'):
                    BE_next_insert = insert_wave_chunk_by_time(wave_holder.cb_ecg, w, BE_next_insert)
                    wave_holder.cb_ehz = hz
                    pass
                elif (mg.get('name') == 'INVP1'):
                    BI_next_insert = insert_wave_chunk_by_time(wave_holder.cb_inv, w, BI_next_insert)
                    wave_holder.cb_ihz = hz
                elif (mg.get('name') == 'PLETH'):
                    BP_next_insert = insert_wave_chunk_by_time(wave_holder.cb_pleth, w, BP_next_insert)
                    wave_holder.cb_phz = hz
                    pass
                elif (mg.get('name') == 'CO2'):
                    #    BC_next_insert = insert_wave_chunk_by_time(wave_holder.cb_co2,w,BC_next_insert)
                    #    wave_holder.cb_chz = hz
                    pass
                elif (mg.get('name') == 'AWP'):
                    #    BA_next_insert = insert_wave_chunk_by_time(wave_holder.cb_awp,w,BA_next_insert)
                    #    wave_holder.cb_ahz = hz
                    pass
                elif (mg.get('name') == 'FLOW'):
                    #    BF_next_insert = insert_wave_chunk_by_time(wave_holder.cb_flow,w,BF_next_insert)
                    #    wave_holder.cb_fhz = hz
                    pass
                elif (mg.get('name') == 'RESP'):
                    #    BR_next_insert = insert_wave_chunk_by_time(wave_holder.cb_resp,w,BR_next_insert)
                    #    wave_holder.cb_rhz = hz
                    pass
                else:
                    raise Exception("No handler for type '" + mg.get('name'))

    return wave_holder


def check_wave_chunk_sequence(chunk_array):
    """
        The subroutine above will correctly sequence the XML files as they
        are opened and processed ("load_xml_into_wave_assembly()"

        Unfortunately, in rare events the XML chunks themselves WITHIN a
        single file may end up out of order (flipped, or even series of 3 that
        get reversed in order). This is probably a network issue, but it's a
        problem for the parser function below which needs chunks to be in
        correct time order (it can fill missing chunks but not flip mixed
        chunks). Seems to happen more with the IP waves

        Thinking about the cleanest way to do this, we'll run each chunk_array
        right after loading and ensure everything is in the right sequence.

        Assuming the chunk loading parser gets the XML in order, most of the
        time this will be a quick run down the chunks (and it only has to
        happen once per wave, instead of repeatedly when loading)
    """
    _lost_count = 0

    #
    #    print (chunk_array[3].timestamp)
    #    print (chunk_array[2].timestamp)
    #    print (chunk_array[1].timestamp)
    #    print (chunk_array[0].timestamp)
    #    print ("\t\t\tI have "+str(len(chunk_array))+" chunks")

    for i in range(len(chunk_array) - 1):

        # sys.stdout.write(str(i)+". ")

        #        if (len(chunk_array) < 16719):
        #            print ("I have less at ",i)
        #            sys.exit()

        # if the next one is later than this one, all good
        if (chunk_array[i].timestamp < chunk_array[i + 1].timestamp):
            # sys.stdout.write("c")
            continue

        # Equal timestamps with inverted sequences exist!! Ugh!
        # If the timestamps are equal, we need to check to be sure the ith
        # sequence ID is less than the i+1th.
        if (chunk_array[i].timestamp == chunk_array[i + 1].timestamp):
            if (chunk_array[i].device_sequence <
                    chunk_array[i + 1].device_sequence):
                # all good - they're in the right sequence
                # sys.stdout.write("d")
                continue
            # else sequence is reversed and they need resorting; go on below...

        # We passed our two continue statements above...
        # Uh-oh.  The next one is BEFORE this one. (or it's idential time but
        # a higher sequence number.) That's wrong. Pull it out (the next one))
        # print ("Popped off",(i+1),"of",len(chunk_array))
        # print ("\tTS:",chunk_array[i].timestamp,",",chunk_array[i+1].timestamp)
        # print ("\tDS:",chunk_array[i].device_sequence,",",chunk_array[i+1].device_sequence)
        lost_chunk = chunk_array.pop(i + 1)
        _lost_count += 1

        # We'll start from this one and go backwards until we find where it
        # DOES belong and insert it there.  Sort of a backwards bubble sort.
        for j in range(i, -1, -1):

            # If j==0 we're out of array and the chunk must go here.
            if (j == 0):
                chunk_array.insert(j, lost_chunk)

            if (chunk_array[j].timestamp <= lost_chunk.timestamp):
                # Sometimes they're exactly equal on time, so we need to
                # check the damn sequences. If the sequence of the lost_chunk
                # is LESS than the one we're looking at, we need to keep
                # going backward in the wave
                # This condition only applies if the times are equal
                if (chunk_array[j].timestamp == lost_chunk.timestamp):
                    # print ("\ttimestamps match exactly")
                    if (chunk_array[j].device_sequence >
                            lost_chunk.device_sequence):
                        # print ("\tcontinuing")
                        continue

                chunk_array.insert(j + 1, lost_chunk)
                # print ("\tRe-inserted at ",(j+1))
                break
            # print ("timestamps are <=")

        # THIS ALSO MEANS WE HAVE ADVANCED OUR COUNTER ONE MORE!!
        # We picked up one ahead and
        # So advance i.
        i = i + 1

    return _lost_count


def build_wave(wave, hz, wave_type, debug = False):
    """
    Builds the binary waveform from the XML chunks
    Returns a tuple of (np.waveform[], true_hz_rate, length, start_time,
                        end_time, process_note_codes)

    wave_type = 0 : IP wave
    wave_type = 1 : CB wave

    A) The waveform HZ rates are sometimes wrong.  The waves are continuous and
    smooth, but using the hz rates in the files we don't get the right
    times based on the timestamps in the chunks. We will do our best to
    calculate the 'true' Hz rate using the timestamps coming in the chunks.
    The only way I can conceive of this is that the hz rates are inaccurate.

    Talking to Bernoulli, the Hz is not something that comes from the
    machines - it's preset by their recording software. So it may be that the
    remote machines down adjust Hz rates sometimes?

    B) The sequences reset sometimes. This can cause a mismatch. Worse,
        sometimes the cpc and device sequence both reset, sometimes only
        the device sequence.  And cpc sequence is non-sequential at baseline.
        (I think it's counting total sub-entries in the cpc tag). Anyway,
        IF device sequence is suddenly back down to below 200 it's probably
        a reset. One thing I note: The port changes?  It seems like on a
        reset the IP port on the sender may change. Keep an eye on that.

    C) Ways the waveforms chunks can be f'd up (mistakes come from
    the original XML files):

    For "IP" XML files:

        1) Duplicated - identical chunks back to back. Identifiable by
                        identical cpc_seq and dev_seq ID's, as well as
                        identical timestamps, in the successive chunks.
                        HANDLING: Drop duplicate

        2) Missing    - A chunk is not recorded. ID'd by a missing
                        device_seq code between sequential codes
                        HANDLING: Insert '-1' values for missing data

        3) Inverted   - Two chunks end up flipped in order.  Actually, the one
                        I encountered was THREE that suddenly reversed order
                        right in the middle (i.e. 68,69,72,71,70,73,74...)
                        Handling: Above using check_wave_chunk_seq() which
                        was built just to handle these.

    For "CB" XML files:

        1) Skip Seq   - The sequence skips 1 value because the 'missing' seq
                        contained only non-wave field data. We have to check
                        time stamps to detect when a single missing seq is
                        in fact not missing. The MAXIMUM timespan difference
                        found between ANY two valid waveform chunks is 2 sec.
                        So we check and if: 1) The file is a CB XML; 2) the
                        sequence difference is 1 only; and 3) the time diff
                        is 2 sec or less - then we conclude no missing data

        2) Gaps       - A period of 22-23 seconds goes missing; no waveforms
                        at all, just empty XML containers. Seems to ALWAYS be
                        exactly 33 chunks missing. Occurs at regular intervals.
                        Nothing we can do, fill with -1 blank space. Seems to
                        only occur in earlier files, not seeing them in 2017+

        3) "Points"   - 95%+ of the wave chunks have 300 points exactly, but
                        a small number have +/- 20.  They seem to offset one
                        another and rebalance; over 11,000 chunks the average
                        was 300.1, so I think it's safe enough to ignore. We
                        won't bother checking how many points there are, we'll
                        just add whatever IS there and move on.
                        (no correlation with this and Gaps, either, nor any
                        correlation with time error that I can detect)

    """
    if (len(wave) < 1):
        _log("\t\tEmpty wave at process start, nothing to do")
        return ([], 0, 0, 0, 0, "[]")

    # Prescan the wave - this adds time to process, but ultimately we cut
    # the wave down to only the true data which will save disk space and time
    # with all future processing. Some waves without any real data are never
    # saved or processed beyond this, too, saving trouble later looking for
    # 'real' data vs. bad data

    # First, drop all the chunks at the beginning that have
    # zero values >= 50 in the entire chunk (i.e. dead chunk)
    #
    # Scanning has shown that all 'valid' waveforms have data
    # that exceeds 100 in the chunks, and non-valid chunks are generally
    # < 20 to -32000.  That said, some a-lines have aberrant values of 2000
    # so, we'll count and only accept chunks with > 50% values above 10
    _found = 0
    i = 0

    # DISABLED AT THE REQUEST OF Edwards
    # # Trim from the front end
    # for i in range(len(wave)):
    #     for j in range(len(wave[i].wave)):
    #         if (wave[i].wave[j] > 30):
    #            _found += 1
    #     if (_found == 0):
    #         continue
    #     _r = (float(_found))/(float(len(wave[i].wave)))

    #     if (_r > 0.3):
    #         break
    #     else:
    #         _found = 0
    # if (i == len(wave)-1):
    #     _log ("\t\tNo usable data packets!")
    #     return ([],0,0,0,0,"[TRIMMED]")
    # wave = wave[i:len(wave)]
    # _temp = "\t\tTrimmed (" + str(i) + " & "

    # # Now trim from the back end
    # _found = -1
    # for i in range(len(wave)-1,0,-1):
    #     for j in range(len(wave[i].wave)):
    #         if (wave[i].wave[j] > 30):
    #            _found += 1
    #     if (_found == 0):
    #         continue
    #     _r = (float(_found))/(float(len(wave[i].wave)))

    #     if (_r > 0.3):
    #         break
    #     else:
    #         _found = 0
    # _log(_temp + str(len(wave)-i) + ") empty chunks from wave")
    # wave = wave[0:i]

    # If there's less than 15-30 minutes of data it's not very useful
    # or it's noise / error
    # if (len(wave) < 1800):
    #     _log (str(len(wave)) + " chunks remain")
    #     _log ("\t\tLess than 30 min of data, ignoring")
    #     return ([],0,0,0,0,"[<30]")

    # check the wave chunk sequences here for inversions or out of seq.
    # doing it here is deliberate - these inversions are 2-3 chunks at most,
    # so cutting the fat off the front and back end up top means we have less
    # garbage data to have to sort through, meaning faster execution
    _lost_chunks = check_wave_chunk_sequence(wave)
    # _log("\t\tFound "+str(_lost_chunks)+" lost chunks")

    # record keeping
    _dupes = 0
    _misses = 0
    _biggaps = 0
    _nonmiss = 0

    # windex tracks the current insert index of the outwave (and thus
    # also the total length of the wave)
    windex = 0

    # Create a numpy array with enough length to fit the entire incoming wave
    # Because of missing chunks, this needs to be longer than we'd expect.
    # Originally I calculated it by chunks * hz, but a more reliable method
    # is (endtime-starttime).as_seconds * hz

    # _bin_length = int(len(wave)*hz*1.1)
    stamps_length = wave[-1].timestamp - wave[0].timestamp
    _alt_length = int((max(stamps_length.seconds + 1, len(wave))) * hz * 3)
    if debug:
        print(("\t\tStart time : " + str(wave[0].timestamp) + " (" + str(hz) + " hz)"))
        print(("\t\tEnd time   : " + str(wave[-1].timestamp)))
        print(("\t\tXML time   : " + str(stamps_length)))
        print(("\t\tAllocating " + str(_alt_length) + " integer array"))

    gc.collect()
    outwave = np.zeros(_alt_length, dtype=np.int16)

    # initialize off the first chunk in the remaining wave
    seq = int(wave[0].device_sequence)
    for i in range(len(wave[0].wave)):
        outwave[windex] = wave[0].wave[i]
        windex += 1
    last_timestamp = wave[0].timestamp
    cpc_seq = int(wave[0].cpc_sequence)

    c = 0
    # Now run the rest of the chunks
    for wave_chunk in wave[1:len(wave)]:
        c += 1
        # print ("Chunk",c,windex)
        # FIRST: Look for the single missing seq chunks
        # If it's EXACTLY one missing it might be a non-missing chunk
        # But only if this is a CB wave

        if (wave_chunk.device_sequence == (seq + 2)) and (wave_type == 1):
            d = wave_chunk.timestamp - last_timestamp
            if (d.seconds <= 2):
                # NOT actually missing, just the chunk in between had
                # non-waveform information. Iterate seq here so it matches
                # on the check below (i.e. no fill) - pass execution on down
                seq = seq + 1
                _nonmiss += 1

        # Check for the 21-23 second gaps in the CB files
        elif (wave_chunk.device_sequence == (seq + 33)) and (wave_type == 1):
            # This is a 'CB gap'; always exactly 33 packets and 20-24 sec
            # calculate the total missing time & fill, then update seq
            # so it matches on the check below (i.e. yes fill)
            d = wave_chunk.timestamp - last_timestamp
            for i in range(hz * d.seconds):
                outwave[windex] = -1
                windex += 1
            seq = seq + 32
            _biggaps += 1
        # Check for sequence reset
        elif (wave_chunk.device_sequence < seq and
              wave_chunk.device_sequence < 500):
            # Sequence reset has occured.  Move the reference markers to
            # make them match the reset
            seq = wave_chunk.device_sequence - 1

        # By the time we get here, if the wave_chunk.seq is still > seq+1
        # then we have true missing time not accounted for in either of the
        # above. We have to fill it in. We CANNOT do this by chunks*hz. For
        # CB waves, the 'non-gap' missing chunks occur every 2 chunks and will
        # throw off the counter. We'll do the best we can with timestamps.

        if (wave_chunk.device_sequence > (seq + 1)):
            d = (wave_chunk.timestamp - last_timestamp).seconds
            _log
            for i in range(hz * d):
                outwave[windex] = -1
                windex += 1
            _misses += 1
            # Correct the sequence so it matches
            seq = wave_chunk.device_sequence - 1

        #        while (wave_chunk.device_sequence > (seq+1)):
        #            for i in range (hz):
        #                outwave[windex] = -1
        #                windex += 1
        #            _misses += 1
        #            seq = seq + 1
        #

        # After the work above, the new chunk sequence should be = old seq +1
        if (wave_chunk.device_sequence == (seq + 1)):
            # sequence is good, add it to the array
            for i in range(len(wave_chunk.wave)):
                outwave[windex] = wave_chunk.wave[i]
                windex += 1
            # Update all of the existing data for the next cycle
            seq = wave_chunk.device_sequence
            last_timestamp = wave_chunk.timestamp
            cpc_seq = wave_chunk.cpc_sequence
        elif (
                (wave_chunk.device_sequence == seq) and
                (wave_chunk.cpc_sequence == cpc_seq) and
                (wave_chunk.timestamp == last_timestamp)
        ):
            # Duplicated chunk; ignore
            _dupes += 1
            continue

        else:
            # Well shit.  Something bad happened.
            print("\t\tSEQUENCE MISMATCH")
            print((seq, cpc_seq, last_timestamp))
            print((wave_chunk.device_sequence, wave_chunk.cpc_sequence,
                  wave_chunk.timestamp))
            break

    # _log outcome
    _log("\t\tProcessed " + str(len(wave)) + " total chunks:\n\t\t\t" + str(_dupes) +
         " dupes," + str(_misses) + " missing," + str(_biggaps) +
         " gaps," + str(_nonmiss) + " non-wave")
    _proc_notes = ("[" + str(len(wave)) + ":" + str(_dupes) +
                   ":" + str(_misses) + ":" + str(_biggaps) +
                   ":" + str(_nonmiss) + ":" + str(_lost_chunks) + "]")

    # Now calculate the true hz rate of this waveform
    wave_length = datetime.timedelta(seconds=(windex / hz))
    wave_end_time = wave[0].timestamp + wave_length
    Err = wave[-1].timestamp - wave_end_time
    true_hz = windex / (stamps_length.total_seconds())
    true_hz = int(true_hz * 1000) / 1000

    _log("\t\tOutwave len: " + str(windex))
    _log("\t\tHz Est time: " + str(wave_length))
    _log("\t\tTime error : " + str(Err))
    _log("\t\tTrue Hz    : " + str(true_hz))

    # Trim the wave down to only the core (cut off whatever we didn't use
    # at the end of our allocated waveform array)
    outwave = outwave[0:windex]

    # Get the final start and end times (from the CHUNKS - so they're accurate
    # and direct from the XML they were pulled from)
    wave_start_time = wave[0].timestamp
    wave_end_time = wave[-1].timestamp

    # DELETE THE XML CHUNKS NOW!  We're done with them
    wave = []
    return (outwave, true_hz, windex, wave_start_time,
            wave_end_time, _proc_notes)


def _log(text):
    # print (text)
    pass
