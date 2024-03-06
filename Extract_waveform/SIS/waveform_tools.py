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
from __future__ import annotations

import datetime
from enum import IntEnum, Enum
import gc
import xml.etree.ElementTree as ET
import os
import base64
import numpy as np
from collections import defaultdict
from pprint import pformat

class WaveType(IntEnum):
    IP = 0
    CB = 1


class WaveChunk:
    timestamp: datetime.datetime
    cpc_sequence: int
    device_sequence: int
    wave: list[int | float]
    points: int

    def __init__(self, timestamp, cpc_sequence, device_sequence, wave, points):
        self.cpc_sequence = int(cpc_sequence)
        self.device_sequence = int(device_sequence)
        self.wave = wave
        self.timestamp = timestamp
        self.points = int(points)

    def __str__(self) -> str:
        return pformat(vars(self), depth=1)


class WaveAssembly:
    class AttributeType(Enum):
        ECG = "GE_ECG"
        ART = "GE_ART"
        PLETH = "GE_PLETH"
        CB_ECG = "ECG1"
        CB_INV = "INVP1"
        CB_PLETH = "PLETH"
        CB_CO2 = "CO2"
        CB_AWP = "AWP"
        CB_FLOW = "FLOW"
        CB_RESP = "RESP"

        @staticmethod
        def from_str(s: str):
            if s == "GE_ECG":
                return WaveAssembly.AttributeType.ECG
            elif s == "GE_ART":
                return WaveAssembly.AttributeType.ART
            elif s == "GE_PLETH":
                return WaveAssembly.AttributeType.PLETH
            elif s == "ECG1":
                return WaveAssembly.AttributeType.CB_ECG
            elif s == "INVP1":
                return WaveAssembly.AttributeType.CB_INV
            elif s == "PLETH":
                return WaveAssembly.AttributeType.CB_PLETH
            elif s == "CO2":
                return WaveAssembly.AttributeType.CB_CO2
            elif s == "AWP":
                return WaveAssembly.AttributeType.CB_AWP
            elif s == "FLOW":
                return WaveAssembly.AttributeType.CB_FLOW
            elif s == "RESP":
                return WaveAssembly.AttributeType.CB_RESP
            else:
                raise NotImplementedError

    class Attribute:
        chunks: list[WaveChunk] = []
        insertion_index: int = 0
        hz: int = 0

        def reorder_chunk_sequence(self) -> int:
            """
                check the wave chunk sequences here for inversions or out of seq.
                doing it here is deliberate - these inversions are 2-3 chunks at most,
                so cutting the fat off the front and back end up top means we have less
                garbage data to have to sort through, meaning faster execution

                The subroutine above will correctly sequence the XML files as they
                are opened and processed ("load_xml_into_wave_assembly()"

                Unfortunately, in rare events the XML chunks themselves WITHIN a
                single file may end up out of order (flipped, or even series of 3 that
                get reversed in order). This is probably a network issue, but it's a
                problem for the parser function below which needs chunks to be in
                correct time order (it can fill missing chunks but not flip mixed
                chunks). Seems to happen more with the IP waves

                Thinking about the cleanest way to do this, we'll run each wave_chunks
                right after loading and ensure everything is in the right sequence.

                Assuming the chunk loading parser gets the XML in order, most of the
                time this will be a quick run down the chunks (and it only has to
                happen once per wave, instead of repeatedly when loading)
            """
            lost_chunks_count = 0

            for i in range(len(self.chunks)-1):

                # if the next one is later than this one, all good
                if (self.chunks[i].timestamp < self.chunks[i+1].timestamp):
                    continue

                # Equal timestamps with inverted sequences exist!! Ugh!
                # If the timestamps are equal, we need to check to be sure the ith
                # sequence ID is less than the i+1th.
                if (self.chunks[i].timestamp == self.chunks[i+1].timestamp):
                    if (self.chunks[i].device_sequence <
                            self.chunks[i+1].device_sequence):
                        # all good - they're in the right sequence
                        # sys.stdout.write("d")
                        continue
                    # else sequence is reversed and they need resorting; go on below...

                # We passed our two continue statements above...
                # Uh-oh.  The next one is BEFORE this one. (or it's idential time but
                # a higher sequence number.) That's wrong. Pull it out (the next one))
                lost_chunk = self.chunks.pop(i+1)
                lost_chunks_count += 1

                # We'll start from this one and go backwards until we find where it
                # DOES belong and insert it there.  Sort of a backwards bubble sort.
                for j in range(i, -1, -1):

                    # If j==0 we're out of array and the chunk must go here.
                    if (j == 0):
                        self.chunks.insert(j, lost_chunk)

                    if (self.chunks[j].timestamp <= lost_chunk.timestamp):
                        # Sometimes they're exactly equal on time, so we need to
                        # check the damn sequences. If the sequence of the lost_chunk
                        # is LESS than the one we're looking at, we need to keep
                        # going backward in the wave
                        # This condition only applies if the times are equal
                        if (self.chunks[j].timestamp == lost_chunk.timestamp):
                            if (self.chunks[j].device_sequence >
                                    lost_chunk.device_sequence):
                                continue

                        self.chunks.insert(j+1, lost_chunk)
                        break

                # THIS ALSO MEANS WE HAVE ADVANCED OUR COUNTER ONE MORE!!
                # We picked up one ahead and
                # So advance i.
                i += 1

            return lost_chunks_count

        def insert_chunk_by_time(self, new_chunk: WaveChunk):
            # Append if FIRST chunk or if next_insert is == -1 (the "end" in python)
            if (len(self.chunks) == 0) or (self.insertion_index == -1):
                self.chunks.append(new_chunk)
                self.insertion_index = -1
                return

            # If next_insert is > 0, we insert there
            if self.insertion_index > 0:
                self.chunks.insert(self.insertion_index, new_chunk)
                self.insertion_index += 1
                return

            # must search for position
            for i, _ in enumerate(self.chunks):
                if new_chunk.timestamp < self.chunks[i].timestamp:
                    self.chunks.insert(i, new_chunk)
                    self.insertion_index = i + 1
                    return

            # We didn't find a later chunk, so this new chunk goes at the end
            self.chunks.append(new_chunk)
            self.insertion_index = -1

    ptid: str
    pdate: str
    wave_type: WaveType
    attributes: defaultdict(Attribute) = defaultdict(Attribute)

    def __init__(self, ptid: str):
        self.ptid = ptid

    def reset_all_insertion_indexes(self):
        for k in self.attributes:
            self.attributes[k].insertion_index = 0

    def __str__(self) -> str:
        return pformat(vars(self))


class WaveForm:
    data_points: np.ndarray[np.int16] = np.zeros(0, dtype=np.int16)
    true_hz_rate: float
    start_time: datetime.datetime
    end_time: datetime.datetime

    class Metrics:

        duplicates: int = 0
        misses: int = 0
        gaps: int = 0
        non_waves: int = 0
        losses: int = 0

        def __str__(self) -> str:
            return pformat(vars(self))
    metrics: Metrics

    def __init__(self, data_points: np.ndarray[np.int16], true_hz_rate: float, start_time: datetime.datetime, end_time: datetime.datetime, metrics: Metrics) -> None:
        self.data_points = data_points
        # make data points immutable
        self.data_points.flags.writeable = False
        self.true_hz_rate = true_hz_rate
        self.start_time = start_time
        self.end_time = end_time
        self.metrics = metrics

    def __str__(self) -> str:
        return pformat(vars(self))


def _set_bit(v, index, x):
    """
        Set the index:th bit of v to 1 if x is truthy,
        else to 0, and return the new value.
    """
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask         # If x was True, set the bit indicated by the mask.
    return v            # Return the result, we're done.


def _convert_to_utc_timestamp(dt: str, tzoffset: str) -> datetime.datetime:
    date_format = '%Y-%m-%dT%H:%M:%S.%fZ' if '.' in dt else '%Y-%m-%dT%H:%M:%SZ'
    toffstart = 0
    if 'UTC' in tzoffset:
        toffstart = 4
    tzoffset = int(tzoffset[toffstart:tzoffset.find(':')])
    return datetime.datetime.strptime(dt, date_format) - datetime.timedelta(hours=tzoffset)


def list_xml_files(dirname: str, wave_type: WaveType) -> list[str]:
    fpaths = []
    for fname in os.listdir(dirname):
        fpath = os.path.join(dirname, fname)
        if not os.path.isfile(fpath):
            continue

        _, extension = os.path.splitext(fpath)
        if extension != ".xml":
            continue

        if '000Z' not in fname and (wave_type is WaveType.IP):
            fpaths.append(fpath)
        elif '000Z' in fname and (wave_type is not WaveType.IP):
            fpaths.append(fpath)

    return fpaths


def list_dirs(dirname: str) -> list[str]:
    sub_dirs = []
    for sub_dir_name in os.listdir(dirname):
        sub_dir_path = os.path.join(dirname, sub_dir_name)
        if os.path.isdir(sub_dir_path) \
            and len(sub_dir_name) == 16 \
                and sub_dir_name.isalnum():
            sub_dirs.append(sub_dir_path)
    return sub_dirs


def _get_hz_from_measurement_group(measurement_group: ET.Element) -> int:
    for measurement in measurement_group:
        if measurement.attrib["name"] == 'Hz':
            return int(measurement.text)
    return 0


def _get_wave_chunk_from_measurement_group(
        measurement_group: ET.Element,
        timestamp: datetime.datetime,
        cpc_seq: str,
        device_seq: str
) -> WaveChunk:
    measurement_group_name = measurement_group.get('name')
    wave = ''
    offset = 0
    gain = 0
    points = 0

    for measurement in measurement_group:
        attribute_name = measurement.attrib["name"]

        if attribute_name == 'Offset':
            offset = int(measurement.text)
        elif attribute_name == 'Gain':
            # GAIN is not correct in the XML for pressures
            if measurement_group_name == 'GE_ART':
                gain = 0.25
            elif measurement_group_name == 'INVP1':
                gain = 0.01
            else:
                gain = float(measurement.text)
        elif attribute_name == 'Wave':
            wave = measurement.text
        elif attribute_name == 'Points':
            points = int(measurement.text)

    # Convert the base64 char string from the XML into
    # SmallInt (0-255) array
    wave = base64.b64decode(wave)

    # Convert the SmallInt array into Int values
    # pairs of the wave array --> single int value
    binwave = []
    for i in range(0, len(wave) - 1, 2):
        value = wave[i] + (wave[i+1] * 256)
        # This is dense: left side CLEARS the 15th bit. Right side
        #    substracts -32768 from the number if that bit was '1'
        #    before it was cleared
        # (t >> 15) grabs the last bit (shifts), leaving 1 or 0
        value = _set_bit(value, 15, 0) + (-32768)*(value >> 15)
        # Adjust by gain & offset then add to bin array
        value = value * gain + offset
        binwave.append(value)

    # Add the wave_chunk to the appropriate WaveAssembly
    # Hz gets set here; it happens repetitively.  This is cleaner,
    #   though, than peeking from above since each waveform may have
    #   a different Hz rate (art vs. ecg vs. pleth)
    return WaveChunk(timestamp, cpc_seq, device_seq, binwave, points)


def get_wave_assembly_from_xml_files(flist: list[str]) -> WaveAssembly:

    # Open each XML file found in turn. (doesn't matter CB or IP dir, since
    # each chunk is identified individually by name code and sorted by time
    # stamp into the chunk arrays).  Load each chunk into our WaveAssembly
    wave_holder = WaveAssembly("")
    wave_holder.pdate = ""

    allowed_attribute_types = set()
    for value in WaveAssembly.AttributeType:
        allowed_attribute_types.add(str(value.value))

    for fname in flist:
        if "filepart" in fname or '/.' in fname:
            continue

        cpc_archive = ET.parse(fname).getroot()

        wave_holder.reset_all_insertion_indexes()

        # Walk the XML tree - grab seq, datetime, txoffset
        for cpc in cpc_archive:

            # get the timestamp and convert to python time
            # These are stored in PST (local time)
            timestamp = _convert_to_utc_timestamp(cpc.get('datetime'), cpc.get('tzoffset'))

            # There should only be one 'device' under cpc; if >1 this will fail
            # (it will silently miss all devices past the first I think)
            device = cpc.find('device')

            # Check the status of the device
            if (device.find('status').text == "DATADOWN"):
                # No data...nothing more to do here
                continue

            # Each 'cpc' contains many 'measurements' types.
            for mg in device.find('measurements'):
                if mg.tag == "m":
                    # These are single values, we want the groups ("mg")
                    continue

                hz = _get_hz_from_measurement_group(mg)
                w = _get_wave_chunk_from_measurement_group(
                    mg, timestamp, cpc.get('seq'), device.get('seq'))

                mg_name = mg.get('name')
                if mg_name in allowed_attribute_types:
                    attribute_type = WaveAssembly.AttributeType.from_str(mg_name)
                    wave_holder.attributes[attribute_type].insert_chunk_by_time(w)
                    wave_holder.attributes[attribute_type].hz = hz
                else:
                    raise TypeError(f"No handler for type '{mg_name}'")

    return wave_holder


def build_wave_form(wave_attribute: WaveAssembly.Attribute, wave_type: WaveType) -> WaveForm:
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
    metrics = WaveForm.Metrics()
    metrics.losses = wave_attribute.reorder_chunk_sequence()
    wave_chunks = wave_attribute.chunks
    hz = wave_attribute.hz

    if not wave_chunks:
        return None

    # windex tracks the current insert index of the outwave (and thus
    # also the total length of the wave)
    windex = 0

    # Create a numpy array with enough length to fit the entire incoming wave
    # Because of missing chunks, this needs to be longer than we'd expect.
    # Originally I calculated it by chunks * hz, but a more reliable method
    # is (endtime-starttime).as_seconds * hz

    # _bin_length = int(len(wave)*hz*1.1)
    stamps_length = wave_chunks[-1].timestamp - wave_chunks[0].timestamp
    _alt_length = int(max(stamps_length.seconds +
                      1, len(wave_chunks)) * hz * 2)

    gc.collect()
    outwave = np.zeros(_alt_length, dtype=np.int16)

    # initialize off the first chunk in the remaining wave
    seq = int(wave_chunks[0].device_sequence)
    for i, _ in enumerate(wave_chunks[0].wave):
        outwave[windex] = wave_chunks[0].wave[i]
        windex += 1

    last_timestamp = wave_chunks[0].timestamp
    cpc_seq = int(wave_chunks[0].cpc_sequence)

    c = 0
    # Now run the rest of the chunks
    for wave_chunk in wave_chunks[1:len(wave_chunks)]:
        c += 1
        # FIRST: Look for the single missing seq chunks
        # If it's EXACTLY one missing it might be a non-missing chunk
        # But only if this is a CB wave

        if (wave_chunk.device_sequence == (seq+2)) and (wave_type == WaveType.CB): # why is here seq +2?
            d = wave_chunk.timestamp - last_timestamp
            if (d.seconds <= 2):
                # NOT actually missing, just the chunk in between had
                # non-waveform information. Iterate seq here so it matches
                # on the check below (i.e. no fill) - pass execution on down
                seq = seq + 1
                metrics.non_waves += 1

        # Check for the 21-23 second gaps in the CB files
        elif (wave_chunk.device_sequence == (seq+33)) and (wave_type == WaveType.CB):
            # This is a 'CB gap'; always exactly 33 packets and 20-24 sec
            # calculate the total missing time & fill, then update seq
            # so it matches on the check below (i.e. yes fill)
            d = wave_chunk.timestamp - last_timestamp
            for i in range(hz * d.seconds):
                outwave[windex] = -1
                windex += 1
            seq = seq + 32
            metrics.gaps += 1
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

        if (wave_chunk.device_sequence > (seq+1)):
            d = (wave_chunk.timestamp - last_timestamp).seconds
            for i in range(hz * d):
                outwave[windex] = -1
                windex += 1
            metrics.misses += 1
            # Correct the sequence so it matches
            seq = wave_chunk.device_sequence - 1

        # After the work above, the new chunk sequence should be = old seq +1
        if (wave_chunk.device_sequence == (seq+1)):
            # sequence is good, add it to the array
            for i, _ in enumerate(wave_chunk.wave):
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
            metrics.duplicates += 1
            continue

        else:
            raise RuntimeError(f"""Sequence Mismatch, next should be prev or prev+1\n"""
                               f"""\tdevice_seq\tcpc_seq\t\ttimestamp\n"""
                               f"""prev:\t{seq}\t\t{cpc_seq}\t\t{last_timestamp}\n"""
                               f"""next:\t{wave_chunk.device_sequence}\t\t{wave_chunk.cpc_sequence}\t\t{wave_chunk.timestamp}\n""")

    # Now calculate the true hz rate of this waveform
    wave_length = datetime.timedelta(seconds=(windex/hz))
    wave_end_time = wave_chunks[0].timestamp + wave_length
    time_error = wave_chunks[-1].timestamp - wave_end_time
    true_hz = windex / (stamps_length.total_seconds())
    true_hz = int(true_hz*1000)/1000

    # Trim the wave down to only the core (cut off whatever we didn't use
    # at the end of our allocated waveform array)
    outwave = outwave[0:windex]

    # Get the final start and end times (from the CHUNKS - so they're accurate
    # and direct from the XML they were pulled from)
    wave_start_time = wave_chunks[0].timestamp
    wave_end_time = wave_chunks[-1].timestamp

    # clear the chunks
    wave_attribute.chunks = []

    return WaveForm(outwave, true_hz, wave_start_time, wave_end_time, metrics)
