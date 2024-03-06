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


def check_data_in_filelist(flist: list[str]) -> bool:

    # Open each XML file found in turn. (doesn't matter CB or IP dir, since
    # each chunk is identified individually by name code and sorted by time
    # stamp into the chunk arrays).  Load each chunk into our WaveAssembly
    wave_holder = WaveAssembly("")
    wave_holder.pdate = ""

    allowed_attribute_types = set()
    for value in WaveAssembly.AttributeType:
        allowed_attribute_types.add(str(value.value))

    flag = False
    for fname in flist:
        if "filepart" in fname or '/.' in fname:
            continue
        cpc_archive = ET.parse(fname).getroot()

        wave_holder.reset_all_insertion_indexes()

        # Walk the XML tree - grab seq, datetime, txoffset
        for cpc in cpc_archive:

            # get the timestamp and convert to python time
            # These are stored in PST (local time)

            # There should only be one 'device' under cpc; if >1 this will fail
            # (it will silently miss all devices past the first I think)
            device = cpc.find('device')

            # Check the status of the device
            if (device.find('status').text != "DATADOWN"):
                # Got data
                flag = True
    return flag