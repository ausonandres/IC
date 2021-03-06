import numpy  as np
import tables as tb

from functools import  partial
from typing    import Callable
from typing    import     List
from typing    import Optional
from typing    import    Tuple

from .. evm .nh5         import           MCEventMap
from .. reco             import        tbl_functions as tbl
from .  run_and_event_io import run_and_event_writer
from .  table_io         import           make_table


def rwf_writer(h5out           : tb.file.File          ,
               *,
               group_name      :          str          ,
               table_name      :          str          ,
               compression     :          str = 'ZLIB4',
               n_sensors       :          int          ,
               waveform_length :          int          ) -> Callable:
    """
    Defines group and table where raw waveforms
    will be written.

    h5out           : pytables file
                      Output file where waveforms to be saved
    group_name      : str
                      Name of the group in h5in.root
                      Known options: RD, BLR
                      Setting to None will save directly in root
    table_name      : str
                      Name of the table
                      Known options: pmtrwf, pmtcwf, sipmrwf
    compression     : str
                      file compression
    n_sensors       : int
                      number of sensors in the table (shape[0])
    waveform_length : int
                      Number of samples per sensor
    """
    if   group_name is None:
        rwf_group = h5out.root
    elif group_name in h5out.root:
        rwf_group = getattr           (h5out.root, group_name)
    else:
        rwf_group = h5out.create_group(h5out.root, group_name)

    rwf_table = h5out.create_earray(rwf_group                                ,
                                    table_name                               ,
                                    atom    =                  tb.Int16Atom(),
                                    shape   = (0, n_sensors, waveform_length),
                                    filters =        tbl.filters(compression))
    def write_rwf(waveform : np.ndarray) -> None:
        """
        Writes raw waveform arrays to file.
        waveform : np.ndarray
                   shape = (n_sensors, waveform_length) array
                   of sensor charge.
        """
        rwf_table.append(waveform.reshape(1, n_sensors, waveform_length))
    return write_rwf


def buffer_writer(h5out, *,
                  run_number :          int           ,
                  n_sens_eng :          int           ,
                  n_sens_trk :          int           ,
                  length_eng :          int           ,
                  length_trk :          int           ,
                  group_name : Optional[str] =    None,
                  compression: Optional[str] = 'ZLIB4'
                  ) -> Callable[[int, List, List], None]:
    """
    Generalised buffer writer which defines a raw waveform writer
    for each type of sensor as well as an event info writer.
    Each call gives a list of 'triggers' to be written as
    separate events in the output.

    parameters
    ----------
    run_number  : int
                  Run number to be saved in runInfo.
    n_sens_eng  : int
                  Number of sensors in the energy plane.
    n_sens_trk  : int
                  Number of sensors in the tracking plane.
    length_eng  : int
                  Number of samples per waveform for energy plane.
    length_trk  : int
                  Number of samples per waveform for tracking plane.
    group_name  : Optional[str] default None
                  Group name within root where waveforms to be saved.
                  Default directly in root
    compression : Optional[str] default 'ZLIB4'
                  Compression level for output file.

    returns
    -------
    write_buffers : Callable
                    A function which takes event information
                    for the tracking and energy planes and
                    the event timestamps and saves to file.
    """

    eng_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =     'pmtrd',
                            n_sensors       =  n_sens_eng,
                            waveform_length =  length_eng)

    trk_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =    'sipmrd',
                            n_sensors       =  n_sens_trk,
                            waveform_length =  length_trk)

    run_and_event = partial(run_and_event_writer(h5out                  ,
                                                 compression=compression),
                            run_number = run_number                      )

    nexus_map = make_table(h5out, 'Run', 'eventMap', MCEventMap,
                           "event & nexus evt for each index", compression)

    def write_buffers(nexus_evt  :        int ,
                      timestamps : List[  int],
                      events     : List[Tuple]) -> None:
        """
        Write run info and event waveforms to file.
        
        parameters
        ----------
        nexus_evt  :  int
                     Event number from MC output file.
        timestamps : List[int]
                     List of event times
        events     : List[Tuple]
                     List of tuples containing the energy and
                     tracking plane info for each identified 'trigger'.
        """

        for i, (t_stamp, (eng, trk)) in enumerate(zip(timestamps, events)):
            ## The exact way to log MC event splitting
            ## still to be decided.
            run_and_event(event_number=nexus_evt, timestamp=t_stamp)
            mrow = nexus_map.row
            mrow["evt_number"] = nexus_evt
            mrow[   "sub_evt"] = i
            mrow.append()
            ##

            eng_writer(eng)
            trk_writer(trk)
    return write_buffers
