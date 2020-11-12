"""
-----------------------------------------------------------------------
                              Isaura
-----------------------------------------------------------------------
From ....
This city computes tracks extracts topology information.
The input is esmeralda energy corrected hits or beersheba deconvoluted hits, and mc info.
The city outputs :
    - MC info (if run number <=0)
    - Tracking/Tracks - summary of per track information
    - Summary/events  - summary of per event information
    - DST/Events      - copy of kdst information from penthesilea
"""

import tables as tb

from .. reco                import tbl_functions        as tbl
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import hits_and_kdst_from_files
from .  components import dhits_from_files
from .  components import compute_and_write_tracks_info
from .  components import copy_E_or_Ec_to_Ep

from ..  evm.event_model import HitEnergy

from ..  io.run_and_event_io import run_and_event_writer
from ..  io.         kdst_io import  kdst_from_df_writer
from ..  io.event_filter_io  import  event_filter_writer




@city
def isaura(files_in, file_out, compression, event_range, print_mod,
              detector_db, run_number,
              paolina_params = dict()):
    """
    The city extracts topology information.
    ----------
    Parameters
    ----------
    files_in  : str, filepath
         input file
    file_out  : str, filepath
         output file
    compression : str
         Default  'ZLIB4'
    event_range : int /'all_events'
         number of events from files_in to process
    print_mode : int
         how frequently to print events
    run_number : int
         has to be negative for MC runs
    paolina_params               :dict
        vox_size                 : [float, float, float]
            (maximum) size of voxels for track reconstruction
        strict_vox_size          : bool
            if False allows per event adaptive voxel size,
            smaller of equal thatn vox_size
        energy_threshold        : float
            if energy of end-point voxel is smaller
            the voxel will be dropped and energy redistributed to the neighbours
        min_voxels              : int
            after min_voxel number of voxels is reached no dropping will happen.
        blob_radius             : float
            radius of blob
        max_num_hits            : int
            maximum number of hits allowed per event to run paolina functions.
    ----------
    Input
    ----------
    Beersheba output
    ----------
    Output
    ----------
    - MC info (if run number <=0)
    - Tracking/Tracks - summary of per track information
    - Summary/events  - summary of per event information
    - DST/Events      - copy of kdst information from penthesilea
"""

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()
    
    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        #write_kdst_table      = fl.sink( kdst_from_df_writer(h5out)                      , args="kdst"               )
        write_no_hits_filter  = fl.sink( event_filter_writer(h5out, "hits_select" )   , args=("event_number", "hits_passed"))

        evtnum_collect = collect()
        
        # Filter events without hits
        filter_events_nohits = fl.map(lambda x : len(x.hits) > 0,
                                      args = 'hits',
                                      out  = 'hits_passed')
        hits_passed          = fl.count_filter(bool, args="hits_passed")


        copy_E_to_Ep_hit_attribute = fl.map(copy_E_or_Ec_to_Ep(HitEnergy.E),
                                            args = 'hits',
                                            out  = 'Ep_hits')

        compute_tracks = compute_and_write_tracks_info(paolina_params, h5out)

        result = push(source = dhits_from_files(files_in),
                      #source = hits_and_kdst_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)        ,
                                    print_every(print_mod)                        ,
                                    event_count_in        .spy                    ,
                                    #fl.branch(fl.fork(write_kdst_table            ,
                                    #                  write_event_info          )),
                                    fl.branch("event_number", evtnum_collect.sink),
                                    filter_events_nohits                          ,
                                    fl.branch(write_no_hits_filter)               ,
                                    hits_passed.              filter              ,
                                    copy_E_to_Ep_hit_attribute                    ,
                                    compute_tracks                                ,
                                    event_count_out       .spy                    ,
                                    write_event_info                              ),
                      result = dict(events_in  =event_count_in .future,
                                    events_out =event_count_out.future,
                                    evtnum_list=evtnum_collect .future))


        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
        
        
        
        
