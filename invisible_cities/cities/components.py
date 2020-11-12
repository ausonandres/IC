from functools   import wraps
from functools   import partial
from collections import Sequence
from argparse    import Namespace
from glob        import glob
from os.path     import expandvars
from itertools   import count
from itertools   import repeat
from enum        import Enum
from typing      import Callable
from typing      import Iterator
from typing      import Mapping
from typing      import Generator
from typing      import List
from typing      import Dict
from typing      import Tuple
from typing      import Union
import tables as tb
import numpy  as np
import pandas as pd
import inspect
import warnings

from collections import OrderedDict

from .. dataflow                  import                  dataflow as  fl
from .. dataflow.dataflow         import                      sink
from .. dataflow.dataflow         import                      pipe
from .. evm    .ic_containers     import                SensorData
from .. evm    .event_model       import                   KrEvent
from .. evm    .event_model       import                       Hit
from .. evm    .event_model       import                   Cluster
from .. evm    .event_model       import             HitCollection
from .. evm    .event_model       import                 HitEnergy
from .. evm    .event_model       import                    MCInfo
from .. evm    .pmaps             import                SiPMCharge
from .. core                      import           system_of_units as units
from .. core   .exceptions        import                XYRecoFail
from .. core   .exceptions        import           MCEventNotFound
from .. core   .exceptions        import              NoInputFiles
from .. core   .exceptions        import              NoOutputFile
from .. core   .exceptions        import InvalidInputFileStructure
from .. core   .configure         import                EventRange
from .. core   .configure         import          event_range_help
from .. core   .random_sampling   import              NoiseSampler
from .. detsim                    import          buffer_functions as  bf
from .. reco                      import           calib_functions as  cf
from .. reco                      import          sensor_functions as  sf
from .. reco                      import   calib_sensors_functions as csf
from .. reco                      import            peak_functions as pkf
from .. reco                      import           pmaps_functions as pmf
from .. reco                      import            hits_functions as hif
from .. reco                      import             wfm_functions as wfm
from .. reco                      import         paolina_functions as plf
from .. reco   .xy_algorithms     import                    corona
from .. filters.s1s2_filter       import               S12Selector
from .. filters.s1s2_filter       import               pmap_filter
from .. database                  import                   load_db
from .. sierpe                    import                       blr
from .. io                        import                 mcinfo_io
from .. io     .pmaps_io          import                load_pmaps
from .. io     .hits_io           import              hits_from_df
from .. io     .dst_io            import                  load_dst
from .. io     .event_filter_io   import       event_filter_writer
from .. io     .pmaps_io          import               pmap_writer
from .. io     .dst_io            import                 df_writer
from .. types  .ic_types          import                        xy
from .. types  .ic_types          import                        NN
from .. types  .ic_types          import                       NNN
from .. types  .ic_types          import                    minmax

NoneType = type(None)


def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        if hasattr(conf, 'config_file'):       del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        if hasattr(conf, 'print_config_only'): del conf.print_config_only
        if hasattr(conf, 'hide_config'):       del conf.hide_config
        if hasattr(conf, 'no_overrides'):      del conf.no_overrides
        if hasattr(conf, 'no_files'):          del conf.no_files
        if hasattr(conf, 'full_files'):        del conf.full_files

        # TODO: we have decided to remove verbosity.
        # Needs to be removed form config parser
        if hasattr(conf, 'verbosity'):         del conf.verbosity

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        # For backward-compatibility we set NEW as the default DB in
        # case it is not defined in the config file
        if 'detector_db' in inspect.getfullargspec(city_function).args and \
           'detector_db' not in kwds:
            conf.detector_db = 'new'

        conf.files_in  = sorted(glob(expandvars(conf.files_in)))
        conf.file_out  =             expandvars(conf.file_out)

        conf.event_range  = event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        result = city_function(**vars(conf))
        index_tables(conf.file_out)
        return result
    return proxy


def index_tables(file_out):
    """
    -finds all tables in output_file
    -checks if any columns in the tables have been marked to be indexed by writers
    -indexes those columns
    """
    with tb.open_file(file_out, 'r+') as h5out:
        for table in h5out.walk_nodes(classname='Table'):        # Walk over all tables in h5out
            if 'columns_to_index' not in table.attrs:  continue  # Check for columns to index
            for colname in table.attrs.columns_to_index:         # Index those columns
                table.colinstances[colname].create_index()


def _check_invalid_event_range_spec(er):
    return (len(er) not in (1, 2)                   or
            (len(er) == 2 and EventRange.all in er) or
            er[0] is EventRange.last                )


def event_range(conf):
    # event_range not specified
    if not hasattr(conf, 'event_range')           : return None, 1
    er = conf.event_range

    if not isinstance(er, Sequence): er = (er,)
    if _check_invalid_event_range_spec(er):
        message = "Invalid spec for event range. Only the following are accepted:\n" + event_range_help
        raise ValueError(message)

    if   len(er) == 1 and er[0] is EventRange.all : return (None,)
    elif len(er) == 2 and er[1] is EventRange.last: return (er[0], None)
    else                                          : return er


def print_every(N):
    counter = count()
    return fl.branch(fl.map  (lambda _: next(counter), args="event_number", out="index"),
                     fl.slice(None, None, N),
                     fl.sink (lambda data: print(f"events processed: {data['index']}, event number: {data['event_number']}")))


def print_every_alternative_implementation(N):
    @fl.coroutine
    def print_every_loop(target):
        with fl.closing(target):
            for i in count():
                data = yield
                if not i % N:
                    print(f"events processed: {i}, event number: {data['event_number']}")
                target.send(data)
    return print_every_loop


def collect():
    """Return a future/sink pair for collecting streams into a list."""
    def append(l,e):
        l.append(e)
        return l
    return fl.reduce(append, initial=[])()


def copy_mc_info(files_in     : List[str],
                 h5out        : tb.File  ,
                 event_numbers: List[int],
                 db_file      :      str ,
                 run_number   :      int ) -> None:
    """
    Copy to an output file the MC info of a list of selected events.

    Parameters
    ----------
    files_in : List of strings
        Name of the input files.
    file_out : tables.File
        The output h5 file.
    event_numbers : List[int]
        List of event numbers for which the MC info is copied
        to the output file.
    """

    writer = mcinfo_io.mc_writer(h5out)

    copied_events = []
    for f in files_in:
        if mcinfo_io.check_mc_present(f):
            event_numbers_in_file = mcinfo_io.get_event_numbers_in_file(f)
            event_numbers_to_copy = np.intersect1d(event_numbers        ,
                                                   event_numbers_in_file)
            mcinfo_io.copy_mc_info(f, writer, event_numbers_to_copy,
                                   db_file, run_number)
            copied_events.extend(event_numbers_to_copy)
        else:
            warnings.warn(f' File does not contain MC tables.\
             Use positve run numbers for data', UserWarning)
            continue
    if len(np.setdiff1d(event_numbers, copied_events)) != 0:
        raise MCEventNotFound(f' Some events not found in MC tables')


def wf_binner(max_buffer: int) -> Callable:
    """
    Returns a function to be used to convert the raw
    input MC sensor info into data binned according to
    a set bin width, effectively
    padding with zeros inbetween the separate signals.

    Parameters
    ----------
    max_buffer : float
                 Maximum event time to be considered in nanoseconds
    """
    def bin_sensors(sensors  : pd.DataFrame,
                    bin_width: float       ,
                    t_min    : float       ,
                    t_max    : float       ) -> Tuple[np.ndarray, pd.Series]:
        return bf.bin_sensors(sensors, bin_width, t_min, t_max, max_buffer)
    return bin_sensors


def signal_finder(buffer_len   : float,
                  bin_width    : float,
                  bin_threshold:   int) -> Callable:
    """
    Decides where there is signal-like
    charge according to the configuration
    and the PMT sum in order to give
    a useful position for buffer selection.
    Currently simple threshold on binned charge.

    Parameters
    ----------
    buffer_len    : float
                    Configured buffer length in mus
    bin_width     : float
                    Sampling width for sensors
    bin_threshold : int
                    PE threshold for selection
    """
    # The stand_off is the minumum number of samples
    # necessary between candidate triggers.
    stand_off = int(buffer_len / bin_width)
    def find_signal(wfs: pd.Series) -> List[int]:
        return bf.find_signal_start(wfs, bin_threshold, stand_off)
    return find_signal


# TODO: consider caching database
def deconv_pmt(dbfile, run_number, n_baseline, selection=None):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist() if selection is None else selection
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        return blr.deconv_pmt(RWF,
                              coeff_c,
                              coeff_blr,
                              pmt_active = pmt_active,
                              n_baseline = n_baseline)
    return deconv_pmt


def get_run_number(h5in):
    if   "runInfo" in h5in.root.Run: return h5in.root.Run.runInfo[0]['run_number']
    elif "RunInfo" in h5in.root.Run: return h5in.root.Run.RunInfo[0]['run_number']

    raise tb.exceptions.NoSuchNodeError(f"No node runInfo or RunInfo in file {h5in}")


class WfType(Enum):
    rwf  = 0
    mcrd = 1


def get_pmt_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.pmtrwf
    elif wf_type is WfType.mcrd: return h5in.root.   pmtrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")

def get_sipm_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.sipmrwf
    elif wf_type is WfType.mcrd: return h5in.root.   sipmrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")


def get_trigger_info(h5in):
    group            = h5in.root.Trigger if "Trigger" in h5in.root else ()
    trigger_type     = group.trigger if "trigger" in group else repeat(None)
    trigger_channels = group.events  if "events"  in group else repeat(None)
    return trigger_type, trigger_channels


def get_event_info(h5in):
    return h5in.root.Run.events


def get_number_of_active_pmts(detector_db, run_number):
    datapmt = load_db.DataPMT(detector_db, run_number)
    return np.count_nonzero(datapmt.Active.values.astype(bool))


def check_nonempty_indices(s1_indices, s2_indices):
    return s1_indices.size and s2_indices.size


def check_empty_pmap(pmap):
    return bool(pmap.s1s) or bool(pmap.s2s)


def length_of(iterable):
    if   isinstance(iterable, tb.table.Table  ): return iterable.nrows
    elif isinstance(iterable, tb.earray.EArray): return iterable.shape[0]
    elif isinstance(iterable, np.ndarray      ): return iterable.shape[0]
    elif isinstance(iterable, NoneType        ): return None
    elif isinstance(iterable, Iterator        ): return None
    elif isinstance(iterable, Sequence        ): return len(iterable)
    elif isinstance(iterable, Mapping         ): return len(iterable)
    else:
        raise TypeError(f"Cannot determine size of type {type(iterable)}")


def check_lengths(*iterables):
    lengths  = map(length_of, iterables)
    nonnones = filter(lambda x: x is not None, lengths)
    if np.any(np.diff(list(nonnones)) != 0):
        raise InvalidInputFileStructure("Input data tables have different sizes")


def mcsensors_from_file(paths     : List[str],
                        db_file   :      str ,
                        run_number:      int ) -> Generator:
    """
    Loads the nexus MC sensor information into
    a pandas DataFrame using the IC function
    load_mcsensor_response_df.
    Returns info event by event as a
    generator in the structure expected by
    the dataflow.

    paths      : List of strings
                 List of input file names to be read
    db_file    : string
                 Name of detector database to be used
    run_number : int
                 Run number for database
    """

    pmt_ids  = load_db.DataPMT(db_file, run_number).SensorID

    for file_name in paths:
        sns_resp = mcinfo_io.load_mcsensor_response_df(file_name              ,
                                                       return_raw = False     ,
                                                       db_file    = db_file   ,
                                                       run_no     = run_number)

        ## MC uses dummy timestamp for now
        ## Only in case of evt splitting will be non zero
        timestamp = 0

        for evt in mcinfo_io.get_event_numbers_in_file(file_name):
            try:
                ## Assumes two types of sensor, all non pmt
                ## assumed to be sipms. NEW, NEXT100 and DEMOPP safe
                ## Flex with this structure too.
                pmt_indx  = sns_resp.loc[evt].index.isin(pmt_ids)
                pmt_resp  = sns_resp.loc[evt][ pmt_indx]
                sipm_resp = sns_resp.loc[evt][~pmt_indx]
            except KeyError:
                pmt_resp = sipm_resp = pd.DataFrame(columns=sns_resp.columns)

            yield dict(event_number = evt      ,
                       timestamp    = timestamp,
                       pmt_resp     = pmt_resp ,
                       sipm_resp    = sipm_resp)


def wf_from_files(paths, wf_type):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            try:
                event_info  = get_event_info  (h5in)
                run_number  = get_run_number  (h5in)
                pmt_wfs     = get_pmt_wfs     (h5in, wf_type)
                sipm_wfs    = get_sipm_wfs    (h5in, wf_type)
                (trg_type ,
                 trg_chann) = get_trigger_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue

            check_lengths(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann)

            for pmt, sipm, evtinfo, trtype, trchann in zip(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann):
                event_number, timestamp         = evtinfo.fetch_all_fields()
                if trtype  is not None: trtype  = trtype .fetch_all_fields()[0]

                yield dict(pmt=pmt, sipm=sipm, run_number=run_number,
                           event_number=event_number, timestamp=timestamp,
                           trigger_type=trtype, trigger_channels=trchann)


def pmap_from_files(paths):
    for path in paths:
        try:
            pmaps = load_pmaps(path)
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue
            except IndexError:
                continue

            check_lengths(event_info, pmaps)

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                yield dict(pmap=pmaps[event_number], run_number=run_number,
                           event_number=event_number, timestamp=timestamp)


def cdst_from_files(paths: List[str]) -> Iterator[Dict[str,Union[pd.DataFrame, MCInfo, int, float]]]:
    """Reader of the files, yields collected hits,
       pandas DataFrame with kdst info, mc_info, run_number, event_number and timestamp"""
    for path in paths:
        try:
            cdst_df    = load_dst (path,   'CHITS', 'lowTh')
            summary_df = load_dst (path, 'Summary', 'Events')
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
                evts, _     = zip(*event_info[:])
                bool_mask   = np.in1d(evts, cdst_df.event.unique())
                event_info  = event_info[bool_mask]
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue
            check_lengths(event_info, cdst_df.event.unique())
            for evtinfo in event_info:
                event_number, timestamp = evtinfo
                yield dict(cdst    = cdst_df   .loc[cdst_df   .event==event_number],
                           summary = summary_df.loc[summary_df.event==event_number],
                           run_number=run_number,
                           event_number=event_number, timestamp=timestamp)
            # NB, the monte_carlo writer is different from the others:
            # it needs to be given the WHOLE TABLE (rather than a
            # single event) at a time.

def hits_and_kdst_from_files(paths: List[str]) -> Iterator[Dict[str,Union[HitCollection, pd.DataFrame, MCInfo, int, float]]]:
    """Reader of the files, yields HitsCollection, pandas DataFrame with
    kdst info, run_number, event_number and timestamp."""
    for path in paths:
        try:
            hits_df = load_dst (path, 'RECO', 'Events')
            kdst_df = load_dst (path, 'DST' , 'Events')
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            check_lengths(event_info, hits_df.event.unique())

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                hits = hits_from_df(hits_df.loc[hits_df.event == event_number])
                yield dict(hits = hits[event_number],
                           kdst = kdst_df.loc[kdst_df.event==event_number],
                           run_number = run_number,
                           event_number = event_number,
                           timestamp = timestamp)


def sensor_data(path, wf_type):
    with tb.open_file(path, "r") as h5in:
        if   wf_type is WfType.rwf :   (pmt_wfs, sipm_wfs) = (h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf)
        elif wf_type is WfType.mcrd:   (pmt_wfs, sipm_wfs) = (h5in.root.    pmtrd ,   h5in.root.    sipmrd )
        else                       :   raise TypeError(f"Invalid WfType: {type(wf_type)}")
        _, NPMT ,  PMTWL =  pmt_wfs.shape
        _, NSIPM, SIPMWL = sipm_wfs.shape
        return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)

####### Transformers ########

def build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
               s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
               s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2):
    s1_params = dict(time        = minmax(min = s1_tmin,
                                          max = s1_tmax),
                    length       = minmax(min = s1_lmin,
                                          max = s1_lmax),
                    stride       = s1_stride,
                    rebin_stride = s1_rebin_stride)

    s2_params = dict(time        = minmax(min = s2_tmin,
                                          max = s2_tmax),
                    length       = minmax(min = s2_lmin,
                                          max = s2_lmax),
                    stride       = s2_stride,
                    rebin_stride = s2_rebin_stride)

    datapmt = load_db.DataPMT(detector_db, run_number)
    pmt_ids = datapmt.SensorID[datapmt.Active.astype(bool)].values

    def build_pmap(ccwf, s1_indx, s2_indx, sipmzs): # -> PMap
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            s1_params, s2_params, thr_sipm_s2, pmt_ids,
                            pmt_samp_wid, sipm_samp_wid)

    return build_pmap


def calibrate_pmts(dbfile, run_number, n_MAU, thr_MAU):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    adc_to_pes = np.abs(DataPMT.adc_to_pes.values)
    adc_to_pes = adc_to_pes[adc_to_pes > 0]

    def calibrate_pmts(cwf):# -> CCwfs:
        return csf.calibrate_pmts(cwf,
                                  adc_to_pes = adc_to_pes,
                                  n_MAU      = n_MAU,
                                  thr_MAU    = thr_MAU)
    return calibrate_pmts


def calibrate_sipms(dbfile, run_number, thr_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)

    def calibrate_sipms(rwf):
        return csf.calibrate_sipms(rwf,
                                   adc_to_pes = adc_to_pes,
                                   thr        = thr_sipm,
                                   bls_mode   = csf.BlsMode.mode)

    return calibrate_sipms


def calibrate_with_mean(dbfile, run_number):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mean(wfs):
        return csf.subtract_baseline_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mean

def calibrate_with_mau(dbfile, run_number, n_mau_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mau(wfs):
        return csf.subtract_baseline_mau_and_calibrate(wfs, adc_to_pes, n_mau_sipm)
    return calibrate_with_mau


def zero_suppress_wfs(thr_csum_s1, thr_csum_s2):
    def ccwfs_to_zs(ccwf_sum, ccwf_sum_mau):
        return (pkf.indices_and_wf_above_threshold(ccwf_sum_mau, thr_csum_s1).indices,
                pkf.indices_and_wf_above_threshold(ccwf_sum    , thr_csum_s2).indices)
    return ccwfs_to_zs


def compute_pe_resolution(rms, adc_to_pes):
    return np.divide(rms                              ,
                     adc_to_pes                       ,
                     out   = np.zeros_like(adc_to_pes),
                     where = adc_to_pes != 0          )


def simulate_sipm_response(detector, run_number, wf_length, noise_cut, filter_padding):
    datasipm      = load_db.DataSiPM (detector, run_number)
    baselines     = load_db.SiPMNoise(detector, run_number)[-1]
    noise_sampler = NoiseSampler(detector, run_number, wf_length, True)

    adc_to_pes    = datasipm.adc_to_pes.values
    thresholds    = noise_cut * adc_to_pes + baselines
    single_pe_rms = datasipm.Sigma.values.astype(np.double)
    pe_resolution = compute_pe_resolution(single_pe_rms, adc_to_pes)

    def simulate_sipm_response(sipmrd):
        wfs = sf.simulate_sipm_response(sipmrd, noise_sampler, adc_to_pes, pe_resolution)
        return wfm.noise_suppression(wfs, thresholds, filter_padding)
    return simulate_sipm_response


####### Filters ########

def peak_classifier(**params):
    selector = S12Selector(**params)
    return partial(pmap_filter, selector)


def compute_xy_position(dbfile, run_number, **reco_params):
    # `reco_params` is the set of parameters for the corona
    # algorithm either for the full corona or for barycenter
    datasipm = load_db.DataSiPM(dbfile, run_number)

    def compute_xy_position(xys, qs):
        return corona(xys, qs, datasipm, **reco_params)
    return compute_xy_position


def compute_z_and_dt(t_s2, t_s1, drift_v):
    dt  = t_s2 - np.array(t_s1)
    z   = dt * drift_v
    dt *= units.ns / units.mus
    return z, dt


def build_pointlike_event(dbfile, run_number, drift_v,
                          reco, charge_type = SiPMCharge.raw):
    datasipm   = load_db.DataSiPM(dbfile, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise

    def build_pointlike_event(pmap, selector_output, event_number, timestamp):
        evt = KrEvent(event_number, timestamp * 1e-3)

        evt.nS1 = 0
        for passed, peak in zip(selector_output.s1_peaks, pmap.s1s):
            if not passed: continue

            evt.nS1 += 1
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.time_at_max_energy)

        evt.nS2 = 0

        for passed, peak in zip(selector_output.s2_peaks, pmap.s2s):
            if not passed: continue

            evt.nS2 += 1
            evt.S2w.append(peak.width / units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.time_at_max_energy)

            xys = sipm_xys[peak.sipms.ids           ]
            qs  = peak.sipm_charge_array(sipm_noise, charge_type,
                                         single_point = True)
            try:
                clusters = reco(xys, qs)
            except XYRecoFail:
                c    = NNN()
                Z    = tuple(NN for _ in range(0, evt.nS1))
                DT   = tuple(NN for _ in range(0, evt.nS1))
                Zrms = NN
            else:
                c = clusters[0]
                Z, DT = compute_z_and_dt(evt.S2t[-1], evt.S1t, drift_v)
                Zrms  = peak.rms / units.mus

            evt.Nsipm.append(c.nsipm)
            evt.S2q  .append(c.Q)
            evt.X    .append(c.X)
            evt.Y    .append(c.Y)
            evt.Xrms .append(c.Xrms)
            evt.Yrms .append(c.Yrms)
            evt.R    .append(c.R)
            evt.Phi  .append(c.Phi)
            evt.DT   .append(DT)
            evt.Z    .append(Z)
            evt.Zrms .append(Zrms)

        return evt

    return build_pointlike_event


def hit_builder(dbfile, run_number, drift_v, reco,
                rebin_slices, rebin_method,
                charge_type = SiPMCharge.raw):
    datasipm = load_db.DataSiPM(dbfile, run_number)
    sipm_xs  = datasipm.X.values
    sipm_ys  = datasipm.Y.values
    sipm_xys = np.stack((sipm_xs, sipm_ys), axis=1)

    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise

    barycenter = partial(corona,
                         all_sipms      =  datasipm,
                         Qthr           =  0 * units.pes,
                         Qlm            =  0 * units.pes,
                         lm_radius      = -1 * units.mm,
                         new_lm_radius  = -1 * units.mm,
                         msipm          =  1)

    def empty_cluster():
        return Cluster(NN, xy(0,0), xy(0,0), 0)

    def build_hits(pmap, selector_output, event_number, timestamp):
        hitc = HitCollection(event_number, timestamp * 1e-3)

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.
        if np.any(selector_output.s1_peaks):
            first_s1 = np.where(selector_output.s1_peaks)[0][0]
            s1_t     = pmap.s1s[first_s1].time_at_max_energy
        else:
            first_s2 = np.where(selector_output.s2_peaks)[0][0]
            s1_t     = pmap.s2s[first_s2].times[0]

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_slices, rebin_method)

            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)
            try              : cluster = barycenter(xys, qs)[0]
            except XYRecoFail: xy_peak = xy(NN, NN)
            else             : xy_peak = xy(cluster.X, cluster.Y)

            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)
            for slice_no, (t_slice, qs) in enumerate(zip(peak.times ,
                                                         sipm_charge)):
                z_slice = (t_slice - s1_t) * units.ns * drift_v
                e_slice = peak.pmts.sum_over_sensors[slice_no]
                try:
                    xys      = sipm_xys[peak.sipms.ids]
                    clusters = reco(xys, qs)
                    es       = hif.split_energy(e_slice, clusters)
                    for c, e in zip(clusters, es):
                        hit       = Hit(peak_no, c, z_slice, e, xy_peak)
                        hitc.hits.append(hit)
                except XYRecoFail:
                    hit = Hit(peak_no, empty_cluster(), z_slice,
                              e_slice, xy_peak)
                    hitc.hits.append(hit)

        return hitc
    return build_hits


def waveform_binner(bins):
    def bin_waveforms(wfs):
        return cf.bin_waveforms(wfs, bins)
    return bin_waveforms


def waveform_integrator(limits):
    def integrate_wfs(wfs):
        return cf.spaced_integrals(wfs, limits)[:, ::2]
    return integrate_wfs


# Compound components
def compute_and_write_pmaps(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                  s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                  s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                  h5out, compression, sipm_rwf_to_cal=None):

    # Filter events without signal over threshold
    indices_pass    = fl.map(check_nonempty_indices,
                             args = ("s1_indices", "s2_indices"),
                             out = "indices_pass")
    empty_indices   = fl.count_filter(bool, args = "indices_pass")

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                                         s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")

    # Filter events with zero peaks
    pmaps_pass      = fl.map(check_empty_pmap, args = "pmap", out = "pmaps_pass")
    empty_pmaps     = fl.count_filter(bool, args = "pmaps_pass")

    # Define writers...
    write_pmap_         = pmap_writer        (h5out,                compression=compression)
    write_indx_filter_  = event_filter_writer(h5out, "s12_indices", compression=compression)
    write_pmap_filter_  = event_filter_writer(h5out, "empty_pmap" , compression=compression)

    # ... and make them sinks
    write_pmap         = sink(write_pmap_        , args=(        "pmap", "event_number"))
    write_indx_filter  = sink(write_indx_filter_ , args=("event_number", "indices_pass"))
    write_pmap_filter  = sink(write_pmap_filter_ , args=("event_number",   "pmaps_pass"))

    fn_list = (indices_pass,
               fl.branch(write_indx_filter),
               empty_indices.filter,
               sipm_rwf_to_cal,
               compute_pmap,
               pmaps_pass,
               fl.branch(write_pmap_filter),
               empty_pmaps.filter,
               fl.branch(write_pmap))

    # Filter out simp_rwf_to_cal if it is not set
    compute_pmaps = pipe(*filter(None, fn_list))

    return compute_pmaps, empty_indices, empty_pmaps




def copy_Ec_to_Ep_hit_attribute_(hitc : HitCollection) -> HitCollection:
    """
    The functions copies values of Ec attributes into Ep attributes. Takes as input HitCollection and returns a copy.
    """
    mod_hits = []
    for hit in hitc.hits:
        hit = Hit(hit.npeak, Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm),
                  hit.Z, hit.E, xy(hit.Xpeak, hit.Ypeak), s2_energy_c=hit.Ec, Ep=hit.Ec)
        mod_hits.append(hit)
    mod_hitc = HitCollection(hitc.event, hitc.time, hits=mod_hits)
    return mod_hitc


types_dict_summary = OrderedDict({'event'     : np.int32  , 'evt_energy' : np.float64, 'evt_charge'    : np.float64,
                                  'evt_ntrks' : np.int    , 'evt_nhits'  : np.int    , 'evt_x_avg'     : np.float64,
                                  'evt_y_avg' : np.float64, 'evt_z_avg'  : np.float64, 'evt_r_avg'     : np.float64,
                                  'evt_x_min' : np.float64, 'evt_y_min'  : np.float64, 'evt_z_min'     : np.float64,
                                  'evt_r_min' : np.float64, 'evt_x_max'  : np.float64, 'evt_y_max'     : np.float64,
                                  'evt_z_max' : np.float64, 'evt_r_max'  : np.float64, 'evt_out_of_map': bool      })

def make_event_summary(event_number  : int              ,
                       topology_info : pd.DataFrame     ,
                       paolina_hits  : HitCollection,
                       out_of_map    : bool
                       ) -> pd.DataFrame:
    """
    For a given event number, timestamp, topology info dataframe, paolina hits and kdst information returns a
    dataframe with the whole event summary.

    Parameters
    ----------
    event_number  : int
    timestamp     : long int
    topology_info : DataFrame
        Dataframe containing track information,
        output of track_blob_info_creator_extractor
    paolina_hits  : HitCollection
        Hits table passed through paolina functions,
        output of track_blob_info_creator_extractor
    kdst          : DataFrame
        Kdst information read from penthesilea output


    Returns
    ----------
    DataFrame containing relevant per event information.
    """
    es = pd.DataFrame(columns=list(types_dict_summary.keys()))

    ntrks = len(topology_info.index)
    nhits = len(paolina_hits.hits)

    S2ec = sum(h.Ec for h in paolina_hits.hits)
    S2qc = -1 #not implemented yet

    pos   = [h.pos for h in paolina_hits.hits]
    x, y, z = map(np.array, zip(*pos))
    r = np.sqrt(x**2 + y**2)

    e     = [h.Ec  for h in paolina_hits.hits]
    ave_pos = np.average(pos, weights=e, axis=0)
    ave_r   = np.average(r  , weights=e, axis=0)


    list_of_vars  = [event_number, S2ec, S2qc, ntrks, nhits,
                     *ave_pos, ave_r,
                     min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r),
                     out_of_map]

    es.loc[0] = list_of_vars
    #change dtype of columns to match type of variables
    es = es.apply(lambda x : x.astype(types_dict_summary[x.name]))
    return es



def track_writer(h5out, compression='ZLIB4'):
    """
    For a given open table returns a writer for topology info dataframe
    """
    def write_tracks(df):
        return df_writer(h5out              = h5out              ,
                         df                 = df                 ,
                         compression        = compression        ,
                         group_name         = 'Tracking'         ,
                         table_name         = 'Tracks'           ,
                         descriptive_string = 'Track information',
                         columns_to_index   = ['event']          )
    return write_tracks


def summary_writer(h5out, compression='ZLIB4'):
    """
    For a given open table returns a writer for summary info dataframe
    """
    def write_summary(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         compression        = compression                ,
                         group_name         = 'Summary'                  ,
                         table_name         = 'Events'                   ,
                         descriptive_string = 'Event summary information',
                         columns_to_index   = ['event']                  )
    return write_summary


types_dict_tracks = OrderedDict({'event'           : np.int32  , 'trackID'       : np.int    , 'energy'      : np.float64,
                                 'length'          : np.float64, 'numb_of_voxels': np.int    , 'numb_of_hits': np.int    ,
                                 'numb_of_tracks'  : np.int    , 'x_min'         : np.float64, 'y_min'       : np.float64,
                                 'z_min'           : np.float64, 'r_min'         : np.float64, 'x_max'       : np.float64,
                                 'y_max'           : np.float64, 'z_max'         : np.float64, 'r_max'       : np.float64,
                                 'x_ave'           : np.float64, 'y_ave'         : np.float64, 'z_ave'       : np.float64,
                                 'r_ave'           : np.float64, 'extreme1_x'    : np.float64, 'extreme1_y'  : np.float64,
                                 'extreme1_z'      : np.float64, 'extreme2_x'    : np.float64, 'extreme2_y'  : np.float64,
                                 'extreme2_z'      : np.float64, 'blob1_x'       : np.float64, 'blob1_y'     : np.float64,
                                 'blob1_z'         : np.float64, 'blob2_x'       : np.float64, 'blob2_y'     : np.float64,
                                 'blob2_z'         : np.float64, 'eblob1'        : np.float64, 'eblob2'      : np.float64,
                                 'ovlp_blob_energy': np.float64,
                                 'vox_size_x'      : np.float64, 'vox_size_y'    : np.float64, 'vox_size_z'  : np.float64})

def track_blob_info_creator_extractor(vox_size         : [float, float, float],
                                      strict_vox_size  : bool                 ,
                                      energy_threshold : float                ,
                                      min_voxels       : int                  ,
                                      blob_radius      : float                ,
                                      max_num_hits     : int
                                     ) -> Callable:
    """
    For a given paolina parameters returns a function that extract tracks / blob information from a HitCollection.

    Parameters
    ----------
    vox_size         : [float, float, float]
        (maximum) size of voxels for track reconstruction
    strict_vox_size  : bool
        if False allows per event adaptive voxel size,
        smaller of equal thatn vox_size
    energy_threshold : float
        if energy of end-point voxel is smaller
        the voxel will be dropped and energy redistributed to the neighbours
    min_voxels       : int
        after min_voxel number of voxels is reached no dropping will happen.
    blob_radius      : float
        radius of blob

    Returns
    ----------
    A function that from a given HitCollection returns a pandas DataFrame with per track information.
    """
    def create_extract_track_blob_info(hitc):
        df = pd.DataFrame(columns=list(types_dict_tracks.keys()))
        if len(hitc.hits) > max_num_hits:
            return df, hitc, True
        #track_hits is a new Hitcollection object that contains hits belonging to tracks, and hits that couldnt be corrected
        track_hitc = HitCollection(hitc.event, hitc.time)
        out_of_map = np.any(np.isnan([h.Ep for h in hitc.hits]))
        if out_of_map:
            #add nan hits to track_hits, the track_id will be -1
            track_hitc.hits.extend  ([h for h in hitc.hits if np.isnan   (h.Ep)])
            hits_without_nan       = [h for h in hitc.hits if np.isfinite(h.Ep)]
            #create new Hitcollection object but keep the name hitc
            hitc      = HitCollection(hitc.event, hitc.time)
            hitc.hits = hits_without_nan

        if len(hitc.hits) > 0:
            voxels           = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, HitEnergy.Ep)
            (    mod_voxels,
             dropped_voxels) = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)
            tracks           = plf.make_track_graphs(mod_voxels)

            for v in dropped_voxels:
                track_hitc.hits.extend(v.hits)

            vox_size_x = voxels[0].size[0]
            vox_size_y = voxels[0].size[1]
            vox_size_z = voxels[0].size[2]
            del(voxels)
            #sort tracks in energy
            tracks     = sorted(tracks, key=plf.get_track_energy, reverse=True)

            track_hits = []
            for c, t in enumerate(tracks, 0):
                tID = c
                energy = plf.get_track_energy(t)
                length = plf.length(t)
                numb_of_hits   = len([h for vox in t.nodes() for h in vox.hits])
                numb_of_voxels = len(t.nodes())
                numb_of_tracks = len(tracks   )
                pos   = [h.pos for v in t.nodes() for h in v.hits]
                x, y, z = map(np.array, zip(*pos))
                r = np.sqrt(x**2 + y**2)

                e     = [h.Ep for v in t.nodes() for h in v.hits]
                ave_pos = np.average(pos, weights=e, axis=0)
                ave_r   = np.average(r  , weights=e, axis=0)
                extr1, extr2 = plf.find_extrema(t)
                extr1_pos = extr1.XYZ
                extr2_pos = extr2.XYZ

                blob_pos1, blob_pos2 = plf.blob_centres(t, blob_radius)

                e_blob1, e_blob2, hits_blob1, hits_blob2 = plf.blob_energies_and_hits(t, blob_radius)
                overlap = float(sum(h.Ep for h in set(hits_blob1).intersection(set(hits_blob2))))
                list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels,
                                numb_of_hits, numb_of_tracks,
                                min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r),
                                *ave_pos, ave_r, *extr1_pos,
                                *extr2_pos, *blob_pos1, *blob_pos2,
                                e_blob1, e_blob2, overlap,
                                vox_size_x, vox_size_y, vox_size_z]

                df.loc[c] = list_of_vars

                for vox in t.nodes():
                    for hit in vox.hits:
                        hit.track_id = tID
                        track_hits.append(hit)

            #change dtype of columns to match type of variables
            df = df.apply(lambda x : x.astype(types_dict_tracks[x.name]))
            track_hitc.hits.extend(track_hits)
        return df, track_hitc, out_of_map

    return create_extract_track_blob_info


def compute_and_write_tracks_info(paolina_params, h5out):

    # Create tracks and compute topology-related information
    create_extract_track_blob_info = fl.map(track_blob_info_creator_extractor(**paolina_params),
                                            args = 'Ep_hits',
                                            out  = ('topology_info', 'paolina_hits', 'out_of_map'))

    # Filter empty topology events
    filter_events_topology         = fl.map(lambda x : len(x) > 0,
                                            args = 'topology_info',
                                            out  = 'topology_passed')
    events_passed_topology         = fl.count_filter(bool, args="topology_passed")

    # Create table with summary information
    make_final_summary             = fl.map(make_event_summary,
                                            args = ('event_number', 'topology_info', 'paolina_hits', 'out_of_map'),
                                            out  = 'event_info')


    # Define writers and make them sinks
    write_tracks          = fl.sink(   track_writer     (h5out=h5out)             , args="topology_info"      )
    write_summary         = fl.sink( summary_writer     (h5out=h5out)             , args="event_info"         )
    write_topology_filter = fl.sink( event_filter_writer(h5out, "topology_select"), args=("event_number", "topology_passed"    ))


    fn_list = (create_extract_track_blob_info              ,
               filter_events_topology                      ,
               fl.branch(make_final_summary, write_summary),
               fl.branch(write_topology_filter)            ,
               events_passed_topology.   filter            ,
               fl.branch(write_tracks)                     )

    compute_tracks = pipe(*fn_list)

    return compute_tracks
