"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""

import numpy        as np

from .. core.system_of_units_c import units
from .. evm .ic_containers     import ZsWf
from .. evm .pmaps             import S1
from .. evm .pmaps             import S2
from .. evm .pmaps             import PMap
from .. evm .pmaps             import PMTResponses
from .. evm .pmaps             import SiPMResponses


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf > thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return ZsWf(indices_above_thr, wf_above_thr)


def select_wfs_above_time_integrated_thr(wfs, thr):
    selected_ids = np.where(np.sum(wfs, axis=1) >= thr)[0]
    selected_wfs = wfs[selected_ids]
    return selected_ids, selected_wfs


def split_in_peaks(indices, stride):
    where = np.where(np.diff(indices) > stride)[0]
    return np.split(indices, where + 1)


def select_peaks(peaks, time, length, pmt_sample_f=25*units.ns):
    def is_valid(indices):
        return (time  .contains(indices[ 0] * pmt_sample_f) and
                time  .contains(indices[-1] * pmt_sample_f) and
                length.contains(indices[-1] + 1 - indices[0]))
    return tuple(filter(is_valid, peaks))


def pick_slice_and_rebin(indices, times, widths,
                         wfs, rebin_stride, pad_zeros=False,
                         sipm_pmt_bin_ratio=40):
    slice_ = slice(indices[0], indices[-1] + 1)
    times_  = times [   slice_]
    widths_ = widths[   slice_]
    wfs_    = wfs   [:, slice_]
    if pad_zeros:
        n_miss = indices[0] % sipm_pmt_bin_ratio
        n_wfs  = wfs.shape[0]
        times_  = np.concatenate([np.zeros(        n_miss) ,  times_])
        widths_ = np.concatenate([np.zeros(        n_miss) , widths_])
        wfs_    = np.concatenate([np.zeros((n_wfs, n_miss)),    wfs_], axis=1)
    (times ,
     widths,
     wfs   ) = rebin_times_and_waveforms(times_, widths_, wfs_, rebin_stride)
    return times, widths, wfs


def build_pmt_responses(indices, times, widths, ccwf,
                        pmt_ids, rebin_stride, pad_zeros,
                        sipm_pmt_bin_ratio):
    (pk_times ,
     pk_widths,
     pmt_wfs  ) = pick_slice_and_rebin(indices, times, widths,
                                       ccwf   , rebin_stride,
                                       pad_zeros = pad_zeros,
                                       sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)
    return pk_times, pk_widths, PMTResponses(pmt_ids, pmt_wfs)


def build_sipm_responses(indices, times, widths,
                         sipm_wfs, rebin_stride, thr_sipm_s2):
    _, _, sipm_wfs_ = pick_slice_and_rebin(indices , times, widths,
                                           sipm_wfs, rebin_stride,
                                           pad_zeros = False)
    (sipm_ids,
     sipm_wfs)   = select_wfs_above_time_integrated_thr(sipm_wfs_,
                                                        thr_sipm_s2)
    return SiPMResponses(sipm_ids, sipm_wfs)


def build_peak(indices, times,
               widths, ccwf, pmt_ids,
               rebin_stride,
               with_sipms, Pk,
               pmt_sample_f  = 25 * units.ns,
               sipm_sample_f =  1 * units.mus,
               sipm_wfs      = None,
               thr_sipm_s2   = 0):
    sipm_pmt_bin_ratio = int(sipm_sample_f/pmt_sample_f)
    (pk_times ,
     pk_widths,
     pmt_r    ) = build_pmt_responses(indices, times, widths,
                                     ccwf, pmt_ids,
                                     rebin_stride, pad_zeros = with_sipms,
                                     sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)
    if with_sipms:
        sipm_r = build_sipm_responses(indices // sipm_pmt_bin_ratio,
                                      times // sipm_pmt_bin_ratio,
                                      widths * sipm_pmt_bin_ratio,
                                      sipm_wfs,
                                      rebin_stride // sipm_pmt_bin_ratio,
                                      thr_sipm_s2)
    else:
        sipm_r = SiPMResponses.build_empty_instance()

    return Pk(pk_times, pk_widths, pmt_r, sipm_r)


def find_peaks(ccwfs, index,
               time, length,
               stride, rebin_stride,
               Pk, pmt_ids,
               pmt_sample_f =25*units.ns,
               sipm_sample_f= 1*units.mus,
               sipm_wfs=None, thr_sipm_s2=0,
               times_vect = None):
    ccwfs = np.array(ccwfs, ndmin=2)

    peaks           = []
    if not np.array(times_vect).sum():
        times_vect  = np.arange     (ccwfs.shape[1]) * pmt_sample_f
    widths          = np.full       (ccwfs.shape[1],   pmt_sample_f)
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length, pmt_sample_f)
    with_sipms      = Pk is S2 and sipm_wfs is not None

    for indices in selected_splits:
        pk = build_peak(indices, times_vect,
                        widths, ccwfs, pmt_ids,
                        rebin_stride,
                        with_sipms, Pk,
                        pmt_sample_f, sipm_sample_f,
                        sipm_wfs, thr_sipm_s2)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids,
             pmt_sample_f, sipm_sample_f):
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1, pmt_ids=pmt_ids,
                           pmt_sample_f=pmt_sample_f,
                           **s1_params),
                find_peaks(ccwf, s2_indx, Pk=S2, pmt_ids=pmt_ids,
                           sipm_wfs    = sipm_zs_wf,
                           thr_sipm_s2 = thr_sipm_s2,
                           pmt_sample_f  = pmt_sample_f,
                           sipm_sample_f = sipm_sample_f,
                           **s2_params))


def get_diff_pmaps(ccwf_s1, ccwf_s2, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids,
             pmt_sample_f, sipm_sample_f, rebinned_s1_times, rebinned_s2_times):

    #if rebinned_times:
    #s2_params['length'] = s2_params['length'] / s2_params['rebin_stride']
    s1_rebinned_f = pmt_sample_f * s1_params['rebin_stride']
    s2_rebinned_f = pmt_sample_f * s2_params['rebin_stride']
    #s2_params['rebin_stride'] = 1

    return PMap(find_peaks(ccwf_s1, s1_indx, Pk=S1, pmt_ids=pmt_ids,
                           pmt_sample_f  = s1_rebinned_f,
                           sipm_sample_f = sipm_sample_f,
                           times_vect    = rebinned_s1_times,
                           time          = s1_params['time'],
                           length        = s1_params['length'],#/s2_params['rebin_stride'],
                           stride        = s1_params['stride'],
                           rebin_stride  = 1),
                find_peaks(ccwf_s2, s2_indx, Pk=S2, pmt_ids=pmt_ids,
                           sipm_wfs    = sipm_zs_wf,
                           thr_sipm_s2 = thr_sipm_s2,
                           pmt_sample_f  = s2_rebinned_f,
                           sipm_sample_f = sipm_sample_f,
                           times_vect    = rebinned_s2_times,
                           time          = s2_params['time'],
                           length        = s2_params['length'],#/s2_params['rebin_stride'],
                           stride        = s2_params['stride'],
                           rebin_stride  = 1))


def rebin_times_and_waveforms(times, widths, waveforms,
                              rebin_stride=2, slices=None):
    if rebin_stride < 2: return times, widths, waveforms

    if not slices:
        n_bins = int(np.ceil(len(times) / rebin_stride))
        reb    = rebin_stride
        slices = [slice(reb * i, reb * (i + 1)) for i in range(n_bins)]
    n_sensors = waveforms.shape[0]

    rebinned_times  = np.zeros(            len(slices) )
    rebinned_widths = np.zeros(            len(slices) )
    rebinned_wfs    = np.zeros((n_sensors, len(slices)))

    for i, s in enumerate(slices):
        t  = times    [   s]
        e  = waveforms[:, s]
        w  = np.sum(e, axis=0) if np.any(e) else None
        rebinned_times [   i] = np.average(t, weights=w)
        rebinned_widths[   i] = np.sum    (   widths[s])
        rebinned_wfs   [:, i] = np.sum    (e,    axis=1)
    return rebinned_times, rebinned_widths, rebinned_wfs


def rebin_wf(pmt_sample_f, rebin_stride):
    def create_times_widths_and_rebin(wf):
        times  = np.arange(wf.shape[1]) * pmt_sample_f
        widths = np.full  (wf.shape[1],   pmt_sample_f)
        rebinned_parameters = rebin_times_and_waveforms(times,
                                                        widths,
                                                        wf,
                                                        rebin_stride)
        rebinned_times  = rebinned_parameters[0]
        rebinned_widths = rebinned_parameters[1]
        rebinned_wfs    = rebinned_parameters[2]
        return rebinned_times, rebinned_widths, rebinned_wfs

    return create_times_widths_and_rebin
