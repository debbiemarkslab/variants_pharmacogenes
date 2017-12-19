#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ks_2samp, sem, levene
from scipy.stats.mstats import kruskalwallis
from statsmodels.stats.multitest import multipletests
import warnings
import matplotlib.pyplot as plt

###
#
# Author: CPS
# A set of tools for variant data handling
#
###

# Constants
POPULATIONS = ['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN', 'OTH', 'Adj']
COLORS = {"SAS": '#4f7a41',
          "EAS": '#55973b',
          "AMR": '#870606',
          "AFR": '#ffc300',
          "FIN": '#1d55da',
          "NFE": '#426b94',
          "OTH": '#cccccc',
          "Adj": '#d3d3d3',
          "CAP_SAS": '#4f7a41',
          "CAP_EAS": '#55973b',
          "CAP_AMR": '#870606',
          "CAP_AFR": '#ffc300',
          "CAP_FIN": '#1d55da',
          "CAP_NFE": '#426b94',
          "CAP_OTH": '#cccccc',
          "CAP_Adj": '#d3d3d3'}

####################################################################################################


# Statistical Tests
def pairwise_t_test(reduced_dataframe,
                    populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN'],
                    p_value_threshold=0.05):
    """Perform pairwise t-test between all possible pairs of populations."""
    # Build list of all group-pairs for pairwise comparisons
    ethnic_pairs = []
    for i, group1 in enumerate(populations[:-1]):
        for group2 in populations[i + 1:]:
            ethnic_pairs.append((group1, group2))

    # Conduct t-test on each pair
    df = []
    for group1, group2 in ethnic_pairs:
        row = [group1, group2]
        stat, p_val = stats.ttest_ind(reduced_dataframe[group1], reduced_dataframe[group2])
        row += [stat, p_val]
        df.append(row)
    df = pd.DataFrame(df, columns=["group1", "group2", "statistic", "p-value"])

    # Add Bonferroni-adjusted rejection of null hypothesis
    adjusted_p = p_value_threshold / len(ethnic_pairs)
    print "adjusted threshold p-value: ", adjusted_p
    df["p < {:.4f}".format(p_value_threshold)] = df["p-value"] < p_value_threshold
    df["Bonferroni adjusted (p < {:.4f})".format(adjusted_p)] = df["p-value"] < adjusted_p
    return df


def anova(reduced_dataframe, populations_prefix='',
          populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN']):
    """Calculate F-statistic and p-value using one-way ANOVA."""
    populations_data = [reduced_dataframe[populations_prefix + p] for p in populations]
    return stats.f_oneway(*populations_data)


def kruskal_wallis(reduced_dataframe, populations_prefix='',
                   populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN']):
    """Calculate H-statsistic and p-value using Kruskal-Wallis test (non-parametric ANOVA)."""
    populations_data = [reduced_dataframe[populations_prefix + p] for p in populations]
    return kruskalwallis(populations_data)


def levene_test(reduced_dataframe, populations_prefix='',
                populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN'], **kwargs):
    """Calculate W-statsistic and p-value using Levene test (non-parametric ANOVA)."""
    populations_data = [reduced_dataframe[populations_prefix + p] for p in populations]
    return levene(*populations_data, **kwargs)


def pairwise_ks(reduced_dataframe, populations_prefix='',
                populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE'],
                multipletest_method='bonf',
                p_value_threshold=0.05):
    """Perform  pairwise KS test for all populations in list."""
    cols = [populations_prefix + p for p in populations]
    test_results = []
    for i, c in enumerate(cols[:-1]):
        for c2 in cols[i + 1:]:
            t = ks_2samp(reduced_dataframe[c], reduced_dataframe[c2])
            test_results.append([c, c2, t.statistic, t.pvalue])
    test_result = pd.DataFrame(test_results, columns=['group1', 'group2', 'statistic', 'p-value'])

    reject_a, corrected_p_a, alphacSidak, alphacBonf = multipletests(test_result["p-value"],
                                                                     alpha=p_value_threshold,
                                                                     method=multipletest_method)
    print "alphaSidak", alphacSidak, "alphaBonf", alphacBonf
    test_result["corrected p-value"] = pd.Series(corrected_p_a)
    test_result["reject"] = pd.Series(reject_a)
    return test_result


def pairwise_tukey(reduced_dataframe, groups, p_value_threshold=0.05):
    """Perform Tukey Test (with multiple testing) on a dataframe with one column per group."""
    reduced_dataframe = reduced_dataframe[groups]
    reduced_df_unpivot = reduced_dataframe\
        .stack()\
        .reset_index()\
        .rename(columns={"level_1": 'Pop', 0: 'Prob'})
    tukey = pairwise_tukeyhsd(endog=reduced_df_unpivot["Prob"],   # Data
                              groups=reduced_df_unpivot["Pop"],   # Groups
                              alpha=p_value_threshold)            # Significance level
    return tukey


def plot_tukey(tukey_result, highlight=0.035):
    """Plot tukey test results."""
    tukey_result.plot_simultaneous()                              # Plot group confidence intervals
    plt.vlines(x=highlight, ymin=-0.5, ymax=5.5, color="red")     # Mark end of group of interest


def kruskal_wallis_dunn(reduced_dataframe, populations_prefix='',
                        populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE'],
                        to_compare=None, p_value_threshold=0.05, multipletest_method='bonf'):
    """
    Perform Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test.

    Adapted from https://gist.github.com/alimuldal/fbb19b73fa25423f02e8
    Arguments:
    ---------------
    reduced_dataframe: dataframe
        corresponding to k mutually independent samples from
        continuous populations
    populations: names of populations (form together with populations_prefix as column names)
    to_compare: sequence
        tuples specifying the indices of pairs of groups to compare, e.g.
        [(0, 1), (0, 2)] would compare group 0 with 1 & 2. by default, all
        possible pairwise comparisons between groups are performed.
    p_value_threshold: float
        family-wise error rate used for correcting for multiple comparisons
        (see statsmodels.stats.multitest.multipletests for details)
    method: string
        method used to adjust p-values to account for multiple corrections (see
        statsmodels.stats.multitest.multipletests for options)
    Returns:
    ---------------
    H: float
        Kruskal-Wallis H-statistic
    p_omnibus: float
        p-value corresponding to the global null hypothesis that the medians of
        the groups are all equal
    Z_pairs: float array
        Z-scores computed for the absolute difference in mean ranks for each
        pairwise comparison
    p_corrected: float array
        corrected p-values for each pairwise comparison, corresponding to the
        null hypothesis that the pair of groups has equal medians. note that
        these are only meaningful if the global null hypothesis is rejected.
    reject: bool array
        True for pairs where the null hypothesis can be rejected for the given
        p_value_threshold
    Reference:
    ---------------
    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.
    """
    # omnibus test (K-W ANOVA)
    # -------------------------------------------------------------------------
    groups = [reduced_dataframe[populations_prefix + p] for p in populations]
    groups = [np.array(gg) for gg in groups]

    k = len(groups)

    n = np.array([len(gg) for gg in groups])
    if np.any(n < 5):
        warnings.warn("Sample sizes < 5 are not recommended (K-W test assumes "
                      "a chi square distribution)")

    allgroups = np.concatenate(groups)
    N = len(allgroups)
    ranked = stats.rankdata(allgroups)

    # correction factor for ties
    T = stats.tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    # sum of ranks for each group
    j = np.insert(np.cumsum(n), 0, 0)
    R = np.empty(k, dtype=np.float)
    for ii in range(k):
        R[ii] = ranked[j[ii]:j[ii + 1]].sum()

    # the Kruskal-Wallis H-statistic
    H = (12. / (N * (N + 1.))) * ((R ** 2.) / n).sum() - 3 * (N + 1)

    # apply correction factor for ties
    H /= T

    df_omnibus = k - 1
    p_omnibus = stats.chisqprob(H, df_omnibus)

    # multiple comparisons
    # -------------------------------------------------------------------------

    # by default we compare every possible pair of groups
    if to_compare is None:
        to_compare = tuple(combinations(range(k), 2))

    ncomp = len(to_compare)

    Z_pairs = np.empty(ncomp, dtype=np.float)
    p_uncorrected = np.empty(ncomp, dtype=np.float)
    Rmean = R / n

    pop_groups = []

    for pp, (ii, jj) in enumerate(to_compare):
        pop_groups.append([populations[ii], populations[jj]])
        # standardized score
        Zij = (np.abs(Rmean[ii] - Rmean[jj]) /
               np.sqrt((1. / 12.) * N * (N + 1) * (1. / n[ii] + 1. / n[jj])))
        Z_pairs[pp] = Zij

    # corresponding p-values obtained from upper quantiles of the standard
    # normal distribution
    p_uncorrected = stats.norm.sf(Z_pairs) * 2.

    # correction for multiple comparisons
    reject, p_corrected, alphac_sidak, alphac_bonf = multipletests(
        p_uncorrected, method=multipletest_method, alpha=p_value_threshold
    )

    df = pd.DataFrame(pop_groups, columns=["group1", "group2"])
    df["p-value"] = p_uncorrected
    df["corrected p-value"] = p_corrected
    df["reject"] = reject

    print "H-statistic", H, "p-value", p_omnibus
    print "alphaSidak", alphac_sidak, "alphaBonf", alphac_bonf
    return df

####################################################################################################


# Filter variants
def dataframe_filter(df, column, terms):
    """
    Filter a column to those containing at least one of terms.

    :param df: dataframe with the data to be filtered
    :type df: Pandas DataFrame
    :param column: name of column to filter
    :type column: String
    :param terms: terms to keep in dataframe
    :type terms: list of strings
    """
    return df[column].str.contains("|".join(terms), na=False, case=False)


def multi_filter(df, filters, conj_fun=np.logical_or):
    """
    Combine multiple contitions through a numpy logical function.

    :param df: dataframe with the data to be filtered
    :type df: Pandas DataFrame
    :param filters: boolean values for rows to keep and which not to keep
    :type filters: list of boolean arrays
    :param conj_fun: function to use in conjunction
    :type conj_fun: numpy logic function
    """
    return df[conj_fun.reduce(filters)]


def filter_variants(variants,
                    terms={'Consequence': ['transcript_ablation', 'splice_acceptor_variant',
                                           'splice_donor_variant', 'stop_gained',
                                           'frameshift_variant', 'stop_lost',
                                           'start_lost', 'initiator_codon_variant'],
                           'PolyPhen': ['probably_damaging'],
                           'LoF': ['HC']},
                    conj_fun=np.logical_or):
    """
    Filter variant dataframe columns by terms defined as parameter.

    :param variants: dataframe with variant data
    :type variants: Pandas DataFrame
    :param terms: columns and terms to filter in those
    :type terms: dict
    """
    filters = [dataframe_filter(variants, c, t) for c, t in terms.items()]
    return multi_filter(variants, filters, conj_fun)


def add_variud(variants):
    """
    Add variant unique id to each row.

    :param variants: variant to append id to
    """
    variants.loc[:, "var_uid"] = variants.CHROM.astype(str) + "-" + variants.SYMBOL + "-" +\
        variants.POS.astype(str) + "-" +\
        variants.REF.astype(str) + "-" + variants.ALT.astype(str)


def aggregate(variants, agg_funs, groupby_col=['HGNC_ID', 'SYMBOL']):
    """Aggregate a dataframe based on the provided aggregation functions."""
    agg_df = variants.groupby(groupby_col)\
                     .agg(agg_funs)\
                     .reset_index()
    return agg_df


####################################################################################################


# Aggregate variant dataframe
def aggregate_by_gene(variants,
                      populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN', 'OTH', 'Adj'],
                      populations_prefix='COMP.AF_'):
    """
    Aggregate variants at gene level and calculate stats on different subpopulations.

    :param variants: dataframe with variant data
    :type variants: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param populations_prefix: the prefix of the population column names
    :type populations_prefix: str

    :returns: aggregated dataframe with one row per gene
    """
    agg_funs = {'var_uid': {'n_deleterious_variants': 'count'}}
    pop_agg_funs = {k: {'sum': np.sum, 'mean': np.mean, 'max': np.max, 'median': np.median,
                        'P_no_LoF': probability_no_del_gene}
                    for k in ['AF'] + [populations_prefix + p for p in populations]}

    agg_funs.update(pop_agg_funs)
    summed_variants = aggregate(variants, agg_funs).sort_values(by=('AF', 'sum'), ascending=False)

    return summed_variants


def aggregate_accross_genes(variants, label2gene,
                            populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN', 'OTH', 'Adj'],
                            populations_prefix='COMP.AF_',
                            p_col='P_no_LoF',
                            detailed=True):
    """
    Aggregate the variants dataframe by labels (e.g. drugs).

    :param variants: dataframe with single row per gene
    :type variants: Pandas DataFrame with MultiIndex
    :param label2gene: mapping between gene and label (can contain mutle labels)
    :type label2gene: Pandas DataFrame with MultiIndex
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param populations_prefix: the prefix of the population column names
    :type populations_prefix: str
    :param p_col: column name of column with probability of interest
    :type p_col: str
    """
    if detailed:
        p_cols = [('AF', p_col)] + [(populations_prefix + p, p_col) for p in populations]
    else:
        p_cols = [(populations_prefix + p, p_col) for p in populations]

    aggs = {'P_var_in_any': probability_one_allele_in_any_gene}
    if detailed:
        aggs = aggs.update({'mean': np.mean, 'max': np.max,
                            'median': np.median,
                            'P_var_in_all': probability_one_allele_in_all_gene})
    p_agg = {p: aggs for p in p_cols}
    agg_funs = {('SYMBOL', ''): {'n_targets': lambda x: len(set(x)),
                                 'targets': lambda x: ", ".join(set(x))},
                ('var_uid', 'n_deleterious_variants'): {'n_variants': np.sum}}
    agg_funs.update(p_agg)

    drugs_lof = variants\
        .merge(label2gene, left_on=('HGNC_ID',), right_on=('hgnc_id', ))

    drugs_probs = drugs_lof.groupby(['name', 'midrug_id'])\
        .agg(agg_funs)\
        .sort_values(by=[(populations_prefix + 'Adj', p_col, 'P_var_in_any')], ascending=False)
    return drugs_probs


def add_pseudocounts(variants,
                     populations=['AFR', 'AMR', 'EAS', 'FIN', 'NFE', 'OTH', 'SAS'],
                     adjusted_counts="Adj",
                     ac_col_prefix='AC_',
                     an_col_prefix='AN_',
                     af_col_prefix='COMP.AF_',
                     prior=1):
    """Add pseudocount to allele count for each populations and recalculate allele frequencies."""
    v = variants.copy()
    for p in populations:
        v[af_col_prefix + p] = v[ac_col_prefix + p] / v[an_col_prefix + p]
        v["pc_" + af_col_prefix + p] = (v[ac_col_prefix + p] + prior) / (v[an_col_prefix + p] + prior)

    if adjusted_counts:
        v[af_col_prefix + adjusted_counts] = v[ac_col_prefix + adjusted_counts] / \
            v[an_col_prefix + adjusted_counts]
        total = len(populations) * prior
        v["pc_" + af_col_prefix + adjusted_counts] = (v[ac_col_prefix + adjusted_counts] + total) / \
            (v[an_col_prefix + adjusted_counts] + total)
    return v


def add_allele_probabilities(agg_vars,
                             populations=['AFR', 'AMR', 'EAS', 'FIN', 'NFE', 'OTH', 'SAS', 'Adj'],
                             populations_prefix='COMP.AF_',
                             p_col='P_no_LoF'):
    """
    Add additional annotations to the aggregated dataframe.

    :param agg_vars: dataframe with single entry per gene
    :type agg_vars: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param populations_prefix: the prefix of the population column names
    :type populations_prefix: str
    :param p_col: column name of column with probability of interest
    :type p_col: str
    """
    for (af_col, p_col) in [('AF', p_col)] + [(populations_prefix + p, p_col) for p in populations]:
        agg_vars.loc[:, (af_col, 'P_min_one_LoF')] = agg_vars.apply(invert_probability,
                                                                    axis=1,
                                                                    args=((af_col, p_col),))
        agg_vars.loc[:, (af_col, 'odds_no_LoF')] = agg_vars.apply(odds_from_probability,
                                                                  axis=1,
                                                                  args=((af_col, p_col),))
        agg_vars.loc[:, (af_col, 'odds_min_one_LoF')] = 1.0 / agg_vars.loc[:, (af_col, 'odds_no_LoF')]
        agg_vars.loc[:, (af_col, 'log_odds_no_LoF')] = agg_vars.apply(log_odds,
                                                                      axis=1,
                                                                      args=((af_col, 'odds_no_LoF'),))
        agg_vars.loc[:, (af_col, 'log_odds_min_one_LoF')] = agg_vars.apply(log_odds,
                                                                           axis=1,
                                                                           args=((af_col,
                                                                                  'odds_min_one_LoF'),))


def summarize_allele_probabilities(agg_vars,
                                   populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE'],
                                   populations_prefix='COMP.AF_',
                                   p_col='P_min_one_LoF'):
    """
    Summarize probability annotations of the aggregated dataframe.

    :param agg_vars: dataframe with single entry per gene
    :type agg_vars: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param populations_prefix: the prefix of the population column names
    :type populations_prefix: str
    :param p_col: column name of column with probability of interest
    :type p_col: str
    """
    p_cols = [(populations_prefix + p, p_col) for p in populations]
    agg_vars.loc[:, (p_col, 'min')] = agg_vars.apply(summarize_cols,
                                                     axis=1,
                                                     args=(np.min, p_cols))
    agg_vars.loc[:, (p_col, 'max')] = agg_vars.apply(summarize_cols,
                                                     axis=1,
                                                     args=(np.max, p_cols))
    agg_vars.loc[:, (p_col, 'risk_difference')] = agg_vars.loc[:, (p_col, 'max')] -\
        agg_vars.loc[:, (p_col, 'min')]
    agg_vars.loc[:, (p_col, 'log_risk_ratio')] = agg_vars.apply(max_log_odds_ratio,
                                                                axis=1,
                                                                args=p_cols)
    agg_vars.loc[:, (p_col, 'max_pop')] = agg_vars.apply(find_pop,
                                                         axis=1,
                                                         args=((p_col, 'max'), p_cols))
    agg_vars.loc[:, (p_col, 'min_pop')] = agg_vars.apply(find_pop,
                                                         axis=1,
                                                         args=((p_col, 'min'), p_cols))
    agg_vars.loc[:, (p_col, 'pop')] = agg_vars[(p_col, 'max_pop')].apply(
        lambda x: x[0][0].replace(populations_prefix, ''))

    agg_vars.loc[:, (p_col, 'min pop')] = agg_vars[(p_col, 'min_pop')].apply(
        lambda x: x[0][0].replace(populations_prefix, ''))


def summarize_drug_probabilities(drugs_probs,
                                 populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE', ],
                                 p_col_temp=("COMP.AF_{}", "P_no_LoF", "P_var_in_any")):
    """
    Summarize aggregated probabilities across multiple groups.

    :param drugs_probs: aggregated dataframe
    :type drug_probs: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param p_col_temp: template for columns to include in calculation (population will be filled in from popolations list)
    :type p_col_temp: tuple

    """
    p_cols = [tuple([s.format(p) for s in p_col_temp]) for p in populations]
    drugs_probs.loc[:, p_col_temp[1:] + ('max',)] = drugs_probs.apply(summarize_cols,
                                                                      axis=1,
                                                                      args=(np.max, p_cols))
    drugs_probs.loc[:, p_col_temp[1:] + ('min',)] = drugs_probs.apply(summarize_cols,
                                                                      axis=1,
                                                                      args=(np.min, p_cols))
    drugs_probs.loc[:, p_col_temp[1:] + ('risk_difference',)] = \
        drugs_probs.loc[:, p_col_temp[1:] + ('max',)] - drugs_probs.loc[
            :, p_col_temp[1:] + ('min',)]
    drugs_probs.loc[:, p_col_temp[1:] + ('log_risk_ratio',)] = drugs_probs.apply(
        max_log_odds_ratio,
        axis=1,
        args=p_cols)

    drugs_probs.loc[:, p_col_temp[1:] + ('max_pop',)] = drugs_probs.apply(
        find_pop,
        axis=1,
        args=(p_col_temp[1:] + ('max',), p_cols))

    # drugs_probs.loc[:, p_col_temp[1:] + ('min_pop',)] = drugs_probs.apply(
    #     find_pop,
    #     axis=1,
    #     args=(p_col_temp[1:] + ('min',), p_cols))


def calculate_mean_pop_difference(drugs_probs,
                                  populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN'],
                                  p_col_temp=("COMP.AF_{}", "P_no_LoF", "P_var_in_any"),
                                  drug_col="name"):
    """
    Calculate the difference between individual populations and all others.

    :param drugs_probs: aggregated dataframe
    :type drug_probs: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param p_col_temp: template for columns to include in calculation (population will be filled in from popolations list)
    :type p_col_temp: tuple
    :param drug_col: name of column to be made index of the output dataframe
    :type drug_col: str or tuple

    :returns: dataframe containing columns for risk difference/ratio for each population. Drug is index.
    """
    drug_props_extended = drugs_probs.copy()

    for p in populations:
        # column names of populations in comparison group
        comp_cols = [tuple([s.format(x) for s in p_col_temp]) for x in populations if x != p]
        # reference population column name
        p_col = tuple([s.format(p) for s in p_col_temp])

        # calculate all pairwise differences as well as summary stats
        new_cols = drug_props_extended.apply(pairwise_diff, axis=1, args=(comp_cols, p_col, p))
        drug_props_extended = pd.concat([drug_props_extended, new_cols], axis=1)

    drug_props_extended.set_index(drug_col, inplace=True)
    return drug_props_extended


def calculate_pop_specific_difference(drugs_probs,
                                      direction='min',
                                      populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN'],
                                      p_col_temp=("COMP.AF_{}", "P_no_LoF", "P_var_in_any"),
                                      drug_col="name"):
    """
    Calculate the difference between individual populations and all others.

    :param drugs_probs: aggregated dataframe
    :type drug_probs: Pandas DataFrame
    :param direction: min or max depending on whether the minimal or maximal value in the comparison group should be picked
    :type direction: str
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param p_col_temp: template for columns to include in calculation (population will be filled in from popolations list)
    :type p_col_temp: tuple
    :param drug_col: name of column to be made index of the output dataframe
    :type drug_col: str or tuple

    :returns: dataframe containing columns for risk difference/ratio for each population. Drug is index.
    """
    drug_props_extended = drugs_probs.copy()

    direction = 'max' if direction is None else direction
    fun = np.max if direction == 'max' else np.min

    for p in populations:
        # column names of populations in comparison group
        comp_cols = [tuple([s.format(x) for s in p_col_temp]) for x in populations if x != p]
        # reference population column name
        p_col = tuple([s.format(p) for s in p_col_temp])
        # output column containing min/max value of comparison population
        out_col = p_col_temp[1:] + ('non_{}_{}'.format(p, direction),)
        # output column containing the risk difference between reference and comparison group
        diff_col = p_col_temp[1:] + ('{}_risk_difference_{}'.format(p, direction),)
        ratio_col = p_col_temp[1:] + ('{}_risk_ratio_{}'.format(p, direction),)
        # add column with min/max value in comparison population

        drug_props_extended.loc[:, out_col] = drug_props_extended.apply(summarize_cols,
                                                                        axis=1,
                                                                        args=(fun, comp_cols))
        # add comparison column
        drug_props_extended.loc[:, diff_col] = \
            drug_props_extended.loc[:, p_col] -\
            drug_props_extended.loc[:, out_col]

        drug_props_extended.loc[:, ratio_col] = drug_props_extended.apply(log_odds_ratio,
                                                                          axis=1,
                                                                          args=(p_col, out_col))

    drug_props_extended.set_index(drug_col, inplace=True)
    return drug_props_extended


####################################################################################################


# Aggregation functions for Pandas dataframes
def probability_no_del_gene(del_allele_frequencies, homozygous=False, diploid=True):
    """
    Calculate combined probability for none of the possible alleles occuring on either chromosome.

    Set diploid flag for consideration of both chromosome copies, e.g. in humans
    """
    if diploid is True:
        return np.prod([(1 - af) ** 2 for af in del_allele_frequencies])

    return np.prod([(1 - af) for af in del_allele_frequencies])


def probability_one_allele_in_any_gene(probabilities_genes):
    """
    Calculate combined probability for a specific variant feature to occur in any of multiple proteins.

    Combined probability for a list of probabilities of an event occuring in at least one of
    multiple proteins

    :param probabilities_genes: a list of probabilities (one for each gene) of the event not occuring.
    :type probabilities_genes: Iterable

    :returns: 1 - probability no variant in all genes
    """
    return 1.0 - np.prod(probabilities_genes)


def probability_one_allele_in_all_gene(probabilities_genes):
    """
    Calculate the combined probability for a specific variant feature to occur in any of multiple proteins.

    Combined probability for a list of probabilities of an event occuring on at least one chromosome
    in each one of multiple proteins.

    :param probabilities_genes: a list of probabilities (one for each gene) of the event not occuring.
    :type probabilities_genes: Iterable

    returns: combined probability of any variants
    """
    return np.prod([1.0 - p_no_lof for p_no_lof in probabilities_genes])


# "Apply" functions for Pandas dataframes
def invert_probability(series, p_column):
    """
    Invert the probability for an event occuring.

    :param series: the row or column object from the dataframe
    :param p_column: column name of the pandas column containing the probability of interest
    """
    return 1.0 - series[p_column]


def odds_from_probability(series, p_column):
    """
    Calculate odds of an event from the probability of that event.

    Odds can be caluclated from probabilities as the ration between the probability of the
    event occuring divided by the probability of the event not occuring.

    :param series: the row or column object from the dataframe
    :param p_colum: column name of the pandas column containing the probability of interest

    :returns: odds ratio derived from input probability
    """
    if series[p_column] == 1.0:
        return np.nan
    return series[p_column] / (1.0 - series[p_column])


def log_odds(series, odds_column):
    """
    Perform logarithmic transform of an odds value.

    :param series: the row or column object from the dataframe
    :param odds_column: the column name of the precomputed odds

    :returns: logarithm of column values
    """
    return np.log(series[odds_column])


def log_odds_ratio(series, odds_col_1, odds_col_2):
    """
    Calculate the log odds ratio for odds from two different groups (e.g. treated vs placebo).

    Here, the positive ratio is computed by enforcing the maximal value from both groups to be
    the numerator and the smaller odds to be the denominator

    :param series: the row or column object from the dataframe
    :param odds_col_1: the column name of the precomputed odds for group 1
    :param odds_col_2: the column name of the precomputed odds for group 2

    :returns: logarithm of ratio between both columns
    """
    col_1 = series[odds_col_1]
    col_2 = series[odds_col_2]
    if min(col_1, col_2) == 0.0:
        return np.inf
    return np.log(max(col_1, col_2) /
                  min(col_1, col_2))


def max_log_odds_ratio(series, *odds_cols):
    """
    Calculate the maximum log odds ratio for odds from multiple groups.

    Here, the positive ratio is computed by enforcing the largest odds from all groups to be
    the numerator and the smalles odds to be the denominator

    :param series: the row or column object from the dataframe
    :param *odds_col: the column names of the precomputed odds for the different groups

    :returns: logarithm of maximal odds ratio
    """
    vals = [series[c] for c in odds_cols]
    no_zero_vals = [v for v in vals if v > 0.0]
    min_odds = np.inf
    if len(no_zero_vals) > 0:
        min_odds = min(no_zero_vals)

    max_odds = max(vals)
    if min_odds == 0.0:
        return max_odds * np.inf
    return np.log(max_odds / min_odds)


def summarize_cols(series, summary_fun, cols):
    """
    Apply a summary function summary_fun to a set of columns.

    :param series: the row or column object from the dataframe
    :param summary_fun: a function that works on a list of values
    :param cols: the column names in the series object to be included
    :type cols: iterable

    :returns: the return value of summary fun applied to columns selected by cols
    """
    vals = [series[c] for c in cols if not np.isnan(series[c])]
    if len(vals) == 0:
        return np.nan
    return summary_fun(vals)


def pairwise_diff(series, cols, ref_col, p):
    """
    Apply a summary functions to a set of columns.

    :param series: the row or column object from the dataframe
    :type series: pandas Series (row)
    :param cols: the column names in the series object to be included
    :type cols: iterable
    :param ref_col: the column name in the series object that corresponds to the reference
    :type ref_col: tuple
    :param p: population name
    :type p: str

    :returns: the return value of summary fun applied to columns selected by cols
    """
    vals = [series[c] for c in cols if not np.isnan(series[c])]
    ref = series[ref_col]
    if len(vals) == 0:
        return pd.Series()

    all_diffs = [ref - x for x in vals]
    min_diff = min(all_diffs)
    max_diff = max(all_diffs)
    mean_diff = np.mean(all_diffs)
    median_diff = np.median(all_diffs)
    std_diff = np.std(all_diffs)
    sem_diff = sem(all_diffs)

    return pd.Series([all_diffs, min_diff, max_diff, mean_diff, median_diff, std_diff, sem_diff],
                     index=[ref_col[1:] + (p + '_all_diffs',),
                            ref_col[1:] + (p + '_min_diff',),
                            ref_col[1:] + (p + '_max_diff',),
                            ref_col[1:] + (p + '_mean_diff',),
                            ref_col[1:] + (p + '_median_diff',),
                            ref_col[1:] + (p + '_std_diff',),
                            ref_col[1:] + (p + '_sem_diff',)])


def find_pop(series, value_col, selection_cols):
    """
    Look up the column names of the columns containing a specific value in a dataframe row.

    This function can be useful to retrieve the column name after summarizing multiple columns
    using summarize_col() using np.min or np.max as an summary function.

    :param series: the row or column object from the dataframe
    :param value_col: the name of the column containing the query values
    :param selection_cols: the names of the column subset that should be searched for the query
        value
    """
    search_value = series[value_col]
    matching_pop = [c for c in selection_cols if series[c] == search_value]
    return matching_pop


def find_pop2(series, value_col, selection_cols, tup_idx=0):
    """
    Look up the column names of the columns containing a specific value in a dataframe row.

    This function can be useful to retrieve the column name after summarizing multiple columns
    using summarize_col() using np.min or np.max as an summary function.

    :param series: the row or column object from the dataframe
    :param value_col: the name of the column containing the query values
    :param selection_cols: the names of the column subset that should be searched for the query
        value
    :param tup_idx: index of multindex tuple to use, default 0
    """
    search_value = series[value_col]
    matching_pop = [c[tup_idx] for c in selection_cols if series[c] == search_value]
    return "|".join(map(str, matching_pop))


def allele_frequency(series, allele_count_col, allele_number_col, homozygous=False):
    """
    Calculate allele frequency from count data of allele and total number of covered alleles.

    :param series: a row in the dataframe
    :param allele_count_col: the column name of the dataframe column containing
        the count of variant allele
    :param allele_number_col: the column name of the dataframe column containing the total number
        of alleles covered at that position (should be roughly the same as 2 * samples for humans)
    :param homozygous: flag to indicate that only homozygous counts are provided
    """
    if homozygous:
        return (2 * series[allele_count_col]) / series[allele_number_col]
    return series[allele_count_col] / series[allele_number_col]
