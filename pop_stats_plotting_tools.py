#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.readwrite import json_graph
import math
import json
import jinja2
from matplotlib_venn import venn3, venn3_circles

from IPython.display import Javascript, HTML
from pop_stats_tools import COLORS


def plot_hist(variants, col=("COMP.AF_Adj", "log_odds_no_LoF"), label='Adj',
              fig=None, figsize=(21, 7),
              bins=10, color=sns.color_palette("Set1", 8)[0], grid=False):
    """Plot histogram of values in column."""
    if not fig:
        fig = plt.figure(figsize=figsize)
    variants.loc[:, col]

    sns.distplot(variants.loc[:, col].dropna(), label=label,
                 bins=bins, hist=True, color=color)
    plt.xlabel(col)
    plt.ylabel("denisty")
    plt.legend()


def plot_subpopulations_hist(variants,
                             populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE', 'OTH', 'Adj'],
                             populations_prefix='COMP.AF_',
                             p_col='P_var_in_any',
                             probability=True,
                             logspace=True,
                             single_plot=True,
                             bins=None,
                             grid=False,
                             ax=None,
                             figsize=(21, 7)):
    """Create a joined histogram of different subpopulations."""
    b = 100

    p_cols = [(populations_prefix + p, p_col) for p in populations]
    if ax is None:
        if single_plot is True:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            plot_cols = 2.0
            fig, axes = plt.subplots(int(math.ceil(len(p_cols) / plot_cols)), int(plot_cols),
                                     figsize=figsize, sharex=True, sharey=True)
            axs = axes.flatten()

    if bins is not None:
        b = bins

    if logspace is True:
        start, end = -10, 50
        if probability is True:
            start = -4
            end = 0
        b = np.logspace(start, end, b)

    for i, c in enumerate(p_cols):
        p = c[0].replace(populations_prefix, '')
        if single_plot is False:
            ax = axs[i]
        variants.hist(c, bins=b, ax=ax, grid=grid, alpha=0.5,
                      label=p,
                      color=COLORS[p])
    if single_plot is False:
        for ax in axs:
            if logspace:
                ax.set_xscale("log")
            ax.set_ylabel("entries")
            ax.set_xlabel(p_col)
            ax.set_title("")
            ax.legend()
            ax.axis('tight')
    else:
        ax.axis('tight')
        if logspace:
            ax.set_xscale("log")
        ax.set_ylabel("entries")
        ax.set_xlabel(p_col)
        ax.set_title("")
        ax.legend()


def add_color_column(variants,
                     pop_col=('P_min_one_LoF', 'pop'),):
    """Add a column that contains the color hex code based on a population."""
    color_col = pop_col[:-1] + ('color',)
    variants.loc[:, color_col] = variants[pop_col].map(COLORS)


def plot_gene_bubbles(variants,
                      populations=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE'],
                      populations_prefix='COMP.AF_',
                      p_col=('P_min_one_LoF', 'log_risk_ratio'),
                      color_col=('P_min_one_LoF', 'color'),
                      size_col=('var_uid', 'n_deleterious_variants'),
                      label_col=('SYMBOL', ),
                      max_rank=75,
                      xlabel="Genes",
                      ylabel="Risk Ratio",
                      ax=None,
                      figsize=(15, 7.5),
                      dpi=150):
    """
    Plot.

    :param variants: Variant dataframe
    :type variants: Pandas DataFrame
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param populations_prefix: the prefix of the population column names
    :type populations_prefix: str
    :param p_col: column name of column with probability of interest
    :type p_col: str
    """
    subset = variants.loc[variants[p_col] < np.inf]
    subset.sort_values(by=p_col, inplace=True, ascending=False)
    subset.loc[:, 'i'] = subset.reset_index().index
    if max_rank:
        subset = subset[:max_rank]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if color_col is None:
        color_setting = "grey"
    else:
        color_setting = subset.get(color_col, 'grey')

    subset.plot('i', p_col,
                kind='scatter',
                marker='o',
                ax=ax,
                s=subset[size_col],
                alpha=0.9,
                c=color_setting)

    ax.set_xlim(-1, len(subset))

    plt.sca(ax)
    plt.xticks(subset['i'],
               [str(i) for i in subset[label_col]],
               rotation='vertical')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Legend
    markers = [(k, plt.Line2D([0, 0], [0, 0], color=COLORS[k], marker='o', linestyle=''))
               for k in populations]
    plt.legend(zip(*markers)[1], zip(*markers)[0], numpoints=1)


def make_pop_specific_barplot(drugs_probs_extended,
                              direction='min',
                              populations=['AFR', 'AMR', 'EAS', 'SAS', 'NFE', 'FIN'],
                              p_col_temp=("COMP.AF_{}", "P_no_LoF", "P_var_in_any"),
                              max_rank=10,
                              error=("P_no_LoF", "P_var_in_any", "{}_std_diff"),
                              extreme=("P_no_LoF", "P_var_in_any", "{}_min_diff"),
                              ascending=False,
                              color_dict=COLORS,
                              figsize=(20, 20),
                              dpi=300,
                              ylim=(0.0, 1.0),
                              axes=None,
                              title=None):
    """
    Plot population specific largest difference.

    Requires specific dataframe containing the risk ratio or risk difference specific for
    each population of interest.

    :param drugs_probs_extended: Variant dataframe containing population specific differences
    :type drugs_probs_extended: Pandas DataFrame
    :param direction: min or max depending on whether the minimal or maximal value in the comparison group should be picked
    :type direction: str
    :param populations: all subpopulations to include in aggregation
    :type populations: list
    :param p_col_temp: template for columns to include in calculation (population will be filled in from popolations list)
    :type p_col_temp: tuple
    :param max_rank: maximal number of rows to plot
    :type max_rank: int
    :param error: template of error bar column
    :type error: tuple or None
    :param ascending: Direction for sorting the dataframe prior to plotting
    :type ascending: bool
    :param color_dict: colors to use for plotting
    :type color_dict: dictionary
    :param figsize: matplotlib figure size
    :type figsize: tuple
    :param dpi: figure resolution
    :type dpi: int


    """
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi, sharey=True)

    for i, ax in enumerate(axes.reshape(-1)):
        p = populations[i]
        plot_col = p_col_temp[1:] + ('{}_{}_diff'.format(p, direction),)
        sub_df = drugs_probs_extended.sort_values(by=plot_col, ascending=ascending).head(max_rank)

        # handle error bar columns
        e = None
        if error:
            error_col = tuple([s.format(p) for s in error])
            e = sub_df[error_col].tolist()

        # if extreme: plot extreme value also as dot
        if extreme:
            extreme_col = tuple([s.format(p) for s in extreme])
            sub_df[[extreme_col]].plot(ax=ax, legend=False,
                                       color=color_dict.get(p, 'grey'),
                                       linestyle='None', marker='o')

        # plot bar chart
        sub_df[[plot_col]].plot.bar(yerr=e, ax=ax, legend=False,
                                    color=color_dict.get(p, 'grey'))

        if title:
            ax.set_title(p)

        ax.set_ylim(ylim)
        ax.set_xlabel('')
        # ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

    sns.despine()
    # plt.tight_layout()


def annotate_plot(pgx_genes, established_genes, other_highlight, ax=None,
                  highlight_color=sns.xkcd_rgb["dark red"]):
    """
    Add additional annotations.

    :param pgx_genes: list of x-labels with known PGX annotation (e.g. gene symbol or drug name)
    :param established_genes: list of x-labels that are annotated drug targets ()
    :param other_highlight: other list to annotate (e.g. targets/names of top 100 drugs)
    :param highlight_color: color to use for highlighting gene symbols
    """
    if ax is None:
        ax = plt.gca()
    ticks = ax.get_xticklabels()
    for i, t in enumerate(ticks):
        # annotate genes that have data in PGx
        if t.get_text() in pgx_genes:
            t.set_color(highlight_color)
        # annotate genes that are established targets
        if established_genes:
            if t.get_text() in established_genes:
                pos = (i, ax.get_ylim()[0])
                ax.annotate('*', xy=pos, xycoords='data',
                            xytext=(-4, 0), textcoords='offset points',
                            weight='bold', size=20)
        # annotate genes that are targets in top 100 drugs
        if other_highlight:
            if t.get_text() in other_highlight:
                pos = (i, ax.get_ylim()[0])
                ax.annotate('.', xy=pos, xycoords='data',
                            xytext=(-2, 0), textcoords='offset points',
                            weight='bold', size=20)


def plot_probability_heatmap(df, ax=None, figsize=(10, 20),
                             cmap=None, discrete_cmap=True,
                             color_groups=False, group_cmaps=None,
                             groups=['AFR', 'AMR', 'EAS', 'SAS', 'FIN', 'NFE'],
                             cluster=True):
    from matplotlib import colors

    if cmap is None:
        cmap = "Blues"

    if group_cmaps is None:
        group_cmaps = {
            p: {'cmap': sns.light_palette(COLORS.get(p, "blue"), as_cmap=True),
                'pal': sns.light_palette(COLORS.get(p, "blue"))}
            for p in groups}

    if discrete_cmap:
        cmap = colors.ListedColormap(sns.color_palette(cmap, 10))

    if color_groups:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        for p in groups:
            c = group_cmaps[p]['cmap']
            if discrete_cmap:
                c = colors.ListedColormap(sns.color_palette(group_cmaps[p]['pal'], 10))
            sns.heatmap(df.mask(df.isin(df[p]) != 1), ax=ax, cmap=c, cbar=False)
        return

    if cluster:
        sns.clustermap(df, figsize=figsize, cmap=cmap)
        return
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df, ax=ax, cmap=cmap)
        return


def plot_probability_boxplot(df, ax=None, figsize=(5, 5),
                             color_palette=COLORS, xlabel="Geographic Ancestry",
                             ylabel="", yrange=(0, 1.0)):
    """Plot colored boxplot for probability distribution across multiple populations."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(data=df, ax=ax, fliersize=0, palette=color_palette)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(yrange)


def tree_from_df(drugs_p_lof, drug_name_col="name", target_name_col=('SYMBOL', '', 'targets'),
                 drug_size_col=('P_no_LoF', 'P_var_in_any', 'risk_difference'),
                 gene_p_lof=None, gene_name_col=('SYMBOL'),
                 gene_size_col=('P_min_one_LoF', 'risk_difference'),
                 multiindex=True):
    """Create networkx tree from the two grouped dataframes containing LoF probabilities."""
    if gene_p_lof is None:
        print 'Please specify the dataframe containing lower level probabilities.'
        return

    nx_tree = nx.DiGraph()
    nx_tree.add_node(0, name='root', size=0)
    idx = 1
    for i, row in drugs_p_lof.iterrows():
        if multiindex:
            drugname = row[drug_name_col].values[0]
        else:
            drugname = row[drug_name_col]
        genes = row[target_name_col]
        drug_p = row[drug_size_col]

        drug_idx = idx
        nx_tree.add_node(drug_idx, name=drugname, size=drug_p, parent_name='root', parent_size=0)
        nx_tree.add_edge(0, drug_idx)
        idx += 1
        for g in genes.split(", "):
            try:
                gene_size = gene_p_lof.loc[(gene_p_lof[gene_name_col] == g), gene_size_col].values[0]
            except:
                print g, gene_p_lof.loc[(gene_p_lof[gene_name_col] == g), gene_size_col].values
            nx_tree.add_node(idx, name=g, size=gene_size, parent_name=drugname, parent_size=drug_p)
            nx_tree.add_edge(drug_idx, idx)
            idx += 1
    return nx_tree


def plot_interactive_treemap(nx_tree, div_id="test_chart"):
    """
    Create HTML and JS objects for interactive treemap.

    In notebook do:
    G = tree_from_df(drugs_AF_lof.head(20))
    html, js = plot_interactive_treemap(G)

    then display:
    display(html)
    display(js)
    """
    html_template = jinja2.Template("""
<style>

body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  margin: auto;
  position: relative;
}

form {
  position: absolute;
  right: 10px;
  top: 10px;
}

.node {
  border: solid 1px white;
  font: 12px sans-serif;
  line-height: 12px;
  overflow: hidden;
  position: absolute;
  text-indent: 2px;
}

#ttooltip {
  position: absolute;
  width: 300px;
  height: auto;
  padding: 10px;
  background-color: white;
  -webkit-border-radius: 10px;
  -moz-border-radius: 10px;
  border-radius: 10px;
  -webkit-box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
  -moz-box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
  box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
  pointer-events: none;
}

#ttooltip.hidden {
  display: none;
}

#ttooltip p {
  margin: 0;
  font-family: sans-serif;
  font-size: 16px;
  line-height: 20px;
}

</style>


<div id="{{ chart_div }}_container">
    <div>
    <form>
      <label><input type="radio" name="mode" value="size" checked> Size</label>
      <label><input type="radio" name="mode" value="count"> Count</label>
    </form>
    </div>
    <div id="tooltip">
      <p><strong id="heading"></strong></p>
      <p><span id="probs"></span></p>
    </div>
    <div id="{{ chart_div }}"></div>
</div>

""")

    treemap_template = jinja2.Template("""
// Based on https://bl.ocks.org/mbostock/4063582

require(["d3"], function(d3) {
    var root = {{ data }}
    d3.select("#{{ chart_div }} svg").remove()

    var margin = {top: 40, right: 10, bottom: 10, left: 10},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

    var color = d3.scale.category20c();

    var treemap = d3.layout.treemap()
        .size([width, height])
        .sticky(true)
        .value(function(d) { return d.size; });

    var div = d3.select("#{{ chart_div }}").append("div")
        .style("position", "relative")
        .style("width", (width + margin.left + margin.right) + "px")
        .style("height", (height + margin.top + margin.bottom) + "px")
        .style("left", margin.left + "px")
        .style("top", margin.top + "px");

    var mousemove = function(d) {
        var xPosition = d.x + 10;
        var yPosition = d.y + 10;
        var xPosition = -10;
        var yPosition = -10;
        d3.select("#ttooltip")
            .style("left", xPosition + "px")
            .style("top", yPosition + "px");
        d3.select("#{{ chart_div }}_container #tooltip #heading")
            .text(d.parent_name + " (" + d.parent_size.toFixed(4) + ")");
        d3.select("#{{ chart_div }}_container #tooltip #probs")
            .text(d.name + " (" + d.size.toFixed(4) + ")");

        d3.select("#ttooltip").classed("hidden", false);
    };

    var mouseout = function() {
        d3.select("#ttooltip").classed("hidden", true);
    };

    var node = div.datum(root).selectAll(".node")
          .data(treemap.nodes)
        .enter().append("div")
          .attr("class", "node")
          .call(position)
          .style("background", function(d) { return d.children ? color(d.name) : null; })
          .text(function(d) { return d.children ? null : d.name; })
          .on("mousemove", mousemove)
          .on("mouseout", mouseout);

    d3.selectAll("#{{ chart_div }}_container input").on("change", function change() {
        var value = this.value === "count"
            ? function() { return 1; }
            : function(d) { return d.size; };

        node
            .data(treemap.value(value).nodes)
          .transition()
            .duration(1500)
            .call(position);
    });


    function position() {
      this.style("left", function(d) { return d.x + "px"; })
          .style("top", function(d) { return d.y + "px"; })
          .style("width", function(d) { return Math.max(0, d.dx - 1) + "px"; })
          .style("height", function(d) { return Math.max(0, d.dy - 1) + "px"; });
    }

});
""")

    full_js = Javascript(treemap_template.render(data=json.dumps(json_graph.tree_data(nx_tree, 0)),
                                                 chart_div=div_id))
    full_html = HTML(html_template.render(chart_div=div_id))
    return full_html, full_js


def plot_interactive_treemap_v2(nx_tree, div_id="test_chart"):
    """
    Create HTML and JS objects for interactive treemap.

    In notebook do:
    G = tree_from_df(drugs_AF_lof.head(20))
    html, js = plot_interactive_treemap(G)

    then display:
    display(html)
    display(js)
    """
    html_template = jinja2.Template("""
<style>

body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  margin: auto;
  position: relative;
}

.footer {
    z-index: 1;
    display: block;
    font-size: 26px;
    font-weight: 200;
    text-shadow: 0 1px 0 #fff;
}

svg {
    overflow: hidden;
}

rect {
    pointer-events: all;
    cursor: pointer;
    stroke: #EEEEEE;
}

.chart {
    display: block;
    margin: auto;
}

.parent .label {
    color: 'white';
}

.labelbody {
    background: transparent;
}

.label {
    margin: 2px;
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
    font: 12px sans-serif;
}

.child .label {
    color: #000000;
    white-space: pre-wrap;
    text-align: center;
    text-overflow: ellipsis;
}

</style>

<div id="{{ chart_div }}_container">
    <div id="{{ chart_div }}"></div>
    <div>
        <button id="save_{{ chart_div }}" type="submit" value="save_{{ chart_div }}">make static</button>
    </div>
    <div id="{{ chart_div }}_static">
    </div>
</div>

""")

    treemap_template = jinja2.Template("""
// Based on http://www.billdwhite.com/wordpress/2012/12/16/d3-treemap-with-title-headers/

require(["d3"], function(d3) {
    d3.select("#{{ chart_div }} svg").remove()
    var isIE = false;
    var margin = {top: 40, right: 10, bottom: 10, left: 10},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;
    var chartWidth = width;
    var chartHeight = height;
    var xscale = d3.scale.linear().range([0, chartWidth]);
    var yscale = d3.scale.linear().range([0, chartHeight]);
    var color = d3.scale.category10();
    var headerHeight = 20;
    var headerColor = "#555555";
    var transitionDuration = 500;
    var node;
    var root;

    var treemap = d3.layout.treemap()
        .round(false)
        .size([chartWidth, chartHeight])
        .sticky(true)
        .value(function(d) {
            return d.size;
        });

    var chart = d3.select("#{{ chart_div }}")
        .style("font", "10px sans-serif")
        .append("svg:svg")
        .attr("width", chartWidth)
        .attr("height", chartHeight)
        .append("svg:g");

    node = root = {{ data }};
    var nodes = treemap.nodes(root);

    var children = nodes.filter(function(d) {
        return !d.children;
    });
    var parents = nodes.filter(function(d) {
        return d.children;
    });

    // create parent cells
    var parentCells = chart.selectAll("g.cell.parent")
        .data(parents, function(d) {
            return "p-" + d.id;
        });
    var parentEnterTransition = parentCells.enter()
        .append("g")
        .attr("class", "cell parent")
        .on("click", function(d) {
            zoom(d);
        })
        .append("svg")
        .attr("class", "clip")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", headerHeight);
    parentEnterTransition.append("rect")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", headerHeight)
        .style("fill", headerColor);
    parentEnterTransition.append('text')
        .attr("class", "label")
        .attr("transform", "translate(3, 13)")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", headerHeight)
        .style("fill", "#FFFFFF")
        .text(function(d) {
            return d.name;
        });

    // update transition
    var parentUpdateTransition = parentCells.transition().duration(transitionDuration);
    parentUpdateTransition.select(".cell")
        .attr("transform", function(d) {
            return "translate(" + d.dx + "," + d.y + ")";
        });
    parentUpdateTransition.select("rect")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", headerHeight)
        .style("fill", headerColor);
    parentUpdateTransition.select(".label")
        .attr("transform", "translate(3, 13)")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", headerHeight)
        .text(function(d) {
            return d.name;
        });

    // remove transition
    parentCells.exit()
        .remove();

    // create children cells
    var childrenCells = chart.selectAll("g.cell.child")
        .data(children, function(d) {
            return "c-" + d.id;
        });
    // enter transition
    var childEnterTransition = childrenCells.enter()
        .append("g")
        .attr("class", "cell child")
        .on("click", function(d) {
            zoom(node === d.parent ? root : d.parent);
        })
        .append("svg")
        .attr("class", "clip");
    childEnterTransition.append("rect")
        .classed("background", true)
        .style("fill", function(d) {
            return color(d.parent.name);
        });
    childEnterTransition.append('text')
        .attr("class", "label")
        .attr('x', function(d) {
            return d.dx / 2;
        })
        .attr('y', function(d) {
            return d.dy / 2;
        })
        .attr("dy", ".35em")
        .attr("text-anchor", "middle")
        .style("fill", "#FFFFFF")
        .text(function(d) {
            return d.name;
        });
    // update transition
    var childUpdateTransition = childrenCells.transition().duration(transitionDuration);
    childUpdateTransition.select(".cell")
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
    childUpdateTransition.select("rect")
        .attr("width", function(d) {
            return Math.max(0.01, d.dx);
        })
        .attr("height", function(d) {
            return d.dy;
        })
        .style("fill", function(d) {
            return color(d.parent.name);
        });
    childUpdateTransition.select(".label")
        .attr('x', function(d) {
            return d.dx / 2;
        })
        .attr('y', function(d) {
            return d.dy / 2;
        })
        .attr("dy", ".35em")
        .attr("text-anchor", "middle")
        .style("display", "none")
        .text(function(d) {
            return d.name;
        });

    // exit transition
    childrenCells.exit()
        .remove();

    d3.select("select").on("change", function() {
        console.log("select zoom(node)");
        treemap.value(this.value == "size" ? size : count)
            .nodes(root);
        zoom(node);
    });

    zoom(node);

    function size(d) {
        return d.size;
    }
    function count(d) {
        return 1;
    }

    //and another one
    function textHeight(d) {
        var ky = chartHeight / d.dy;
        yscale.domain([d.y, d.y + d.dy]);
        return (ky * d.dy) / headerHeight;
    }

    function getRGBComponents(color) {
        var r = color.substring(1, 3);
        var g = color.substring(3, 5);
        var b = color.substring(5, 7);
        return {
            R: parseInt(r, 16),
            G: parseInt(g, 16),
            B: parseInt(b, 16)
        };
    }
    function idealTextColor(bgColor) {
        var nThreshold = 105;
        var components = getRGBComponents(bgColor);
        var bgDelta = (components.R * 0.299) + (components.G * 0.587) + (components.B * 0.114);
        console.log(bgDelta);
        return ((255 - bgDelta) < nThreshold) ? "#000000" : "#ffffff";
    }

    function zoom(d) {
        treemap
            .padding([headerHeight / (chartHeight / d.dy), 0, 0, 0])
            .nodes(d);

        // moving the next two lines above treemap layout messes up padding of zoom result
        var kx = chartWidth / d.dx;
        var ky = chartHeight / d.dy;
        var level = d;

        xscale.domain([d.x, d.x + d.dx]);
        yscale.domain([d.y, d.y + d.dy]);

        var zoomTransition = chart.selectAll("g.cell").transition().duration(transitionDuration)
            .attr("transform", function(d) {
                return "translate(" + xscale(d.x) + "," + yscale(d.y) + ")";
            })
            .each("start", function() {
                d3.select(this).select("label")
                    .style("display", "none");
            })
            .each("end", function(d, i) {
                if (!i && (level !== self.root)) {
                    chart.selectAll(".cell.child")
                        .filter(function(d) {
                            return d.parent === self.node; // only get the children for selected group
                        })
                        .select(".label")
                        .style("display", "")
                        .style("fill", function(d) {
                            return idealTextColor(color(d.parent.name));
                        });
                }
            });

        zoomTransition.select(".clip")
            .attr("width", function(d) {
                return Math.max(0.01, (kx * d.dx));
            })
            .attr("height", function(d) {
                return d.children ? headerHeight : Math.max(0.01, (ky * d.dy));
            });

        zoomTransition.select(".label")
            .attr("width", function(d) {
                return Math.max(0.01, (kx * d.dx));
            })
            .attr("height", function(d) {
                return d.children ? headerHeight : Math.max(0.01, (ky * d.dy));
            })
            .text(function(d) {
                return d.name;
            });

        zoomTransition.select(".child .label")
            .attr("x", function(d) {
                return kx * d.dx / 2;
            })
            .attr("y", function(d) {
                return ky * d.dy / 2;
            });

        zoomTransition.select("rect")
            .attr("width", function(d) {
                return Math.max(0.01, (kx * d.dx));
            })
            .attr("height", function(d) {
                return d.children ? headerHeight : Math.max(0.01, (ky * d.dy));
            })
            .style("fill", function(d) {
                return d.children ? headerColor : color(d.parent.name);
            });

        node = d;

        if (d3.event) {
            d3.event.stopPropagation();
        }
    }


    d3.select("#save_{{ chart_div }}").on("click", function(){
      var html = d3.select("#{{ chart_div }} svg")
            .attr("version", 1.1)
            .attr("xmlns", "http://www.w3.org/2000/svg")
            .style("font-family", "sans-serif")
            .style("font", "12px sans-serif")
            .node().parentNode.innerHTML;

      console.log(html);
      var imgsrc = 'data:image/svg+xml;base64,'+ btoa(html);
      var img = '<img src="'+imgsrc+'">';
      d3.select("#{{ chart_div }}").html(img);

    });

});
""")

    full_js = Javascript(treemap_template.render(data=json.dumps(json_graph.tree_data(nx_tree, 0)),
                                                 chart_div=div_id))
    full_html = HTML(html_template.render(chart_div=div_id))
    return full_html, full_js


def three_circle_venn(set1, set2, set3, labels=('set1', 'set2', 'set3'), title=None,
                      colors=None, alpha=.8, hatch_set=None,
                      figsize=(10, 10), fontsize=14, fontcolor='white',
                      return_sets=False):
    if colors is None:
        colors = {'set1': '#B5003E', 'set2': '#F0B710', 'set3': '#2A3FA6',
                  'set12': '#D35C27', 'set13': '#ca59db', 'set23': '#702072',
                  'set123': '#6C9C0F'}
    joint_set = set1 & set2 & set3
    overlap_all = len(joint_set)
    only_set1 = len(set1 - set(list(set2) + list(set3)))
    only_set2 = len(set2 - set(list(set1) + list(set3)))
    overlap_set1_set2 = len(set1 & set2) - overlap_all
    only_set3 = len(set3 - set(list(set1) + list(set2)))
    overlap_set1_set3 = len(set1 & set3) - overlap_all
    overlap_set2_set3 = len(set2 & set3) - overlap_all

    plt.figure(figsize=figsize)

    v = venn3(subsets=(only_set1, only_set2, overlap_set1_set2, only_set3, overlap_set1_set3,
                       overlap_set2_set3, overlap_all),
              set_labels=labels)

    c = venn3_circles(subsets=(only_set1, only_set2, overlap_set1_set2, only_set3, overlap_set1_set3,
                               overlap_set2_set3, overlap_all),
                      linestyle='dashed')
    for circle in c:
        circle.set_lw(0.)
#     c[0].set_lw(1.0)
#     c[0].set_ls('dotted')

    if v.get_patch_by_id('100'):
        v.get_patch_by_id('100').set_color(colors['set1'])  # only set 1
        v.get_patch_by_id('100').set_alpha(alpha)
    if v.get_patch_by_id('010'):
        v.get_patch_by_id('010').set_color(colors['set2'])  # only set 2
        v.get_patch_by_id('010').set_alpha(alpha)
    if v.get_patch_by_id('001'):
        v.get_patch_by_id('001').set_color(colors['set3'])  # only set 3
        v.get_patch_by_id('001').set_alpha(alpha)
    if v.get_patch_by_id('110'):
        v.get_patch_by_id('110').set_color(colors['set12'])  # set 1 & set 2
        v.get_patch_by_id('110').set_alpha(alpha)
    if v.get_patch_by_id('101'):
        v.get_patch_by_id('101').set_color(colors['set13'])  # set 1 & set 3
        v.get_patch_by_id('101').set_alpha(alpha)
    if v.get_patch_by_id('011'):
        v.get_patch_by_id('011').set_color(colors['set23'])  # set 2 & set 3
        v.get_patch_by_id('011').set_alpha(alpha)
    if v.get_patch_by_id('111'):
        v.get_patch_by_id('111').set_color(colors['set123'])  # set 1 & 2 & 3
        v.get_patch_by_id('111').set_alpha(alpha)

    for text in v.set_labels:
        if text:
            text.set_fontsize(fontsize)

    for text in v.subset_labels:
        if text:
            text.set_fontsize(fontsize)
            text.set_color(fontcolor)

    if hatch_set:
        if v.get_patch_by_id(hatch_set):
            v.get_patch_by_id(hatch_set).set_hatch("///")
            v.get_patch_by_id(hatch_set).set_edgecolor("white")
            v.subset_labels[0].set_color("grey")

    if title:
        plt.title(title)

    if return_sets:
        return dict(only1=set1 - set(list(set2) + list(set3)),
                    only2=set2 - set(list(set1) + list(set3)),
                    only3=set3 - set(list(set1) + list(set2)),
                    overlap12=(set1 & set2) - joint_set,
                    overlap13=(set1 & set3) - joint_set,
                    overlap23=(set2 & set3) - joint_set,
                    overlap=joint_set)
