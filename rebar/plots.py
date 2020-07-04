import time
import numpy as np
import pandas as pd
from bokeh import plotting as bop, io as boi
from bokeh import models as bom, events as boe, layouts as bol
from bokeh.palettes import Category10_10
from itertools import cycle
from . import stats, paths
from contextlib import contextmanager
from IPython.display import clear_output

bop.output_notebook(hide_banner=True)

def array(fig):
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)
    return (np.frombuffer(renderer.buffer_rgba(), np.uint8)
                        .reshape((h, w, 4))
                        [:, :, :3]
                        .copy())

def timedelta_xaxis(f):
    f.xaxis.ticker = bom.tickers.DatetimeTicker()
    f.xaxis.formatter = bom.FuncTickFormatter(code="""
        // TODO: Add support for millis

        // Calculate the hours, mins and seconds
        var s = Math.floor(tick / 1e3);
        
        var m = Math.floor(s/60);
        var s = s - 60*m;
        
        var h = Math.floor(m/60);
        var m = m - 60*h;
        
        var h = h.toString();
        var m = m.toString();
        var s = s.toString();
        var pm = m.padStart(2, "0");
        var ps = s.padStart(2, "0");

        // Figure out what the min resolution is going to be
        var min_diff = Infinity;
        for (var i = 0; i < ticks.length-1; i++) {
            min_diff = Math.min(min_diff, ticks[i+1]-ticks[i]);
        }

        if (min_diff <= 60e3) {
            var min_res = 2;
        } else if (min_diff <= 3600e3) {
            var min_res = 1;
        } else {
            var min_res = 0;
        }

        // Figure out what the max resolution is going to be
        if (ticks.length > 1) {
            var max_diff = ticks[ticks.length-1] - ticks[0];
        } else {
            var max_diff = Infinity;
        }

        if (max_diff >= 3600e3) {
            var max_res = 0;
        } else if (max_diff >= 60e3) {
            var max_res = 1;
        } else {
            var max_res = 2;
        }

        // Format the timedelta. Finally.
        if ((max_res == 0) && (min_res == 0)) {
            return `${h}h`;
        } else if ((max_res == 0) && (min_res == 1)) {
            return `${h}h${pm}`;
        } else if ((max_res == 0) && (min_res == 2)) {
            return `${h}h${pm}m${ps}`;
        } else if ((max_res == 1) && (min_res == 1)) {
            return `${m}m`;
        } else if ((max_res == 1) && (min_res == 2)) {
            return `${m}m${ps}`;
        } else if ((max_res == 2) && (min_res == 2)) {
            return `${s}s`;
        }
    """)

def suffix_yaxis(f):
    f.yaxis.formatter = bom.FuncTickFormatter(code="""
        var min_diff = Infinity;
        for (var i = 0; i < ticks.length-1; i++) {
            min_diff = Math.min(min_diff, ticks[i+1]-ticks[i]);
        }

        var suffixes = [
            'y', 'z', 'a', 'f', 'p', 'n', 'µ', 'm',
            '', 
            'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'];
        var precision = Math.floor(Math.log10(min_diff));
        var scale = Math.floor(precision/3);
        var index = scale + 8;
        if (index < 0) {
            //TODO: Fall back to numbro here
            return tick;
        } else if (index == 7) {
            // Millis are weird. Feels better to rende them as decimals.
            var decimals = -precision;
            return `${tick.toFixed(decimals)}`
        } else if (index < suffixes.length) {
            var suffix = suffixes[index];
            var scaled = tick/Math.pow(10, 3*scale);
            return `${scaled.toFixed(0)}${suffix}`
        } else {
            //TODO: Fall back to numbro here
            return tick;
        }
    """)

def x_zeroline(f):
    f.add_layout(bom.Span(location=0, dimension='height'))

def default_tools(f):
    f.toolbar_location = None
    f.toolbar.active_drag = f.select_one(bom.BoxZoomTool)
    # f.toolbar.active_scroll = f.select_one(bom.WheelZoomTool)
    # f.toolbar.active_inspect = f.select_one(bom.HoverTool)
    f.js_on_event(
        boe.DoubleTap, 
        bom.callbacks.CustomJS(args=dict(p=f), code='p.reset.emit()'))

def styling(f):
    timedelta_xaxis(f)
    suffix_yaxis(f)

def _timeseries(source, x, y):
    #TODO: Work out how to apply the axes formatters to the tooltips
    f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])
    f.line(x=x, y=y, source=source)
    default_tools(f)
    x_zeroline(f)
    styling(f)

    return f

def timeseries(s):
    source = bom.ColumnDataSource(s.reset_index())
    return _timeseries(source, s.index.name, s.name)

def _timedataframe(source, x, ys):
    f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])

    for y, color in zip(ys, cycle(Category10_10)):
        f.line(x=x, y=y, legend_label=y, color=color, width=2, source=source)

    default_tools(f)
    x_zeroline(f)
    styling(f)

    f.legend.label_text_font_size = '8pt'
    f.legend.margin = 7
    f.legend.padding = 0
    f.legend.spacing = 0
    f.legend.background_fill_alpha = 0.3
    f.legend.border_line_alpha = 0.
    f.legend.location = 'top_left'

    return f

def timedataframe(df):
    source = bom.ColumnDataSource(df.reset_index())
    return _timedataframe(source, df.index.name, df.columns)

def timegroups(df):
    tags = df.columns.str.extract(r'^(?P<chart1>.*)/(?P<label>.*)|(?P<chart2>.*)$')
    tags['chart'] = tags.chart1.combine_first(tags.chart2)
    tags.index = df.columns
    return tags[['chart', 'label']].fillna('')


class Stream:

    def __init__(self, run_name=-1, prefix=''):
        super().__init__()

        self._reader = stats.Reader(run_name, prefix)

        self._source = bom.ColumnDataSource({'time': np.array([0])})
        self._handle = None

    def _new_grid(self, children):
        return bol.gridplot(children, ncols=4, plot_width=350, plot_height=300, merge_tools=False)

    def _init(self, df):
        self._source = bom.ColumnDataSource(df.reset_index())

        children = []
        for name, group in timegroups(df).groupby('chart'):
            if group.label.eq('').all():
                assert len(group) == 1
                f = _timeseries(self._source, 'time', group.index[0])
                f.title = bom.Title(text=name)
            else:
                f = _timedataframe(self._source, 'time', group.index)
                f.title = bom.Title(text=name)
            children.append(f)
        self._grid = self._new_grid(children)
        ## TODO: Not wild about this
        clear_output(wait=True)
        self._handle = bop.show(self._grid, notebook_handle=True)

    def update(self, rule='60s', df=None):
        # Drop the last row as it'll be constantly refreshed as the period occurs
        df = self._reader.resample(rule).iloc[:-1] if df is None else df

        has_new_cols = not df.columns.isin(self._source.data).all()
        if has_new_cols:
            self._init(df)
        else:
            threshold = len(self._source.data['time'])
            new = df.iloc[threshold:]
            self._source.stream(new.reset_index())
        
        boi.push_notebook(handle=self._handle)

def view(run_name=-1, prefix='', rule='60s'):
    stream = Stream(run_name, prefix)
    while True:
        stream.update(rule=rule)
        time.sleep(1)

def review(run_name=-1, prefix='', rule='60s'):
    stream = Stream(run_name, prefix)
    stream.update(rule=rule)

def test_stream():
    times = pd.TimedeltaIndex([0, 60e3, 120e3])
    dfs = [
        pd.DataFrame([[0]], columns=['a'], index=times[:1]),
        pd.DataFrame([[0, 1], [10, 20]], columns=['a', 'b/a'], index=times[:2]),
        pd.DataFrame([[0, 1, 2], [10, 20, 30], [100, 200, 300]], columns=['a', 'b/a', 'b/b'], index=times[:3])]

    stream = Stream()
    for df in dfs:
        stream.update(df)
        time.sleep(1)
    