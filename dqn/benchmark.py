import pickle
from os import listdir
from os.path import isfile, join
import plotly.graph_objects as go

from .utils import calc_rollling_mean

COLORS = [
     '#3195DA',
     '#FF0063',
     '#0C1425',
     '#57D493'
 ]


class Benchmarks():

    def __init__(self):
        pass

    def save_score(self, name, scores, description):
        with open("benchmarks/{}_score.txt".format(name), "wb") as fp:
            pickle.dump(scores, fp)
        with open("benchmarks/{}_description.txt".format(name), "wb") as fp:
            pickle.dump(description, fp)

    def description(self, name):
        with open("benchmarks/{}_description.txt".format(name), "rb") as fp:
            description = pickle.load(fp)
        return description

    def load_benchmarks(self):
        files = [f.split('_score')[0] for f in listdir('benchmarks/') \
                 if isfile(join('benchmarks/', f)) and 'score' in f]
        return {f: self.load(f) for f in files} 

    def load(self, name):
        with open("benchmarks/{}_score.txt".format(name), "rb") as fp:
            scores = pickle.load(fp)
        return scores

    def plot_bench(self, scores, title='', mean=None, opacity=0.5):
        if mean is None:
            fig = go.Figure(data=go.Scatter(y=scores, line_color=COLORS[0]))
        else:
            fig = go.Figure(data=go.Scatter(y=scores, line_color=COLORS[0],
                                            opacity=opacity, showlegend=False))
            means = calc_rollling_mean(scores, mean)
            fig.add_trace(go.Scatter(y=means, line_color=COLORS[0], showlegend=False))
        fig.update_layout(title=title,
                          xaxis_title='Episode', yaxis_title='Episode Reward')
        fig.show() 
        return fig

    def plot_benchs(self, bench_dict, title='', mean=None, opacity=0.5):
        fig = go.Figure()
        for i, (name, scores) in enumerate(bench_dict.items()):
            if mean is None:
                fig.add_trace(go.Scatter(y=scores, name=name, showlegend=True, line_color=COLORS[i]))
            else:
                fig.add_trace(go.Scatter(y=scores, name=name, opacity=opacity,
                                         showlegend=True, line_color=COLORS[i]))
                means = calc_rollling_mean(scores, mean)
                fig.add_trace(go.Scatter(y=means, showlegend=False, line_color=COLORS[i]))
        fig.update_layout(title=title,
                    xaxis_title='# Episode', yaxis_title='Episode Reward')
        fig.show()