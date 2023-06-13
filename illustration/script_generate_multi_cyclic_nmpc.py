import plotly.express as px
from plotly import graph_objects as go

import plotly.io as pio
pio.kaleido.scope.mathjax = None
# otherwise a strange mathjax loading appears at the bottom of the figure

fig = go.Figure()

final_sol_str = "Optimal\nSolution"
mlw = 3

fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP n</b>", "<b>...</b>", "<b>OCP 2</b>", "<b>OCP 1</b>", "<b>OCP 0</b>"],
        x=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        orientation="h",
        marker_color="rgba(0,0,0,0)",
        showlegend=False,
        hoverinfo="skip",
    )
)

fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP 0</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape="+",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 0</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape="+",
        marker_color=px.colors.qualitative.Plotly[1],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 0</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape="+",
        marker_color=px.colors.qualitative.Plotly[2],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)

fig.add_trace(
    go.Bar(
        y=["<b>OCP 1</b>"],
        x=[1],
        orientation="h",
        marker_color="rgba(255, 0, 0, 0)",
        showlegend=False,
    )
)
fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP 1</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape=".",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 1</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape=".",
        marker_color=px.colors.qualitative.Plotly[1],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 1</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape=".",
        marker_color=px.colors.qualitative.Plotly[2],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)

fig.add_trace(
    go.Bar(
        y=["<b>OCP 2</b>"],
        x=[2],
        orientation="h",
        marker_color="rgba(255, 0, 0, 0)",
        showlegend=False,
    ),
)
fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP 2</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape="x",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 2</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape="x",
        marker_color=px.colors.qualitative.Plotly[1],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=["<b>OCP 2</b>"],
        x=[1],
        orientation="h",
        marker_pattern_shape="x",
        marker_color=px.colors.qualitative.Plotly[2],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)

fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>"],
        x=[1],
        orientation="h",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>"],
        x=[1],
        orientation="h",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)

fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>"],
        x=[1],
        orientation="h",
        marker_color="rgba(255, 0, 0, 0)",
        showlegend=False,
    )
)

fig.add_trace(
    go.Bar(
        y=["<b>OCP n</b>"],
        x=[6],
        orientation="h",
        marker_color="rgba(255, 0, 0, 0)",
        showlegend=False,
    ),
)

fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP n</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape="/",
        marker_color=px.colors.qualitative.Plotly[0],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP n</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape="/",
        marker_color=px.colors.qualitative.Plotly[1],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)
fig.add_trace(
    go.Bar(
        y=[f"<b>{final_sol_str}</b>", "<b>OCP n</b>"],
        x=[1, 1],
        orientation="h",
        marker_pattern_shape="/",
        marker_color=px.colors.qualitative.Plotly[2],
        showlegend=False,
        marker_line_color="rgb(1,1,1)",
        marker_line_width=mlw,
    )
)

fig.update_layout(
    barmode="stack",
    template="plotly_white",
)
# update xlabel and y label
fig.update_xaxes(title_text="<b>Cycles</b>")
# show x axis arrow
fig.update_xaxes(showline=False, linewidth=mlw, linecolor="black", mirror=False)
# show y axis arrow
fig.update_yaxes(showline=False, linewidth=mlw, linecolor="black", mirror=False)
# black font color
fig.update_layout(
    font=dict(
        color="black",
        family="Times New Roman",
        size=20,
        # bold font
        # bold=True,
    ),
    showlegend=False,
)


fig.update_xaxes(
    showgrid=False,
    gridwidth=1,
    gridcolor="black",
    zeroline=False,
    tick0=0.5,
    dtick=1,
    tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    ticktext=["1", "2", "3", "4", "5", "...", "n-2", "n-1", "n"],
    range=[0.25, 9.75],
    minor=dict(
        showgrid=True,
        griddash="dash",
        gridwidth=1.5,
        gridcolor="black",
        tick0=0.5,
        dtick=1,
        # tickvals=[1, 2, 3, 4, 5],
    ),
)

# figure size
fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
)

# fig.show()
# export figure in eps and remove mathjax
fig.write_image("multi_cycli_nmpc.eps", format="eps", engine="kaleido")
# in png to
fig.write_image("multi_cycli_nmpc.png", format="png", engine="kaleido")
