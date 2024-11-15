import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_increasing_objective_history(study):
    """
    Plots the multi-objective optimization history for an Optuna study, showing all points and
    connecting only those points that form an increasing sequence for each objective.

    Parameters:
        study (optuna.study.Study): The Optuna study containing trials and objective values.

    Returns:
        fig (plotly.graph_objects.Figure): The generated plotly figure.
    """
    objective_labels = ['AUC', 'Accuracy', 'F1 Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    trials = study.trials
    objective_values = list(zip(*[trial.values for trial in trials]))  # Transpose to get separate lists for each objective

    fig = go.Figure()
    def get_increasing_segments(values):
        x_vals = []
        y_vals = []
        current_best = float('-inf')

        for i in range(len(values)):
            if values[i] > current_best:
                x_vals.append(i)
                y_vals.append(values[i])
                current_best = values[i]
        return x_vals, y_vals

    for idx, values in enumerate(objective_values):
        color = colors[idx % len(colors)]
        label = objective_labels[idx]

        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.5),
            showlegend=False,
            name=f'{label}',
        ))

        x_increasing, y_increasing = get_increasing_segments(values)

        fig.add_trace(go.Scatter(
            x=x_increasing,
            y=y_increasing,
            mode='lines+markers',
            name=f'{label}',
            line=dict(color=color, shape='linear'),
            marker=dict(color=color),
        ))

    fig.update_layout(
        template='plotly_dark',
        title='<b>Multi-Objective Optimization History Plot</b>',
        title_x=0.5,
        xaxis_title='Trial',
        yaxis_title='Objective Value',
        legend_title='Objectives',
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, len(trials)])
    return fig

def plot_metrics(trials, dataset_ids, model):
    """
    Generate three plots, one for each metric (AUC, Accuracy, F1 Score), stacked vertically.

    Parameters:
    trials (list): List of Optuna trials.
    dataset_ids (list): List of dataset IDs.
    """
    metrics = ['auc', 'accuracy', 'f1']
    fig = make_subplots(rows=3, cols=1, subplot_titles=[f'{metric.upper()}' for metric in metrics])

    added_to_legend = set()

    for row, metric in enumerate(metrics, start=1):
        metric_data = []

        for trial in trials:
            metric_data.append(trial.user_attrs.get(f'{metric}_scores', []))

        mean_metric_scores = [sum(trial_metric) / len(trial_metric) for trial_metric in metric_data]

        fig.add_trace(go.Scatter(
            x=list(range(len(mean_metric_scores))),
            y=mean_metric_scores,
            mode='lines',
            name=f'Mean Across Datasets',
            line=dict(dash='dash', color='black'),
            showlegend=(row == 1)  # Only show legend for mean metrics in the first plot
        ), row=row, col=1)

        for dataset_idx in range(len(metric_data[0])):
            id = dataset_ids[dataset_idx]
            metric_scores = [trial_metric[dataset_idx] for trial_metric in metric_data]

            showlegend = id not in added_to_legend
            if showlegend:
                added_to_legend.add(id)

            fig.add_trace(go.Scatter(
                x=list(range(len(metric_scores))),
                y=metric_scores,
                mode='lines+markers',
                name=f'Dataset {id}',
                showlegend=showlegend
            ), row=row, col=1)

    fig.update_layout(
        height=700,
        title=f'Metrics for Each Optuna Trial on {model}',
        legend_title='Datasets',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,  # Adjust the position of the legend
            xanchor='center',
            x=0.5
        )
    )
    fig.update_xaxes(range=[0, len(trials)-0.9], title_text = 'Trail', row=3, col=1)
    fig.update_yaxes(title_text='Value', range=[0, 1])
    fig.show()