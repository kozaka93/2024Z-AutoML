import matplotlib
import plotly
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly as py
from scipy.stats import gaussian_kde
import plotly.express as px
class StockAnalysis:
    def __init__(self, data):
        """
        Initialize the StockAnalysis class with the given dataset and library info.

        :param data: DataFrame containing the stock data.
        """
        self.data = data
        self.libraries = self.get_library_info()
        self.data_prepared = self.prepare_data()
        self.segment_stats = None
        self.cluster_stats = None

    @staticmethod
    def get_library_info():
        """
        Get information about the libraries used for the analysis.

        :return: Dictionary with library names and their versions.
        """
        import matplotlib
        import plotly
        import scipy
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        import plotly as py
        from scipy.stats import gaussian_kde
        import plotly.express as px
        
        return {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "plotly": plotly.__version__,
            "scipy": scipy.__version__
        }

    def display_library_info(self):
        """Print the versions of libraries used in the analysis."""
        print("Library Information:")
        for lib, version in self.libraries.items():
            print(f"{lib}: {version}")

    def display_recomanded_library_info(self):
        recomanded = {'numpy': '1.24.4',
        'pandas': '2.0.3',
        'matplotlib': '3.7.5',
        'plotly': '5.24.1',
        'scipy': '1.10.1'}
        print("Recomanded Library Information:")
        for lib, version in recomanded.items():
            print(f"{lib}: {version}")

    # Attribute Preparation Methods
    def prepare_data(self):
        """Prepare the data for analysis by cleaning and transforming it as needed."""
        data = self.data.copy()
        # data['datetime'] = pd.to_datetime(data['datetime'])
        data.reset_index(inplace=True)
        data['high_low_diff'] = data['high'] - data['low']
        data['segment_id'] = (data['clusters'] != data['clusters'].shift()).cumsum()
        self.data = data
        pass

    def calculate_cluster_stats(self):
        """Compute cluster-wise statistics like average prices and total volume."""
        pass

    def calculate_segment_stats(self):
        """Compute segment-wise statistics such as duration, volume, and price change."""
        pass

    # Plotting Methods
    def high_low_difference_boxplot(self):
        data = self.data.copy()

        # Create box plot
        fig = px.box(
            data,
            x="clusters",
            y="high_low_diff",
            color="clusters",
            title="High-Low Price Difference by Cluster",
            labels={"clusters": "Cluster", "high_low_diff": "High-Low Difference"}
        )

        fig.show()

    def segment_length_dist_plot(self,nbins=0):
        data = self.data.copy()

        segment_lengths = data.groupby("segment_id").size()

        if nbins == 0:
            fig = px.histogram(
            segment_lengths,
            title="Segment Length Distribution",
            labels={"value": "Segment Length", "count": "Frequency"}
            )
        else:
            fig = px.histogram(
            segment_lengths,
            nbins=nbins,
            title="Segment Length Distribution",
            labels={"value": "Segment Length", "count": "Frequency"}
            )
        fig.update_layout(
            xaxis_title="Segment Length",
            yaxis_title="Frequency",
            hovermode="x unified"
        )
        fig.update_traces(
            hovertemplate="<b>Segment Length: %{x}</b><br>Frequency: %{y}<extra></extra>"
        )
        fig.show()

    def price_change_distribution(self,percentage=True,nbins=0):
        data = self.data.copy()
        segment_price_change = data.groupby("segment_id").agg({
            "open": "first",
            "close": "last"
        })
        if (percentage):
            segment_price_change["price_change"] = (segment_price_change["close"] - segment_price_change["open"]) / segment_price_change["open"]
            title = "Price Change Distribution (%)"
            x_axis_title = "Price Change (%)"
        else:
            segment_price_change["price_change"] = (segment_price_change["close"] - segment_price_change["open"])
            title = "Price Change Distribution"
            x_axis_title = "Price Change"

        # Plot a histogram of price changes
        if nbins == 0:
            fig = px.histogram(
            segment_price_change,
            x="price_change",
            title=title,
            labels={"price_change": x_axis_title, "count": "Frequency"}
            )
        else:
            fig = px.histogram(
                segment_price_change,
                x="price_change",
                nbins=nbins,
                title=title,
                labels={"price_change": x_axis_title, "count": "Frequency"}
            )
        fig.update_layout(xaxis_title=x_axis_title, yaxis_title="Frequency")
        fig.show()

    def cluster_transition_heatmap(self):
        data = self.data.copy()
        
        # Compute transitions
        transitions = data[["segment_id", "clusters"]].drop_duplicates().reset_index(drop=True)
        transitions["next_cluster"] = transitions["clusters"].shift(-1)

        # Count transitions
        transition_counts = transitions.groupby(["clusters", "next_cluster"]).size().reset_index(name="count")

        # Create a pivot table for the heatmap
        pivot = transition_counts.pivot(index="clusters", columns="next_cluster", values="count")

        # Normalize by row sum
        normalized_pivot = pivot.div(pivot.sum(axis=1), axis=0)

        # Plot the heatmap with normalized values
        fig = px.imshow(
            normalized_pivot,
            title="Normalized Cluster Transition Heatmap",
            labels=dict(x="Next Cluster", y="Current Cluster", color="Proportion"),
            color_continuous_scale="Viridis",
            zmin=0, zmax=1  # Set the color scale range from 0 to 1

        )

        # Add text annotations for proportions
        fig.update_traces(
            text=normalized_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
            texttemplate="%{text}",
            textfont=dict(size=12)
        )

        fig.show()

    def plot_radar_cluster_stats(self,column='close'):
        data = self.data.copy()

        data["segment_row"] = data.groupby("segment_id").cumcount()

        # Plot line chart
        fig = px.line(
            data,
            x="segment_row",
            y=column,
            color="segment_id",
            title=f"Segment-Wise {column} Price Trends",
            labels={"segment_row": "Row within Segment", "close": f"{column} Price"}
        )

        fig.update_layout(legend_title="Segment ID")
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Show All Lines",
                            "method": "update",
                            "args": [{"visible": [True] * len(data["segment_id"].unique())}],
                        },
                        {
                            "label": "Hide All Lines",
                            "method": "update",
                            "args": [{"visible": ["legendonly"] * len(data["segment_id"].unique())}],
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.8,
                    "xanchor": "left",
                    "y": 1.3,
                    "yanchor": "top",
                }
            ],
            legend_title="Segment ID",
            title_font_size=18
        )

        fig.show()

    def segment_stats_paralel_plot(self):
        data = self.data.copy()

        segment_stats = data.groupby("segment_id").agg({
            "datetime": lambda x: (x.max() - x.min()).total_seconds() / 60,  # Duration in minutes
            "volume": "sum",
            "open": "first",
            "close": "last"
        })
        segment_stats["price_change"] = segment_stats["close"] - segment_stats["open"]
        fig = px.parallel_coordinates(
            segment_stats.reset_index(),
            dimensions=["datetime", "volume", "price_change"],
            color="price_change",
            title="Parallel Coordinates for Segment Features",
            labels={"datetime": "Duration (minutes)", "volume": "Total Volume", "price_change": "Price Change"},
            color_continuous_scale=px.colors.diverging.Spectral
        )
        fig.show()

    def plot_moving_window(self, window_size=0, plot_type='Candlestick', x_axis_type = 1):
        df = self.data.copy()
        df['datetime_text'] = df['datetime'].map(lambda x: f'Date: {x}')
        df['MA'] = df['close'].rolling(window=window_size).mean()
        df['VMA'] = df['volume'].rolling(window=window_size).mean()
        colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'black'}  # Define custom colors for clusters
        def_of_colors = {0: 'Stable', 1: 'Dropping', 2: 'Increasing', 3: 'Nothing'}  # Define custom names for clusters

        if x_axis_type == 1:
            x_axis = df['datetime']
            fig = make_subplots(
                rows=2, cols=1,  # Two rows, one column
                shared_xaxes=True,  # Share the x-axis
                vertical_spacing=0,  # No gap between charts
                subplot_titles=("Stock Price Episodes", "")  # Titles for each chart
            )
            fig.update(layout_xaxis_rangeslider_visible=False)

            for cluster in df['clusters'].unique():
                cluster_data = df[df['clusters'] == cluster]
                x_axis_iter = x_axis[df['clusters'] == cluster]
                
                # Calculate percentage change
                cluster_data['percentage_change'] = ((cluster_data['close'] - cluster_data['open']) / cluster_data['open']) * 100
                
                # Create hover text
                hover_text = (
                    "Datetime: " + cluster_data['datetime'].astype(str) + "<br>" +
                    "Open: " + cluster_data['open'].astype(str) + "<br>" +
                    "High: " + cluster_data['high'].astype(str) + "<br>" +
                    "Low: " + cluster_data['low'].astype(str) + "<br>" +
                    "Close: " + cluster_data['close'].astype(str) + "<br>" +
                    "Percentage Change: " + cluster_data['percentage_change'].round(2).astype(str) + "%"
                )
                
                if plot_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(
                        x=x_axis_iter,
                        open=cluster_data['open'],
                        high=cluster_data['high'],
                        low=cluster_data['low'],
                        close=cluster_data['close'],
                        increasing_line_color=colors[cluster],  # Use color for increasing values
                        decreasing_line_color=colors[cluster],  # Use color for decreasing values
                        name=f'{def_of_colors[cluster]}',
                        text=hover_text,  # Add custom hover text
                        hoverinfo='text'  # Use custom hover text
                    ),
                        row=1, col=1
                    )
                elif plot_type == 'Ohlc':
                    fig.add_trace(go.Ohlc(
                        x=x_axis_iter,
                        open=cluster_data['open'],
                        high=cluster_data['high'],
                        low=cluster_data['low'],
                        close=cluster_data['close'],
                        increasing_line_color=colors[cluster],  # Use color for increasing values
                        decreasing_line_color=colors[cluster],  # Use color for decreasing values
                        name=f'{def_of_colors[cluster]}',
                        text=hover_text,  # Add custom hover text
                        hoverinfo='text'  # Use custom hover text
                    ),
                        row=1, col=1
                    )
                else:
                    return "Wrong plot_type"

            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MA"], name=f"{window_size}-MA", line=dict(color="black")),
            row=1, col=1
            )

            daily_volume = df.resample('D', on='datetime').sum().reset_index()
            fig.add_trace(go.Bar(
                x=daily_volume['datetime'],
                y=daily_volume['volume'],
                name='Daily Volume',
                marker=dict(color='purple')
            ),
            row=2, col=1
            )

            fig.add_trace(
            go.Scatter(x=df["datetime"], y=df["VMA"], name=f"{window_size}-VMA", line=dict(color="black")),
            row=2, col=1
            )
            
            # Make the line thinner
            fig.update_traces(line=dict(width=1), selector=dict(type='scatter'))

            fig.update_layout(
                title="Stock Price and Trading Volume",
                xaxis=dict(title="Time", showgrid=False),  # Shared x-axis
                yaxis=dict(title="Price"),
                yaxis2=dict(title="Volume"),
                hovermode="x unified",  # Unified hover
                margin=dict(l=50, r=50, t=60, b=50)  # Adjust margins to reduce padding
            )
            fig.show()
        else:
            x_axis = df.index
            fig = go.Figure()
            fig.update(layout_xaxis_rangeslider_visible=False)
            for cluster in df['clusters'].unique():
                cluster_data = df[df['clusters'] == cluster]
                x_axis_iter = x_axis[df['clusters'] == cluster]
                cluster_data['percentage_change'] = ((cluster_data['close'] - cluster_data['open']) / cluster_data['open']) * 100
                
                # Create hover text
                hover_text = (
                    "Datetime: " + cluster_data['datetime'].astype(str) + "<br>" +
                    "Open: " + cluster_data['open'].astype(str) + "<br>" +
                    "High: " + cluster_data['high'].astype(str) + "<br>" +
                    "Low: " + cluster_data['low'].astype(str) + "<br>" +
                    "Close: " + cluster_data['close'].astype(str) + "<br>" +
                    "Percentage Change: " + cluster_data['percentage_change'].round(2).astype(str) + "%"
                )
                if plot_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(
                        x=x_axis_iter,
                        open=cluster_data['open'],
                        high=cluster_data['high'],
                        low=cluster_data['low'],
                        close=cluster_data['close'],
                        increasing_line_color=colors[cluster],  # Use color for increasing values
                        decreasing_line_color=colors[cluster],  # Use color for decreasing values
                        name=f'{def_of_colors[cluster]}',
                        text=hover_text,  # Add datetime as hover text
                        hoverinfo='text'  # Include x, y, and custom hover text
                    ))
                elif plot_type == 'Ohlc':
                    fig.add_trace(go.Ohlc(
                        x=x_axis_iter,
                        open=cluster_data['open'],
                        high=cluster_data['high'],
                        low=cluster_data['low'],
                        close=cluster_data['close'],
                        increasing_line_color=colors[cluster],  # Use color for increasing values
                        decreasing_line_color=colors[cluster],  # Use color for decreasing values
                        name=f'{def_of_colors[cluster]}',
                        text=hover_text,  # Add datetime as hover text
                        hoverinfo='text'  # Include x, y, and custom hover text
                    ))
                else:
                    return "Wrong plot_type"
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MA"], name=f"{window_size}-MA", line=dict(color="black")))
                
            # Make the line thinner
            fig.update_traces(line=dict(width=1), selector=dict(type='scatter'))

            fig.update_layout(
                title="Stock Price and Trading Volume",
                xaxis=dict(title="Time", showgrid=False),  # Shared x-axis
                yaxis=dict(title="Price"),
                yaxis2=dict(title="Volume"),
                hovermode="x unified",  # Unified hover
                margin=dict(l=50, r=50, t=60, b=50)  # Adjust margins to reduce padding
            )
            fig.show()

    def plot_moving_window_var(self, window_size=500,type=1 ,col='open',x_axis_type = 1):
        df = self.data.copy()

        df['datetime_text'] = df['datetime'].map(lambda x: f'Date: {x}')

        df['MHLDSTD'] = df['high_low_diff'].rolling(window=window_size).std() 
        df['MHLDMEAN'] = df['high_low_diff'].rolling(window=window_size).mean() 

        df['MPSTD'] = df[col].rolling(window=window_size).std()
        df['MPMEAN'] = df[col].rolling(window=window_size).mean()

        df['MVOLMEAN'] = df['volume'].rolling(window=window_size).mean()
        df['MVOLSTD'] = df['volume'].rolling(window=window_size).std()

        if x_axis_type == 1:
            x_axis = df['datetime']
        elif  x_axis_type == 2:
            x_axis = df.index
        else:
            return "Wrong x_axis_type"

        fig = make_subplots(
            rows=3, cols=1,  # Two rows, one column
            shared_xaxes=True,  # Share the x-axis
            vertical_spacing=0,  # No gap between charts
            subplot_titles=("Stock Price Episodes", "")  # Titles for each chart
        )

        fig.update(layout_xaxis_rangeslider_visible=False)
            
        if type == 1:
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MPMEAN"], name=f"{window_size}-MPMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=1, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MPSTD"], name=f"{window_size}-MPSTD", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=2, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MVOLMEAN"], name=f"{window_size}-MVOLMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=3, col=1
            )
            yax1 = dict(title="Price mean")
            yax2 = dict(title="Price std")
            yax3 = dict(title="Volume mean")
            title = f"Stock Price and Trading Volume (mean window {window_size}, column '{col}')"
        elif type == 2:
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MHLDMEAN"], name=f"{window_size}-MHLDMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=1, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MHLDSTD"], name=f"{window_size}-MHLDSTD", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=2, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MVOLMEAN"], name=f"{window_size}-MVOLMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=3, col=1
            )
            yax1 = dict(title="High-Low mean")
            yax2 = dict(title="High-Low std")
            yax3 = dict(title="Volume mean")
            title = f"Stock Price and Trading Volume (mean window {window_size}')"
        elif type == 3:
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MPMEAN"], name=f"{window_size}-MPMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=1, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MVOLMEAN"], name=f"{window_size}-MVOLMEAN", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=2, col=1
            )
            fig.add_trace(
            go.Scatter(x=x_axis, y=df["MVOLSTD"], name=f"{window_size}-MVOLSTD", line=dict(color="black"),text=df['datetime_text'],hoverinfo='text+y'),
            row=3, col=1
            )
            yax1 = dict(title="Price mean")
            yax2 = dict(title="Volume mean")
            yax3 = dict(title="Volume std")
            title = f"Stock Price and Trading Volume (mean window {window_size}')"
        else:
            return "Wrong type"
        
        # fig.update_traces(line=dict(width=1), selector=dict(type='scatter'))

        fig.update_layout(
            title=title,
            xaxis3=dict(title="Time", showgrid=False),  # Shared x-axis
            yaxis=yax1,
            yaxis2=yax2,
            yaxis3=yax3,
            hovermode="x unified",  # Unified hover
            height=800,  # Total figure height
            margin=dict(l=50, r=50, t=60, b=50)  # Adjust margins to reduce padding
        )
        fig.show()

    def hist_plot_price_in_section(self,x_axis_type=1):
        data = self.data.copy()

        def get_dates_from_section(section):
            return data[data['segment_id'] == section]['datetime']

        # Function to Create Histogram and KDE Traces
        def create_density_chart(filtered_data, column):
            # Histogram
            histogram = go.Histogram(
                x=filtered_data[column],
                name=f"{column.capitalize()} Histogram",
                nbinsx=20,  # Adjust bin size
                opacity=0.7,
                marker=dict(color=colours[filtered_data['clusters'].iloc[0]])
            )
            
            # KDE Curve
            kde = gaussian_kde(filtered_data[column])
            x_values = np.linspace(filtered_data[column].min(), filtered_data[column].max(), 100)
            kde_curve = go.Scatter(
                x=x_values,
                y=kde(x_values),
                name=f"{column.capitalize()} KDE",
                mode="lines",
                line=dict(color="black", width=2),
                yaxis="y2"

            )
            
            return [histogram, kde_curve]

        # Initial Parameters
        initial_section = 0
        initial_column = "open"

        # Filter Initial Data
        filtered_data = data[data["segment_id"] == initial_section]

        # Create Initial Chart
        fig = go.Figure()

        # Add Traces for All Combinations of Section and Columns
        sections = sorted(data["segment_id"].unique())
        clusters = data["clusters"]
        clusters_symbols = {0: '~', 1: '-', 2: '+', 3: 'o'}
        columns = ["open", "high", "low"]
        colours = {0: 'blue', 1: 'red', 2: 'green', 3: 'black'} 
        yaxis2 = "y2"
        fig.update_layout(yaxis2=dict(title="Density", overlaying="y", side="right"))

        for section in sections:
            for column in columns:
                section_data = data[data["segment_id"] == section]
                traces = create_density_chart(section_data, column)
                for trace in traces:
                    fig.add_trace(trace)

        # Initially Show Only the First Combination
        visibility_list = [
            True if (section == initial_section and column == initial_column) else False
            for section in sections
            for column in columns
            for _ in range(2)  # Two traces (Histogram and KDE) per combination
        ]
        for i, trace_visibility in enumerate(visibility_list):
            fig.data[i].visible = trace_visibility

        # Update Layout
        fig.update_layout(
            title=f"Density Chart of {initial_column.capitalize()} (Section {initial_section})",
            xaxis_title=initial_column.capitalize(),
            yaxis_title="Density",
            hovermode="x",
            height=600
        )

        # Create Dropdown Menus
        fig.update_layout(
            updatemenus=[
                # Dropdown for Section Selection
                {
                    "buttons": [
                        {
                            "label": f"{get_dates_from_section(section).min()} - {get_dates_from_section(section).max()}, ({clusters_symbols[data[data['segment_id'] == section]['clusters'].iloc[0]]})",
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        True if (sec == section and col == initial_column) else False
                                        for sec in sections
                                        for col in columns
                                        for _ in range(2)  # Two traces per combination
                                    ]
                                },
                                {"title.text": f"Density Chart of {initial_column.capitalize()} (Section {section})"}
                            ],
                            
                            # "bgcolor": "red" if clusters[section] == 0 else ("green" if clusters[section] == 1 else "blue")
                        }
                        for section in sections
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 0.8,
                    "y": 1.15
                },
                # Dropdown for Column Selection
                {
                    "buttons": [
                        {
                            "label": column.capitalize(),
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        True if (sec == initial_section and col == column) else False
                                        for sec in sections
                                        for col in columns
                                        for _ in range(2)  # Two traces per combination
                                    ]
                                },
                                {"title.text": f"Density Chart of {column.capitalize()} (Section {initial_section})"}
                            ]
                        }
                        for column in columns
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1.0,
                    "y": 1.15
                }
            ]
        )

        fig.show()

    # Unusual/Unstalbe period detection
    def outlier_count_plot(self,type=1, outlier_val=0.95,outlier_type=1,return_df=False):
        df = self.data.copy()
        df['sum_volume'] = df.groupby('segment_id')['volume'].transform('sum')
        if outlier_type == 1:
            quantile = outlier_val
            outlier = df['high_low_diff'].quantile(quantile)
            title = f'{outlier_val} quantile to clasify outlier'
        elif outlier_type == 2:
            quantile = outlier_val
            outlier = outlier_val * df['high_low_diff'].std()
            title = f'{outlier_val} std to clasify outlier'
        else:
            return "Wrong outlier_type"
        
        outlier_data = df[df['high_low_diff'] > outlier]
        df1 = outlier_data['segment_id'].value_counts().reset_index()
        df2 = df.groupby('segment_id')['volume'].sum().reset_index()
        df3 = pd.merge(df1, df2, on='segment_id', how='right')
        df3['outlier_ratio'] = (df3['count'] / df3['volume'])
        df3['outlier_ratio'] /= df3['outlier_ratio'].max()
        df4 = df[['clusters', 'segment_id']].drop_duplicates().sort_values('segment_id')
        df5 = pd.merge(df3, df4, on='segment_id', how='right')

        colours = {0: 'blue', 1: 'red', 2: 'green', 3: 'black'} # green means dropping, red means stable, blue means increasing, black means nothing

        fig = go.Figure()

        if type == 1:
            fig.add_trace(go.Bar(
                x=df5['segment_id'],
                y=df5['outlier_ratio'],
                name='Outlier Ratio',
                marker=dict(color=[colours[cluster] for cluster in df5['clusters']])
            ))

            fig.update_layout(
                title=f"Outlier Ratio by Number of Section ({title})",
                xaxis_title="Number of Section",
                yaxis_title="Outlier Ratio",
                hovermode="x unified",
                # height=500
            )
        elif type == 3:
            outlier_point = df.groupby('segment_id')['high_low_diff'].quantile(outlier_val).reset_index()
            fig.add_trace(go.Bar(
            x=outlier_point.segment_id,  # Sorted bin labels
            y=outlier_point.high_low_diff,  # Corresponding frequencies
            name='High-Low Difference',
            opacity=0.7,
            marker=dict(color=[colours[cluster] for cluster in df5['clusters']])
            ))

            fig.update_layout(
                title=f"Difference to clasify as outlier ({title})",
                xaxis_title="Number of Section",
                yaxis_title="Frequency",
                hovermode="x unified",
                # height=500
            )
        elif type == 2:
            frequency = outlier_data['segment_id'].value_counts().sort_values(ascending=False)
            fig.add_trace(go.Bar(
            x=frequency.index,  # Sorted bin labels
            y=frequency.values,  # Corresponding frequencies
            name='High-Low Difference',
            opacity=0.7,
            marker=dict(color=[colours[cluster] for cluster in df5['clusters']]),
            

            ))

            # Update layout
            fig.update_layout(
            title=f"Outlier high-low count ({title})",
            xaxis_title="Number of Section",
            yaxis_title="Frequency",
            hovermode="x unified",
            # height=500
            )

        fig.show()
        
        if return_df:
            return df.groupby('segment_id')['datetime'].agg(['min', 'max']).reset_index()


    def unstable_periods_plot(self, type=1,quan = 0.8,outlier_type = 1,outluer_val = 0.95,x_axis_type=1):
        df = self.data.copy()
        df['sum_volume'] = df.groupby('segment_id')['volume'].transform('sum')
        if outlier_type == 1:
            outlier = df['high_low_diff'].quantile(outluer_val) # !!! my add 3*std
        elif outlier_type == 2:
            outlier = outluer_val * df['high_low_diff'].std()
        else:
            return "Wrong outlier_type"
        outlier_data = df[df['high_low_diff'] > outlier]

        df1 = outlier_data['segment_id'].value_counts().reset_index()
        df2 = df.groupby('segment_id')['volume'].sum().reset_index()
        df3 = pd.merge(df1, df2, on='segment_id', how='right')
        df3['outlier_ratio'] = (df3['count'] / df3['volume'])
        df3['outlier_ratio'] /= df3['outlier_ratio'].max()

        # !!!
        if (type==1):
            # ammount of outliers in perion
            chosen_column = df3['count']
            quantile_c = chosen_column.quantile(quan)
            data_outliers = df3[chosen_column > quantile_c]
            spetial_periods = data_outliers['segment_id'].unique()
            df['is_special'] = df['segment_id'].apply(lambda x: x in spetial_periods)
        elif (type==2):
            # ratio of outliers in perion to volume
            chosen_column = df3['outlier_ratio']
            quantile_c = chosen_column.quantile(quan)
            data_outliers = df3[chosen_column > quantile_c]
            spetial_periods = data_outliers['segment_id'].unique()
            df['is_special'] = df['segment_id'].apply(lambda x: x in spetial_periods)
        elif (type==3):
            # it takes top (1 - quan) quantile of (outluer_quantile == values to be qualified as outlier for period) list of periods
            if outlier_type == 1:
                outlier_limit = df.groupby('segment_id')['high_low_diff'].quantile(outluer_val).reset_index()
                top_quan_outlier_limit = outlier_limit['high_low_diff'].sort_values().quantile(quan)
                spetial_periods = outlier_limit[outlier_limit['high_low_diff'] > top_quan_outlier_limit]['segment_id'].unique()
                df['is_special'] = df['segment_id'].apply(lambda x: x in spetial_periods)
            else:
                return "Wrong type"
        else:
            return "Wrong type"
        
        fig = go.Figure()
        for cluster in df['clusters'].unique():
            cluster_data = df[df['clusters'] == cluster]
            for is_special in cluster_data['is_special'].unique():
                special_data = cluster_data[cluster_data['is_special'] == is_special]
                if x_axis_type == 1:
                    x_axis = special_data['datetime']
                elif x_axis_type == 2:
                    x_axis = special_data.index
                else:
                    return "Wrong x_axis_type"
                color = 'red' if is_special else 'grey'
                fig.add_trace(go.Ohlc(
                    x=x_axis,
                    open=special_data['open'],
                    high=special_data['high'],
                    low=special_data['low'],
                    close=special_data['close'],
                    increasing_line_color=color,  # Use color for increasing values
                    decreasing_line_color=color,  # Use color for decreasing values
                    name=f'{"Special" if is_special else "Normal"}'
                ))

        fig.update_layout(
            title="Difference Between High and Low Prices (Outliers most diverse periods)",
            xaxis_title="Time",
            yaxis_title="Stock Value",
            hovermode="x unified"
            )

        fig.show()

    # Table Generation Methods
    def segment_length_table(self,considere_type=False):
        data = self.data.copy()
        if considere_type:
            segment_size_counts = data.groupby(["segment_id","clusters"]).size().reset_index()
            segment_size_counts.columns = ['segment_id', 'clusters', 'Size']
            segment_size_counts = segment_size_counts[['clusters', 'Size']].value_counts().reset_index()
            segment_size_counts.columns = ['Segment type','Segment Size', 'Count']

        else:
            segment_size_counts = data.groupby("segment_id").size().value_counts().reset_index().sort_values("index").reset_index(drop=True)
            segment_size_counts.columns = ['Segment Size', 'Count']
        return segment_size_counts

    def high_low_outlier_count_table(self,outlier_type=1,outlier_val=0.95):
        df = self.data.copy()
        if outlier_type == 1:
            outlier = df['high_low_diff'].quantile(outlier_val)
        elif outlier_type == 2:
            outlier = outlier_val * df['high_low_diff'].std()
        else:
            return "Wrong outlier_type"
        outlier_data = df[df['high_low_diff'] > outlier]
        table = outlier_data['clusters'].value_counts().reset_index()
        table.columns = ['Cluster', 'Outlier Count']
        return table

    def cluster_cumulative_stats_table(self):
        data = self.data.copy()
        cluster_stats = data.groupby("clusters").agg({
            "open": ["mean", "std"],
            "high": ["mean", "std"],
            "low": ["mean", "std"],
            "close": ["mean", "std"],
            "volume": ["mean", "std", "sum"]
        }).rename(columns={
            "mean": "Mean",
            "std": "Std",
            "sum": "Total"
        })
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        return cluster_stats

    def stats_for_each_segment_table(self):
        data = self.data.copy()
        segment_stats = data.groupby("segment_id").agg({
            "clusters": "first",
            "datetime": ["min", "max"],
            "volume": "sum",
            "open": "first",
            "close": "last"
        })
        segment_stats.columns = ["Cluster Type","Start Time", "End Time", "Total Volume", "First Open", "Last Close"]
        segment_stats["Duration (mins)"] = (segment_stats["End Time"] - segment_stats["Start Time"]).dt.total_seconds() / 60
        segment_stats["Price Change"] = segment_stats["Last Close"] - segment_stats["First Open"]
        segment_stats["Price Change (%)"] = (segment_stats["Last Close"] - segment_stats["First Open"]) / segment_stats["First Open"] * 100

        return segment_stats

    def transition_count_table(self):
        data = self.data.copy()
        transitions = data[["segment_id", "clusters"]].drop_duplicates().reset_index(drop=True)
        transitions["Next Cluster"] = transitions["clusters"].shift(-1)
        transition_counts = transitions.groupby(["clusters", "Next Cluster"]).size().reset_index(name="Count")
        return transition_counts

    def stats_per_dat_table(self):
        data = self.data.copy()
        data["date"] = data["datetime"].dt.date
        daily_stats = data.groupby("date").agg({
            "open": "mean",
            "high": "max",
            "low": "min",
            "close": "mean",
            "volume": "sum"
        }).rename(columns={
            "open": "Avg Open",
            "high": "Max High",
            "low": "Min Low",
            "close": "Avg Close",
            "volume": "Total Volume"
        })

        return daily_stats
