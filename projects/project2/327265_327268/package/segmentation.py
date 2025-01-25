import ruptures as rpt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sktime.annotation.igts import InformationGainSegmentation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from package.preprocessing import SingleStockDataPreprocessor

class Segmentator3000:
    """
    A class for segmenting time-series data and performing clustering on segments.
    Provides methods for segmentation using various algorithms and visualizing results.
    """

    def __init__(self, data, price_columns, volume_column):
        """
        Initializes the Segmentator with preprocessed data and segmentation parameters.

        Args:
            data (pd.DataFrame): Input time-series data.
            price_columns (list): List of price column names.
            volume_column (str): Column name for volume data.
        """
        self.preprocessor = SingleStockDataPreprocessor(data, price_columns, volume_column)
        self.preprocessor.preprocess()
        self.data = self.preprocessor.data[['RSI_level', 'MA_diff', 'MA_indicator']]
        self.segmented_data = None
        self.n = len(self.preprocessor.data)
        self.best_segmentation = {
            'breakpoints': None,
            'model': None,
            'params': None,
            'silhouette_score': None,
            'n_clusters': None,
            'clusters': None
        }

    def optimal_kmeans_clusters(self, X, max_k=10):
        """
        Determines the optimal number of clusters for KMeans using silhouette scores.

        Args:
            X (np.ndarray): Feature data for clustering.
            max_k (int): Maximum number of clusters to evaluate.

        Returns:
            tuple: Best number of clusters, silhouette score, and cluster assignments.
        """
        sil_scores = []
        clusters_list = []
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        for k in range(3, min(len(X), 4)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X)
            sil_score = silhouette_score(X, kmeans.labels_)
            sil_scores.append(sil_score)
            clusters_list.append(clusters)

        if not clusters_list:
            return 0, 0, None

        best_k = sil_scores.index(max(sil_scores)) + 3
        return best_k, max(sil_scores), clusters_list[sil_scores.index(max(sil_scores))]

    def calculate_rsi(self, prices):
        """
        Calculates the Relative Strength Index (RSI) for a given price series.

        Args:
            prices (list): List of price values.

        Returns:
            float: RSI value.
        """
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def cluster_segments(self, breakpoints):
        """
        Clusters segments of data based on calculated statistics (e.g., RSI, slope).

        Args:
            breakpoints (list): List of indices marking segment boundaries.

        Returns:
            tuple: Best number of clusters, silhouette score, and cluster assignments.
        """
        scaler = MinMaxScaler()
        data = self.preprocessor.prices

        if breakpoints[-1] != len(data):
            breakpoints.append(len(data))
        breakpoints = [0] + breakpoints

        clusters = []

        for i, bk in enumerate(breakpoints[1:]):
            segment = data[breakpoints[i]:breakpoints[i + 1]]
            segment = np.array(segment).reshape(-1, 1)

            x = np.arange(len(segment)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, scaler.fit_transform(segment) * len(x))
            y_pred = model.predict(x)

            mse = mean_squared_error(segment, y_pred)
            slope = model.coef_[0]

            rsi = self.calculate_rsi(data[breakpoints[i]:breakpoints[i + 1]])

            clusters.append(pd.DataFrame({
                'slope': slope,
                'rsi': rsi
            }, index=[i]))

        d = pd.concat(clusters)

        # d['rsi_quantile'] = pd.qcut(d['rsi'], q=5, duplicates='drop', labels=False)
        # d['slope_quantile'] = pd.qcut(d['slope'], q=5, duplicates='drop', labels=False)
        bins = pd.qcut(d['rsi'], q=5, duplicates='drop', retbins=True)[1]
        d['rsi_quantile'] = pd.cut(d['rsi'], bins=bins, labels=[0, 0, 1, 2, 2], ordered=False)
        bins = pd.qcut(d['slope'], q=5, duplicates='drop', retbins=True)[1]
        d['slope_quantile'] = pd.cut(d['slope'], bins=bins, labels=[0, 0, 1, 2, 2], ordered=False)   
        d_scaled = scaler.fit_transform(d)
        d = pd.DataFrame(d_scaled, columns=d.columns)

        d = d.fillna(0)
        # d_scaled = scaler.fit_transform(d)
        # d = pd.DataFrame(d_scaled, columns=d.columns).fillna(0)


        best_k, sil_score, clusters_list = self.optimal_kmeans_clusters(X = d, max_k=10) 
        # display(d)
        d['slope_quantile'] = d['slope_quantile']*2
        d['cluster'] = clusters_list
        mean0 = d.loc[d.cluster == 0]['slope_quantile'].mean()
        mean1 = d.loc[d.cluster == 1]['slope_quantile'].mean()
        mean2 = d.loc[d.cluster == 2]['slope_quantile'].mean()
        vals = {mean0:0, mean1:1, mean2:2}
        v = sorted(list(vals.keys()))
        clus = {
            int(vals[v[1]]):0,
            int(vals[min(vals.keys())]):1,
            int(vals[max(vals.keys())]):2
            }

        return best_k, sil_score, [clus[int(i)] for i in clusters_list]
 

    def segment_Binseg(self, min_seg_length=None):
        """
        Performs segmentation using Binary Segmentation (Binseg).

        Args:
            min_seg_length (int): Minimum segment length.

        Returns:
            dict: Segmentation results, including breakpoints, parameters, and clustering information.
        """
        self.min_seg_length = min_seg_length or int(0.02 * self.n)

        data = self.data
        m = 0
        best_breakpoints = None
        best_type = None
        best_n_bkpts = None
        best_clust = None
        best_n_clusters = None

        Binseg_grid = {
            'n_bkps': list(range(
                int(self.n / self.min_seg_length / 2.5),
                int(self.n / self.min_seg_length / 1.5),
                max(1, int(self.n / self.min_seg_length / 6.25))
            )),
            'model': ['l2', 'rbf', 'normal', 'rank']
        }

        for model_binseg in Binseg_grid['model']:
            model = rpt.Binseg(model=model_binseg).fit(data)
            for n_bkps in Binseg_grid['n_bkps']:
                breakpoints = model.predict(n_bkps)
                if not breakpoints:
                    continue

                n_clusters, silhouette, clusters = self.cluster_segments(breakpoints)
                if m < silhouette:
                    m = silhouette
                    best_breakpoints = breakpoints
                    best_type = model_binseg
                    best_n_bkpts = n_bkps
                    best_n_clusters = n_clusters
                    best_clust = clusters
        best_segmentation = {
            'breakpoints': best_breakpoints,
            'params': {'model': best_type, 'n_bkps': best_n_bkpts},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }
        l = [0] + best_segmentation['breakpoints']
        if l[-1] != len(self.preprocessor.prices):
            l.append(len(self.preprocessor.prices))

        l = np.diff(np.array(l))
        x = []
        for i in range(len(l)):
            x.extend([int(best_segmentation['clusters'][i])] * l[i])
        segmented_data = self.preprocessor.original_data
        segmented_data['clusters'] = x

        return {
            'data': segmented_data,
            'breakpoints': best_breakpoints,
            'params': {'model': best_type, 'n_bkps': best_n_bkpts},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }

    def segment_IGS(self, min_seg_length=None):
        """
        Performs segmentation using Information Gain Segmentation (IGS).

        Args:
            min_seg_length (int): Minimum segment length.

        Returns:
            dict: Segmentation results, including breakpoints, parameters, and clustering information.
        """
        self.min_seg_length = min_seg_length or int(0.02 * self.n)

        data = self.data
        m = 0
        best_breakpoints = None
        best_k_max = None
        best_step = None
        best_clust = None
        best_n_clusters = None

        IGS_grid = {
            'k_max': list(range(
                int(self.n / self.min_seg_length / 2.5),
                int(self.n / self.min_seg_length / 1.5),
                max(1, int(self.n / self.min_seg_length / 6.25))
            )),
            'step': list(range(
                self.min_seg_length,
                int(1.5 * self.min_seg_length),
                max(1, int(0.5 * self.min_seg_length) // 5)
            ))
        }

        for k_max in IGS_grid['k_max']:
            for step in IGS_grid['step']:
                model = InformationGainSegmentation(k_max=k_max, step=step)
                breakpoints = model.fit_predict(data.to_numpy())
                array = [0] + np.diff(breakpoints).tolist()
                array = np.nonzero(array)[0].tolist()
                n_clusters, silhouette, clusters = self.cluster_segments(array)
                if m < silhouette:
                    m = silhouette
                    best_breakpoints = array
                    best_k_max = k_max
                    best_step = step
                    best_n_clusters = n_clusters
                    best_clust = clusters
        best_segmentation = {
            'breakpoints': best_breakpoints,
            'params': {'k_max': best_k_max, 'step': best_step},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }
        l = [0] + best_segmentation['breakpoints']
        if l[-1] != len(self.preprocessor.prices):
            l.append(len(self.preprocessor.prices))

        l = np.diff(np.array(l))
        x = []
        for i in range(len(l)):
            x.extend([int(best_segmentation['clusters'][i])] * l[i])
        segmented_data = self.preprocessor.original_data
        segmented_data['clusters'] = x
        return {
            'data': segmented_data,
            'breakpoints': best_breakpoints,
            'params': {'k_max': best_k_max, 'step': best_step},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }

    def segment_KernelCPD(self, min_seg_length=None):
        """
        Performs segmentation using Kernel Change Point Detection (KernelCPD).

        Args:
            min_seg_length (int): Minimum segment length.

        Returns:
            dict: Segmentation results, including breakpoints, parameters, and clustering information.
        """
        self.min_seg_length = min_seg_length or int(0.02 * self.n)

        data = self.data
        m = 0
        best_breakpoints = None
        best_min_size = None
        best_n_bkps = None
        best_clust = None
        best_n_clusters = None

        KernelCPD_grid = {
            'n_bkps': list(range(
                int(self.n / self.min_seg_length / 2.5),
                int(self.n / self.min_seg_length / 1.5),
                max(1, int(self.n / self.min_seg_length / 6.25))
            )),
            'min_size': list(range(
                self.min_seg_length,
                int(1.5 * self.min_seg_length),
                max(1, int(0.5 * self.min_seg_length) // 5)
            ))
        }

        for min_size in KernelCPD_grid['min_size']:
            model = rpt.KernelCPD(min_size=min_size).fit(data.to_numpy())
            for n_bkps in KernelCPD_grid['n_bkps']:
                breakpoints = model.predict(n_bkps)
                if not breakpoints:
                    continue
                n_clusters, silhouette, clusters = self.cluster_segments(breakpoints)
                if m < silhouette:
                    m = silhouette
                    best_breakpoints = breakpoints
                    best_min_size = min_size
                    best_n_bkps = n_bkps
                    best_n_clusters = n_clusters
                    best_clust = clusters
        segmented_data = self.preprocessor.original_data

        best_segmentation = {
            'breakpoints': best_breakpoints,
            'params': {'n_bkps': best_n_bkps, 'min_size': best_min_size},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }

        l = [0] + best_segmentation['breakpoints']
        if l[-1] != len(self.preprocessor.prices):
            l.append(len(self.preprocessor.prices))

        l = np.diff(np.array(l))
        x = []
        for i in range(len(l)):
            x.extend([int(best_segmentation['clusters'][i])] * l[i])

        segmented_data['clusters'] = x
        return {
            'data': segmented_data,
            'breakpoints': best_breakpoints,
            'params': {'n_bkps': best_n_bkps, 'min_size': best_min_size},
            'silhouette_score': m,
            'n_clusters': best_n_clusters,
            'clusters': best_clust
        }

    def segment(self, min_seg_length=None):
        """
        Executes segmentation using multiple algorithms and selects the best result.

        Args:
            min_seg_length (int): Minimum segment length.

        Updates:
            self.best_segmentation (dict): Stores the best segmentation result.
        """
        self.min_seg_length = min_seg_length or int(0.02 * self.n)

        models = ['Binseg', 'IGS', 'KernelCPD']
        segmentations = [
            self.segment_Binseg(self.min_seg_length),
            self.segment_IGS(self.min_seg_length),
            self.segment_KernelCPD(self.min_seg_length)
        ]

        best_segmentation = max(segmentations, key=lambda x: x['silhouette_score'])
        self.best_segmentation = best_segmentation
        self.best_segmentation['model'] = models[segmentations.index(best_segmentation)]

        self.plot_segmentation(best_segmentation['breakpoints'], best_segmentation['clusters'])

        self.segmented_data = self.preprocessor.original_data

        l = [0] + best_segmentation['breakpoints']
        if l[-1] != len(self.preprocessor.prices):
            l.append(len(self.preprocessor.prices))

        l = np.diff(np.array(l))
        x = []
        for i in range(len(l)):
            x.extend([int(best_segmentation['clusters'][i])] * l[i])

        self.segmented_data['clusters'] = x
    
        return self.segmented_data

    def plot_segmentation(self, breakpoints, clusters=None):
        """
        Visualizes the segmented data and clusters.

        Args:
            breakpoints (list): List of segment breakpoints.
            clusters (list, optional): Cluster assignments for segments.
        """
        prices = self.preprocessor.prices

        if clusters is None:
            plt.plot(prices)
            for b in breakpoints:
                plt.axvline(b, color='red')
            plt.show()
        else:
            if breakpoints[-1] != len(prices):
                breakpoints.append(len(prices))
            b = [0] + breakpoints
            b = np.diff(np.array(b))
            cl = []
            cluster_labels = {0: 'Stable', 1: 'Downtrend', 2: 'Uptrend'}
            for i, br in enumerate(b):
                cl.extend([clusters[i]] * br)
                # self.preprocessor.data.index.astype(str)
            # plt.scatter(x=range(len(self.data)), y=prices, c=cl, s=1)
            # handles = [
            #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(key)), markersize=10, label=label)
            #     for key, label in cluster_labels.items()
            # ]

            # plt.legend(handles=handles, title="Trends")
            # plt.gca().xaxis.set_visible(False)
            # plt.show()
            scatter = plt.scatter(x=range(len(prices)), y=prices[:len(cl)], c=cl, s=1, cmap='viridis')

            # Manually create the legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(key)), markersize=10, label=label)
                for key, label in cluster_labels.items()
            ]

            plt.legend(handles=handles, title="Trends")

            # Customize plot
            plt.gca().xaxis.set_visible(False)
            plt.ylabel("Prices")
            plt.title("Price Trends with Cluster Labels")
            plt.show()
                