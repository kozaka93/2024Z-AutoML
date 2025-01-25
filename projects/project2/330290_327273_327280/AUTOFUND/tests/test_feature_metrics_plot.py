import pandas as pd
from numerai_automl.visual.feature_metrics_plot import FeatureMetricsPlot


def test_feature_metrics_plot():
    # Data for the DataFrame
    data = {
        "mean": [0.001558, 0.001427, 0.001414, 0.001326, 0.001110, 0.000940, 0.000903, 0.000822, 0.000775, 0.000643,
                 0.000625, 0.000619, 0.000598, 0.000541],
        "std": [0.009337, 0.007844, 0.006820, 0.007519, 0.007513, 0.007362, 0.008056, 0.007658, 0.006226, 0.007052,
                0.007955, 0.006414, 0.007477, 0.005549],
        "sharpe": [0.166830, 0.181920, 0.207414, 0.176317, 0.147776, 0.127706, 0.112116, 0.107345, 0.124522, 0.091156,
                   0.078606, 0.096574, 0.079913, 0.097476],
        "max_drawdown": [-0.077963, -0.037688, -0.034116, -0.041731, -0.056087, -0.054749, -0.136916, -0.075581,
                         -0.036624, -0.087972, -0.100928, -0.033173, -0.090173, -0.049633],
        "delta": [0.000433, 0.002654, 0.000354, 0.001003, 0.001766, 0.000607, 0.002223, 0.000625, 0.001334, 0.000397,
                  0.001280, 0.000904, 0.000665, 0.001854]
    }

    index = [
        "feature_rabbinism_molluscoid_cichlid",
        "feature_shaded_hallucinatory_dactylology",
        "feature_paracelsian_capable_cosmography",
        "feature_faltering_bashful_cluny",
        "feature_pistachio_atypical_malison",
        "feature_untidier_cherished_abbacy",
        "feature_subalpine_apothegmatical_ajax",
        "feature_unpavilioned_dear_cooly",
        "feature_larine_underfloor_polynomial",
        "feature_cornier_gular_vespucci",
        "feature_cycloid_zymotic_galloway",
        "feature_elusive_vapoury_accomplice",
        "feature_crocodilian_tied_twink",
        "feature_gowned_undiluted_islay"
    ]
    df = pd.DataFrame(data, index=index)
    print("DATA:")
    print(df)
    print("BAR PLOT:")
    rp = FeatureMetricsPlot(df)
    fig = rp.get_plot()
    fig.show()

if __name__ == '__main__':
    test_feature_metrics_plot()
