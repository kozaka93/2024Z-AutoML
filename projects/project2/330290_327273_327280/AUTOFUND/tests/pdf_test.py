from numerai_automl.raport_manager.raport_manager import RaportManager
from numerai_automl.visual.cumsum_cor_plot import CumSumCorPlot
import pandas as pd
from numerai_automl.visual.radar_plot import RadarPlot

def test_cumsum_cor_plot():
    df = pd.read_csv("return_data.csv")
    df = df[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega", 'era',  "target"]]
    csp = CumSumCorPlot(df)

    fig = csp.get_plot()


    rm = RaportManager([fig])
    rm.generate_html("cumsum_cor_plot.html")


    

    # fig.show()
    # fig.savefig('cumsum_cor_plot.png')

def test_radar_plot():
    df = pd.read_csv("return_data_for_scoring.csv", index_col=0)

    # df = df[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_omega", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", 'era',  "target",]]
    df = df.loc[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega"]]
    print(df)
    rp = RadarPlot(df)
    fig = rp.get_plot()
    rm = RaportManager([fig])
    rm.generate_html("radar_plot.html")

def test_all():
    df = pd.read_csv("return_data.csv")
    df = df[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega", 'era',  "target"]]
    csp = CumSumCorPlot(df)

    fig = csp.get_plot()

    df = pd.read_csv("return_data_for_scoring.csv", index_col=0)

    df = df.loc[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega"]]
    rp = RadarPlot(df)
    fig2 = rp.get_plot()
    rm = RaportManager([fig, fig2])
    rm.generate_html("all.html")

if __name__ == "__main__":
    test_all()