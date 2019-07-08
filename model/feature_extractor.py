import pickle
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
pd.set_option('mode.chained_assignment', None)


class FeatureExtractor(object):
    def __init__(self, len_sequences=4):
        self.constant_fields = ["initial_max_wind", "basin", "nature"]
        self.scalar_fields = ['instant_t', 'windspeed', 'latitude', 'longitude', 'Jday_predictor',
                              'max_wind_change_12h', 'dist2land']
        self.spatial_fields = ["u", "v", "sst", "slp", "hum", "z", "vo700"]
        self.scaling_values = pd.DataFrame(index=self.spatial_fields + self.scalar_fields +
                                           self.constant_fields, columns=["mean", "std"],
                                           dtype=float)
        self.gluon_estimator = SimpleFeedForwardEstimator(
            prediction_length=1,
            freq="4H",
            trainer=Trainer(ctx="cpu",
                            epochs=100,
                            learning_rate=1e-4,
                            num_batches_per_epoch=128))
        self.max_len = len_sequences

    def make_sequence(self, X_df, field):
        ids = X_df.stormid.unique()
        f_data = X_df[[field, "stormid"]]
        f_data[field] = (f_data[field] - self.scaling_values.loc[field, "mean"]) / \
            self.scaling_values.loc[field, "std"]
        bloc = np.empty((len(X_df), self.max_len))
        for id in ids:
            ligne = f_data[f_data["stormid"] == id][field]
            seq_line = np.empty((len(ligne), self.max_len))
            for i in range(self.max_len):
                seq_line[:, self.max_len - 1 -
                         i] = np.array(([0] * i + list(ligne))[0:len(ligne)])
            pos = list(X_df["stormid"]).index(id)
            bloc[pos:pos + len(ligne), :] = seq_line
        return bloc

    def bearing(self, line):
        """Compute bearing in degrees for one line of dataframe

            brearing = atan2(
                sin(Δlon).cos(lat2),
                cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlon)
            )

        Source: https://www.movable-type.co.uk/scripts/latlong.html

        Parmeters
        ---------
        lon1, lat1, lon2, lat2: list-like
            coordinates in degrees

        Returns
        -------
        bearing: list-like
            bearing in degrees in [-180, 180]
        """
        if line.instant_t == 0:
            lon1, lat1 = line["longitude"], line["latitude"]
            lon2, lat2 = line["longitude"], line["latitude"]
        else:
            lon1, lat1 = line["longitude"], line["latitude"]
            lon2, lat2 = line["longitude_before"], line["latitude_before"]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        delta_lon = lon2 - lon1
        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

        bearing = np.degrees(np.arctan2(x, y))

        return bearing

    def compute_bearing(self, X_df):
        X_df["longitude_before"] = X_df["longitude"].shift()
        X_df["latitude_before"] = X_df["latitude"].shift()
        X_df["bearing"] = X_df.apply(self.bearing, axis=1)
        X_df["bearing_cos"] = np.cos(X_df["bearing"])
        X_df["bearing_sin"] = np.sin(X_df["bearing"])
        X_df.drop("bearing", axis=1, inplace=True)
        X_df.drop("longitude_before", axis=1, inplace=True)
        X_df.drop("latitude_before", axis=1, inplace=True)
        if "bearing_cos" not in self.scalar_fields:
            self.scalar_fields += ["bearing_cos", "bearing_sin"]
        return X_df

    def compute_gluonts(self, X_df):
        dataset = ListDataset([{'target': x, 'start': 0}
                               for x in self.make_sequence(X_df, "windspeed")],
                              freq=self.gluon_estimator.freq)
        forecast = make_evaluation_predictions(dataset=dataset,
                                               predictor=self.gluon_predictor,
                                               num_eval_samples=self.max_len)
        forecast = list(forecast[0])
        predicted = [f.mean[0] for f in forecast]
        X_df["gluonts_pred"] = predicted
        if "gluonts_pred" not in self.scalar_fields:
            self.scalar_fields += ["gluonts_pred"]
        return X_df

    def cross_features(self, X_df):
        X_df["mw12_lon"] = X_df["max_wind_change_12h"]*X_df["longitude"]
        if "mw12_lon" not in self.scalar_fields:
            self.scalar_fields.append("mw12_lon")
        X_df["mw12_lat"] = X_df["max_wind_change_12h"]*X_df["latitude"]
        if "mw12_lat" not in self.scalar_fields:
            self.scalar_fields.append("mw12_lat")
        X_df["mw12_imw"] = - X_df["max_wind_change_12h"]*X_df["initial_max_wind"]
        if "mw12_imw" not in self.scalar_fields:
            self.scalar_fields.append("mw12_imw")
        # X_df["dist_imw"] = X_df["dist2land"]*X_df["initial_max_wind"]
        # if "dist_imw" not in self.scalar_fields:
        #     self.scalar_fields.append("dist_imw")
        return X_df

    def fit(self, X_df, y):
        print("Starting FeatureExtractor fit")
        self.scaling_values.loc["windspeed", "mean"] = X_df["windspeed"].mean()
        self.scaling_values.loc["windspeed", "std"] = X_df["windspeed"].std()
        train_ds = ListDataset([{'target': x, 'start': 0}
                                for x in self.make_sequence(X_df, "windspeed")],
                               freq=self.gluon_estimator.freq)
        self.gluon_predictor = self.gluon_estimator.train(train_ds)
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11)
            field_grids.append(f_data)
        for f, field in enumerate(self.spatial_fields):
            self.scaling_values.loc[field, "mean"] = np.nanmean(field_grids[f])
            self.scaling_values.loc[field, "std"] = np.nanstd(field_grids[f])
        X_df = self.compute_bearing(X_df)
        X_df = self.compute_gluonts(X_df)
        X_df = self.cross_features(X_df)
        for field in self.scalar_fields:
            self.scaling_values.loc[field, "mean"] = X_df[field].mean()
            self.scaling_values.loc[field, "std"] = X_df[field].std()
        for field in self.constant_fields:
            self.scaling_values.loc[field, "mean"] = X_df[field].mean()
            self.scaling_values.loc[field, "std"] = X_df[field].std()
        print("Fitting FeatureExtractor done")

    def transform(self, X_df):
        print("Starting FeatureExtractor transform")
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11)
            field_grids.append(
                (f_data - self.scaling_values.loc[field, "mean"]) / self.scaling_values.loc[field,
                                                                                            "std"])
            field_grids[-1][np.isnan(field_grids[-1])] = 0
            print("Field ", field, "just done")
        norm_data = np.stack(field_grids, axis=-1)
        print("NaN values found in spatial: ", np.isnan(norm_data).any())

        scalar = self.compute_bearing(X_df)
        scalar = self.compute_gluonts(scalar)
        scalar = self.cross_features(scalar)
        scalar["stormid"] = X_df["stormid"]
        final_scalar = np.empty((len(scalar), self.max_len, len(self.scalar_fields)))
        for field in self.scalar_fields:
            bloc = self.make_sequence(scalar, field)
            final_scalar[:, :, self.scalar_fields.index(field)] = bloc
            print("Field ", field, "just done")
        print("NaN values found in scalar: ", np.isnan(final_scalar).any())
        if np.isnan(final_scalar).any():
            np.save('/home/ubuntu/documents/storm_forecast/data/scalar_nan', final_scalar)
            X_df.to_csv('/home/ubuntu/documents/storm_forecast/data/df_nan.csv', index=None)
            with open('/home/ubuntu/documents/storm_forecast/data/fe', 'wb') as fe:
                pickle.dump(self.scaling_values, fe)
        norm_scalar = np.nan_to_num(final_scalar)

        norm_constant = X_df[self.constant_fields]
        for field in self.constant_fields:
            norm_constant[field] = (norm_constant[field] - self.scaling_values.loc[field, "mean"]) / \
                self.scaling_values.loc[field, "std"]
        norm_constant = norm_constant.values
        print("FeatureExtractor transform done")
        return [norm_data, norm_data, norm_scalar, norm_scalar, norm_constant]
