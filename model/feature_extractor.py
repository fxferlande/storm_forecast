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
        self.tsfresh_params = []

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


tsfresh_params = {'abs_energy': None,
                  'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 20},
                                       {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2},
                                       {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10},
                                       {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5},
                                       {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 10},
                                       {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 20},
                                       {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 5},
                                       {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 20},
                                       {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 2},
                                       {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 10},
                                       {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5},
                                       {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 20},
                                       {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 2},
                                       {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 10},
                                       {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 5},
                                       {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2}],
                  'last_location_of_minimum': None,
                  'linear_trend': [{'attr': 'intercept'},
                                   {'attr': 'pvalue'},
                                   {'attr': 'rvalue'},
                                   {'attr': 'slope'},
                                   {'attr': 'stderr'}],
                  'maximum': None,
                  'mean': None,
                  'mean_change': None,
                  'median': None,
                  'minimum': None,
                  'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                  'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                                      {'coeff': 1, 'attr': 'angle'},
                                      {'coeff': 1, 'attr': 'imag'},
                                      {'coeff': 1, 'attr': 'real'},
                                      {'coeff': 2, 'attr': 'real'},
                                      {'coeff': 2, 'attr': 'angle'},
                                      {'coeff': 0, 'attr': 'real'},
                                      {'coeff': 2, 'attr': 'abs'},
                                      {'coeff': 1, 'attr': 'abs'},
                                      {'coeff': 0, 'attr': 'angle'}],
                  'quantile': [{'q': 0.1},
                               {'q': 0.2},
                               {'q': 0.4},
                               {'q': 0.3},
                               {'q': 0.6},
                               {'q': 0.7},
                               {'q': 0.8},
                               {'q': 0.9}],
                  'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 3},
                                             {'num_segments': 10, 'segment_focus': 0},
                                             {'num_segments': 10, 'segment_focus': 1},
                                             {'num_segments': 10, 'segment_focus': 2}],
                  'first_location_of_maximum': None,
                  'percentage_of_reoccurring_values_to_all_values': None,
                  'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40},
                                          {'f_agg': 'var', 'maxlag': 40},
                                          {'f_agg': 'median', 'maxlag': 40}],
                  'value_count': [{'value': 0}, {'value': -1}],
                  'time_reversal_asymmetry_statistic': [{'lag': 1}],
                  'sum_values': None,
                  'binned_entropy': [{'max_bins': 10}],
                  'c3': [{'lag': 1}],
                  'change_quantiles': [{'f_agg': 'mean', 'isabs': False, 'qh': 0.6, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 0.8, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 0.8, 'ql': 0.2},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.2},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 0.4, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.4},
                                       {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.6},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.2},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.0},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.6},
                                       {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.4},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.2},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.2},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.2},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 0.4, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 0.4, 'ql': 0.0},
                                       {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.0}],
                  'range_count': [{'max': 1, 'min': -1}, {'max': 1000000000000.0, 'min': 0}],
                  'ratio_value_number_to_time_series_length': None,
                  'sum_of_reoccurring_values': None,
                  'ratio_beyond_r_sigma': [{'r': 1}, {'r': 0.5}, {'r': 1.5}],
                  'approximate_entropy': [{'m': 2, 'r': 0.5},
                                          {'m': 2, 'r': 0.3},
                                          {'m': 2, 'r': 0.1},
                                          {'m': 2, 'r': 0.7},
                                          {'m': 2, 'r': 0.9}],
                  'sum_of_reoccurring_data_points': None,
                  'index_mass_quantile': [{'q': 0.8},
                                          {'q': 0.9},
                                          {'q': 0.3},
                                          {'q': 0.7},
                                          {'q': 0.1},
                                          {'q': 0.6}],
                  'autocorrelation': [{'lag': 3}, {'lag': 1}],
                  'number_cwt_peaks': [{'n': 5}, {'n': 1}],
                  'symmetry_looking': [{'r': 0.2},
                                       {'r': 0.25},
                                       {'r': 0.30000000000000004},
                                       {'r': 0.35000000000000003},
                                       {'r': 0.4},
                                       {'r': 0.9500000000000001},
                                       {'r': 0.5},
                                       {'r': 0.55},
                                       {'r': 0.6000000000000001},
                                       {'r': 0.65},
                                       {'r': 0.7000000000000001},
                                       {'r': 0.75},
                                       {'r': 0.8},
                                       {'r': 0.8500000000000001},
                                       {'r': 0.9},
                                       {'r': 0.45},
                                       {'r': 0.15000000000000002},
                                       {'r': 0.1},
                                       {'r': 0.05}],
                  'variance_larger_than_standard_deviation': None,
                  'large_standard_deviation': [{'r': 0.35000000000000003},
                                               {'r': 0.15000000000000002},
                                               {'r': 0.2},
                                               {'r': 0.25},
                                               {'r': 0.30000000000000004},
                                               {'r': 0.05},
                                               {'r': 0.1},
                                               {'r': 0.4},
                                               {'r': 0.45}],
                  'mean_abs_change': None,
                  'absolute_sum_of_changes': None,
                  'variance': None,
                  'standard_deviation': None,
                  'has_duplicate_min': None,
                  'cid_ce': [{'normalize': False}, {'normalize': True}],
                  'has_duplicate': None,
                  'last_location_of_maximum': None,
                  'first_location_of_minimum': None,
                  'count_above_mean': None,
                  'longest_strike_below_mean': None,
                  'longest_strike_above_mean': None,
                  'count_below_mean': None,
                  'has_duplicate_max': None,
                  'number_crossing_m': [{'m': 1}, {'m': 0}, {'m': -1}],
                  'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
                  'augmented_dickey_fuller': [{'attr': 'teststat'}, {'attr': 'pvalue'}],
                  'friedrich_coefficients': [{'m': 3, 'r': 30, 'coeff': 1},
                                             {'m': 3, 'r': 30, 'coeff': 2}],
                  'spkt_welch_density': [{'coeff': 2}],
                  'partial_autocorrelation': [{'lag': 1}, {'lag': 3}, {'lag': 2}],
                  'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}],
                  'sample_entropy': None,
                  'kurtosis': None}
