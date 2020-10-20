import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
pd.set_option('mode.chained_assignment', None)


class FeatureExtractor(object):
    def __init__(self, len_sequences: int = 10):
        self.dummy_field = ["nature", 'basin']
        self.constant_fields = ['initial_max_wind']
        self.scalar_fields = ['instant_t', 'windspeed', 'Jday_predictor',
                              'max_wind_change_12h', 'dist2land']
        self.spatial_fields = ["u", "v", "sst", "slp", "hum", "z", "vo700"]
        self.scaling_values = pd.DataFrame(index=self.spatial_fields +
                                           self.scalar_fields +
                                           self.constant_fields,
                                           columns=["mean", "std"],
                                           dtype=float)
        self.binarizer = {}
        self.len_sequences = len_sequences
        self.num_dummy = 0

    def make_sequence(self, X_df: pd.DataFrame, field: str,
                      padding_value: int = -100) -> np.ndarray:
        """
        Selects the field out of the dataframe X_df, and computes sequences of
        length self.len_sequences for each storm. The padding value is used for
        the beginning of the sequence.

        Args:
            X_df   (pd.DataFrame):   source DataFrame
            field           (str):   name of the field to select
            padding_value   (int):   value to fill at the begining of sequences

        Returns:
            np.ndarray:  Array of size (len(X_df), self.len_sequences)
        """
        ids = X_df.stormid.unique()
        f_data = X_df[[field, "stormid"]]
        f_data[field] = (f_data[field] -
                         self.scaling_values.loc[field, "mean"]) / \
            self.scaling_values.loc[field, "std"]
        bloc = np.empty((len(X_df), self.len_sequences))
        for id in ids:
            ligne = f_data[f_data["stormid"] == id][field]
            seq_line = np.empty((len(ligne), self.len_sequences))
            for i in range(self.len_sequences):
                seq_line[:, self.len_sequences - 1 - i] = \
                    np.array(([padding_value] * i + list(ligne))[0:len(ligne)])
            pos = list(X_df["stormid"]).index(id)
            bloc[pos:pos + len(ligne), :] = seq_line
        return bloc

    def bearing(self, line: pd.Series) -> float:
        """
        Compute bearing in degrees for one line of dataframe

        Args:
            line   (pd.Series):  Row of dataframe with longitude, latitude,
                                 longitude_before and latitude_before fields

        Returns:
            float: bearing in degrees
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
        y = np.cos(lat1) * np.sin(lat2) - \
            np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

        bearing = np.degrees(np.arctan2(x, y))

        return bearing

    def compute_bearing(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bearing in degrees for all the lines of a dataframe

        Args:
            X_df    (pd.DataFrame): source DataFrame

        Returns:
            pd.DataFrame:   transformed dataframe
        """
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

    def cross_features(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features based on other features (differences, multiplications)

        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            pd.DataFrame:   transformed dataframe
        """
        X_df["mw12_imw"] = - X_df["max_wind_change_12h"] * \
            X_df["initial_max_wind"]
        if "mw12_imw" not in self.scalar_fields:
            self.scalar_fields.append("mw12_imw")
        X_df["diff_init_max"] = X_df["initial_max_wind"] - \
            X_df["max_wind_change_12h"]
        if "diff_init_max" not in self.scalar_fields:
            self.scalar_fields.append("diff_init_max")
        X_df["diff_current_init"] = X_df["windspeed"] - \
            X_df["initial_max_wind"]
        if "diff_current_init" not in self.scalar_fields:
            self.scalar_fields.append("diff_current_init")
        return X_df

    def compute_aggregate_image(self, X_df) -> pd.DataFrame:
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols]
            X_df = X_df.drop(f_cols, axis=1)
            X_df[field] = np.nanmean(f_data, axis=1)
            X_df[field].fillna(0, inplace=True)
            if field not in self.scalar_fields:
                self.scalar_fields.append(field)
        return X_df

    def fit(self, X_df: pd.DataFrame, y: np.ndarray = None) -> None:
        """
        Calculates mean and std values for each field. Image fields are
        regrouped before calculation.

        Args:
            X_df   (pd.DataFrame):   source DataFrame
            y        (np.ndarray):   source target

        Returns:
            None
        """
        logging.info("Starting FeatureExtractor fit")
        self.scaling_values.loc["windspeed", "mean"] = X_df["windspeed"].mean()
        self.scaling_values.loc["windspeed", "std"] = X_df["windspeed"].std()

        X_df = self.compute_bearing(X_df)
        X_df = self.cross_features(X_df)
        X_df = self.compute_aggregate_image(X_df)
        for field in self.scalar_fields:
            self.scaling_values.loc[field, "mean"] = X_df[field].mean()
            self.scaling_values.loc[field, "std"] = X_df[field].std()
        for field in self.constant_fields:
            self.scaling_values.loc[field, "mean"] = X_df[field].mean()
            self.scaling_values.loc[field, "std"] = X_df[field].std()
        for field in self.dummy_field:
            self.binarizer[field] = LabelBinarizer()
            self.binarizer[field].fit(X_df[field])
            self.num_dummy += len(self.binarizer[field].classes_)
        logging.info("Fitting FeatureExtractor done")

    def compute_scalar(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Processes scalar features, by making sequences and flattens them.

        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            np.ndarray:   Array of size
                        (len(X_df), self.len_sequences*len(self.scalar_fields))
        """
        scalar = self.compute_bearing(X_df)
        scalar = self.cross_features(scalar)
        scalar = self.compute_aggregate_image(scalar)
        scalar = scalar[self.scalar_fields]
        scalar["stormid"] = X_df["stormid"]
        final_scalar = np.empty((len(scalar), self.len_sequences,
                                 len(self.scalar_fields)))
        for field in self.scalar_fields:
            bloc = self.make_sequence(scalar, field)
            final_scalar[:, :, self.scalar_fields.index(field)] = bloc
        norm_scalar = np.nan_to_num(final_scalar)
        scalar_shape = self.len_sequences*len(self.scalar_fields)
        result = np.reshape(norm_scalar, (len(X_df), scalar_shape))
        logging.info("FeatureExtractor: scalar transform done")
        return result

    def compute_constant(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Processes constant featuresreshapes them. It also binarizes dummy
        fields.

        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            np.ndarray:   Array of constant features
        """
        norm_constant = X_df[self.constant_fields]
        for field in self.constant_fields:
            norm_constant[field] = (norm_constant[field] -
                                    self.scaling_values.loc[field, "mean"]) / \
                self.scaling_values.loc[field, "std"]
        norm_constant = norm_constant.values
        for field in self.dummy_field:
            dummy = self.binarizer[field].transform(X_df[field])
            norm_constant = np.concatenate((norm_constant, dummy), axis=1)
        result = np.copy(norm_constant)
        logging.info("FeatureExtractor: constant transform done")
        return result

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Processes all types of features and returns a flattened array. It is
        a more standard format if we want to use libraries such as sklearn, but
        the regressor will have to separate the different types of features in
        the fit and predict

        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            np.ndarray:   Array of constant features
        """
        logging.info("Starting FeatureExtractor transform")
        norm_scalar_final = self.compute_scalar(X_df)
        norm_constant_final = self.compute_constant(X_df)

        final = np.concatenate((norm_scalar_final,
                                norm_constant_final), axis=1)
        logging.info("FeatureExtractor transform done")
        return final
