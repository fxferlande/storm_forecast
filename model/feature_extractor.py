import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
pd.set_option('mode.chained_assignment', None)


class FeatureExtractor(object):
    def __init__(self):
        self.dummy_field = ["nature"]
        self.constant_fields = ['initial_max_wind', 'basin', 'instant_t',
                                'windspeed', 'Jday_predictor',
                                'max_wind_change_12h', 'dist2land']
        self.spatial_fields = ["u", "v", "sst", "slp", "hum", "z", "vo700"]
        self.binarizer = {}

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
        if "bearing_cos" not in self.constant_fields:
            self.constant_fields += ["bearing_cos", "bearing_sin"]
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
        if "mw12_imw" not in self.constant_fields:
            self.constant_fields.append("mw12_imw")
        X_df["diff_current_max"] = X_df["windspeed"] - \
            X_df["max_wind_change_12h"]
        if "diff_current_max" not in self.constant_fields:
            self.constant_fields.append("diff_current_max")
        X_df["diff_current_init"] = X_df["windspeed"] - \
            X_df["initial_max_wind"]
        if "diff_current_init" not in self.constant_fields:
            self.constant_fields.append("diff_current_init")
        return X_df

    def aggregate_image(self, X_df, field):
        f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
        f_data = X_df[f_cols]
        X_df = X_df.drop(f_cols, axis=1)
        X_df[field] = np.nanmean(f_data, axis=1)
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
        for field in self.dummy_field:
            self.binarizer[field] = LabelBinarizer()
            self.binarizer[field].fit(X_df[field])
        logging.info("Fitting FeatureExtractor done")

    def compute_image(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Processes images
        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            np.ndarray:   Array of size
                          (len(X_df), len(self.spatial_fields)*11*11)
        """
        for field in self.spatial_fields:
            X_df = self.aggregate_image(X_df, field)
        X_df = X_df[self.spatial_fields]
        X_df.fillna(0, inplace=True)
        logging.info("FeatureExtractor: image transform done")
        return X_df

    def compute_constant(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Processes constant featuresreshapes them. It also binarizes dummy
        fields.

        Args:
            X_df   (pd.DataFrame):   source DataFrame

        Returns:
            np.ndarray:   Array of constant features
        """
        norm_constant = self.compute_bearing(X_df)
        norm_constant = self.cross_features(norm_constant)
        norm_constant = X_df[self.constant_fields]

        for field in self.dummy_field:
            dummy = pd.DataFrame(self.binarizer[field].transform(X_df[field]))
            dummy.columns = field + dummy.columns.astype(str)
            norm_constant = pd.concat([norm_constant, dummy], axis=1)

        logging.info("FeatureExtractor: constant transform done")
        return norm_constant

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
        norm_image_final = self.compute_image(X_df)
        norm_constant_final = self.compute_constant(X_df)

        final = pd.concat([norm_image_final, norm_constant_final], axis=1)
        logging.info("FeatureExtractor transform done")
        return final
