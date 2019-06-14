import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureExtractor(object):
    def __init__(self, len_sequences=5):
        self.scalar_fields = ['instant_t', 'windspeed', 'latitude', 'longitude',
                              'hemisphere', 'Jday_predictor', 'initial_max_wind',
                              'max_wind_change_12h', 'dist2land']
        self.spatial_fields = ["u", "v", "sst", "slp", "hum", "z", "vo700"]
        self.scaling_values = pd.DataFrame(index=self.spatial_fields,
                                           columns=["mean", "std"], dtype=float)
        self.scalar_norm = StandardScaler()
        self.max_len = len_sequences
        # Eventuellement fixer max_len dans le fit avec la distribution
        # self.max_len = max(X_df["stormid"].value_counts().quantile(0.5))

    def fit(self, X_df, y):
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11)
            field_grids.append(f_data)
        for f, field in enumerate(self.spatial_fields):
            self.scaling_values.loc[field, "mean"] = np.nanmean(field_grids[f])
            self.scaling_values.loc[field, "std"] = np.nanstd(field_grids[f])
        self.scalar_norm.fit(X_df[self.scalar_fields])

    def transform(self, X_df):
        field_grids = []
        ids = X_df.stormid.unique()
        len_sequences = X_df["stormid"].value_counts()
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[list(f_cols) + ["stormid"]]
            x = np.empty(shape=(len(X_df), self.max_len, 11, 11))
            x[:, :, :, :] = np.nan
            i = 0
            for id in ids:
                for index in range(1, len_sequences[id] + 1):
                    bloc = f_data[f_data["stormid"] == id][f_cols].iloc[max(
                        0, index - self.max_len):index]
                    x[i, self.max_len - len(bloc):self.max_len, :,
                      :] = bloc.values.reshape(-1, 11, 11)
                    i += 1
            to_append = (x - self.scaling_values.loc[field, "mean"]
                         ) / self.scaling_values.loc[field, "std"]
            field_grids.append(to_append)
            field_grids[-1][np.isnan(field_grids[-1])] = 0
            print("Field ", field, "just done")
        norm_data = np.stack(field_grids, axis=-1)

        scalar = self.scalar_norm.transform(X_df[self.scalar_fields])
        scalar = pd.DataFrame(scalar, columns=self.scalar_fields)
        scalar["stormid"] = X_df["stormid"]
        final_scalar = np.empty((len(scalar), self.max_len, len(self.scalar_fields)))
        for field in self.scalar_fields:
            f_data = scalar[[field, "stormid"]]
            bloc = np.empty((len(scalar), self.max_len))
            for id in ids:
                ligne = f_data[f_data["stormid"] == id][field]
                seq_line = np.empty((len(ligne), self.max_len))
                for i in range(self.max_len):
                    seq_line[:, self.max_len - 1 -
                             i] = np.array(([0] * i + list(ligne))[0:len(ligne)])
                pos = list(X_df["stormid"]).index(id)
                bloc[pos:pos + len(ligne), :] = seq_line
            final_scalar[:, :, self.scalar_fields.index(field)] = bloc
            print("Field ", field, "just done")
        norm_scalar = final_scalar

        return [norm_data, norm_scalar]
