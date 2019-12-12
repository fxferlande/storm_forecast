import random


class GridSearch(object):
    def __init__(self, model, params, cv):
        self.model = model
        self.params = params
        self.keys = params.keys()
        self.cv = cv
        self.best_params = None

    def set_candidates(self):
        candidates = [{}]
        for key in list(self.params.keys()):
            old_candidates = candidates.copy()
            candidates = []
            for i in range(len(self.params[key])):
                for candidate in old_candidates:
                    new_candidate = candidate.copy()
                    new_candidate[key] = self.params[key][i]
                    candidates += [new_candidate]
        return candidates

    def fit(self, X, y, X_df):
        candidates = self.set_candidates()
        best_score = None
        history = []
        for candidate in candidates:
            print("Starting candidate {}".format(candidate))
            self.model.set_params(**candidate)
            cv_indexes = self.get_cv_index(X_df)
            score = 0
            for i in range(self.cv):
                train_index = cv_indexes[:i]+cv_indexes[i+1:]
                train_index = [x for sublist in train_index for x in sublist]
                val_index = cv_indexes[i]
                self.model.fit(X[train_index, ], y[train_index])
                score += self.model.score(X[val_index, ], y[val_index])
            score = score/self.cv
            if best_score is None or score < best_score:
                best_score = score
                self.best_params = candidate
                self.best_score = score
            history += [(candidate, score)]
        return history

    def get_best_candidate(self):
        return self.best_params, self.best_score

    def get_cv_index(self, X_df):
        """
        Besoin de regarder dans la dataframe originale la liste des index
        pour pouvoir séparer par tempêtes.
        """

        ids = list(X_df["stormid"].unique())
        random.shuffle(ids)

        indexes = ()
        group_size = int(len(ids)/self.cv)
        break_point = 0
        for i in range(self.cv):
            selection = ids[break_point: break_point+group_size]
            index_group = list(X_df[X_df["stormid"].isin(selection)].index)
            indexes += (index_group,)
            break_point += group_size

        return indexes
