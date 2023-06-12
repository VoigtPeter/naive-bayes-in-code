from math import pi, exp, sqrt


class NaiveBayesClassifier:
    """
    """
    def __init__(self, laplace=1.0):
        """
        """
        self.laplace = laplace

    def fit(self, X, y):
        """
        """
        # get distinct classes
        distinct_classes = set(y)
        record_count = len(y)
        self.class_info = dict()
        for class_label in distinct_classes:
            c = y.count(class_label)
            self.class_info[class_label] = {
                'p': c / record_count,
                'c': c
            }
        # analyse column data
        self.attribute_info = dict()
        for i in range(len(X[0])):
            col = [row[i] for row in X]
            # get attribute info
            numeric = True
            for value in col:
                if not isinstance(value, (int, float)):
                    numeric = False
                    break
            self.attribute_info[i] = {
                'type': 'numeric' if numeric else 'nominal'
            }
            if numeric:
                self.attribute_info[i]['vals'] = dict()
                for v, c in zip(col, y):
                    if not self.attribute_info[i]['vals'].get(c):
                        self.attribute_info[i]['vals'][c] = {
                            'raw': []
                        }
                    self.attribute_info[i]['vals'][c]['raw'].append(v)
                for c in self.attribute_info[i]['vals']:
                    vals = self.attribute_info[i]['vals'][c]['raw']
                    size = len(vals)
                    mean = sum(vals) / size
                    variance = sum([((v - mean) ** 2) for v in vals]) / max(size - 1, 1)
                    self.attribute_info[i]['vals'][c]['mean'] = mean
                    self.attribute_info[i]['vals'][c]['variance'] = variance
            else:
                self.attribute_info[i]['vals'] = dict()
                for v, c in zip(col, y):
                    if not self.attribute_info[i]['vals'].get(v):
                        self.attribute_info[i]['vals'][v] = dict()
                    if self.attribute_info[i]['vals'][v].get(c, None) is None:
                        self.attribute_info[i]['vals'][v][c] = self.laplace
                    self.attribute_info[i]['vals'][v][c] += 1
                for v in self.attribute_info[i]['vals']:
                    for c in self.attribute_info[i]['vals'][v]:
                        self.attribute_info[i]['vals'][v][c] /= (self.class_info[c]['c'] + (self.laplace * len(distinct_classes)))
        return self

    def predict_proba(self, X):
        """
        """
        res = dict()
        for c in self.class_info:
            p_C = self.class_info[c]['p']
            p_A_C = 1
            for i, value in enumerate(X):
                if self.attribute_info[i]['type'] == 'numeric':
                    m = self.attribute_info[i]['vals'][c]['mean']
                    v = self.attribute_info[i]['vals'][c]['variance']
                    if v > 0:
                        p_density = (1 / sqrt(2*pi*v)) * exp(-(((value - m) ** 2) / (2 * v)))
                    else:
                        p_density = 0.1
                    p_A_C *= p_density
                else:
                    if value in self.attribute_info[i]['vals']:
                        p_A_C *= self.attribute_info[i]['vals'][value][c]
                    else:
                        # Laplace correction
                        p_A_C *= (1 / len(self.class_info.keys()))
            res[c] = p_A_C * p_C
        return res

    def predict(self, X):
        """
        """
        if isinstance(X[0], list):
            res = []
            for item in X:
                proba = self.predict_proba(item)
                res.append(max(proba, key=proba.get))
            return res
        else:
            proba = self.predict_proba(X)
            return max(proba, key=proba.get)
