import sys
from functools import reduce
from typing import Tuple, List, Dict, Set


def parse_line(line: str) -> Tuple[List[str], str]:
    words = line.split()
    attributes = list(map(lambda x: x.lower().strip(), words[:-1]))
    klass = words[-1:][0].strip()
    return attributes, klass

def predict(preidctor, attrs):
    pass

if __name__ == '__main__':
    input = sys.stdin
    N, K = map(int, input.readline().split())
    features = list(map(lambda x: x.lower().strip(), input.readline().split()))
    features_set: Set = set([f.lower().strip() for f in features])
    features_len = len(features_set)
    feature_indices_to_name: Dict[int, str] = {i: f for i, f in enumerate(features)}
    feature_name_to_index: Dict[str, int] = {f: i for i, f in enumerate(features)}

    train_data: List[Tuple[List[str], str]] = []
    test_data: List[Tuple[List[str], str]] = []
    for i in range(N):
        train_data.append(parse_line(input.readline()))

    M = int(input.readline())
    for i in range(M):
        words = input.readline().split()
        attributes = list(map(lambda x: x.lower().strip(), words))
        test_data.append(attributes)

    predictor_tables: Dict[str, Dict[str, Dict[str, int]]] = {}
    for f in features_set:
        predictor_tables[f] = {}

    for attrs, klass in train_data:
        for f_i, feature in enumerate(attrs):
            if predictor_tables[feature_indices_to_name[f_i]].get(feature) is None:
                predictor_tables[feature_indices_to_name[f_i]][feature] = {klass: 1}
            else:
                if predictor_tables[feature_indices_to_name[f_i]][feature].get(klass) is None:
                    predictor_tables[feature_indices_to_name[f_i]][feature][klass] = 1
                else:
                    predictor_tables[feature_indices_to_name[f_i]][feature][klass] += 1


    for name, predictor in predictor_tables.items():
        for k, kk in predictor.items():
            if kk.get('0', 0) == kk.get('1', 0):
                predictor_tables[name][k] = '1'
            else:
                predictor_tables[name][k] = max(kk, key=kk.get)

    predictor_errors = {}
    for feature, predictor in predictor_tables.items():
        wrong_results = 0
        for attrs, klass in train_data:
            interesting_feature = attrs[feature_name_to_index[feature]]
            prediction = predictor_tables[feature][interesting_feature]
            wrong_results += 1 if prediction != klass else 0
        predictor_errors[feature] = wrong_results / len(train_data)


    # best_predictor = max(predictor_errors, key=predictor_errors.get)
    best_predictor = sorted(predictor_errors.items(), key=lambda x: (x[1], feature_name_to_index[x[0]]))[0][0]

    for attrs in test_data:
        interesting_feature = attrs[feature_name_to_index[best_predictor]]
        prediction = predictor_tables[best_predictor][interesting_feature]
        print(prediction)

