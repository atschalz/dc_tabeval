from modeling_helpers import get_submission
import numpy as np
import yaml

if __name__ == """__main__""":
    with open('configs/example.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    res = get_submission(configs)

    print(f"Performance:")
    print(f'Private Leaderboard: Score = {np.round(res["performance"]["Test"]["private_score"],4)}, LB position = Top {1-np.round(res["performance"]["Test"]["private_percentile"],4)}')
    print(f'Public Leaderboard: Score = {np.round(res["performance"]["Test"]["public_score"],4)}, LB position = Top {1-np.round(res["performance"]["Test"]["public_percentile"],4)}')
    print(f'Cross-validation: Score = {np.mean(list(res["performance"]["Val"].values())).round(4)} (+/-{np.std(list(res["performance"]["Val"].values())).round(4)})')
    


