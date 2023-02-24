python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 2

python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_cfg0_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 2
