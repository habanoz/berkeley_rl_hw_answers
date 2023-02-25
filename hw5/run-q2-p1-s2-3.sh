# online dqn shifted rewards
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd_shifted --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 2
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd_shifted --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 3

# offline dqn
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --seed 2
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --seed 3

# offline cql
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 2
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 3

# offline dqn shifted reawrds
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_dqn_shift --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 2
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_dqn_shift --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 3

# offline cql shifted reawrds
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_cql_shift --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 2
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --exp_name q2_cql_shift --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 3

