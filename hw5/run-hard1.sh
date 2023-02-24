# explore
# python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd --seed 0
# python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd --seed 1
# python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd --seed 2

# dqn
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0   --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0   --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0   --seed 2

# dqn scaled
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 2

# dqn online
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0   --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0   --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0   --seed 2

# dqn online scaled
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_dqn_online --use_rnd --unsupervised_exploration --cql_alpha=0 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0  --seed 2

# cql 0.1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed 2

# cql 0.3
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.3 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.3 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.3 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.3 --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.3 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.3 --seed 2

# cql 0.1 scaled
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 1
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --exp_name q2_cql_0.1_scaled --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_scale 100.0 --exploit_rew_shift 1.0 --seed 2
