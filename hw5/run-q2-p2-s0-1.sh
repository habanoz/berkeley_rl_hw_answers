# offline cql shifted reawrds
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

# hard env...
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1

python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 0
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_20000 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 1
