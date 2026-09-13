

debug=0
debug_override=0
do_torch_compile=1
n_epochs=100

cond_used="vanilla"
cfm_used="cfm"
si_gamma_fn="sin_sq"

k_kept_ds=0
do_plot_override=0

ckpt_path="./huggingface_downloads/cfm_model/cfm_sin_sq_2/1999/_/lightning_logs/version_0/checkpoints/ckpt_100.ckpt"

python main.py separate_folder=run_inference \
  dataset=sat_gpm_f18 \
  cond_used=${cond_used} \
  cfm_used=${cfm_used} \
  si_gamma_fn=${si_gamma_fn} \
  ckpt_path=${ckpt_path} \
  batch_size=5 \
  n_epochs=${n_epochs} \
  save_every_k_epochs=10 \
  dataset.k_kept=${k_kept_ds} \
  debug=${debug} ++debug_override=${debug_override} \
  torch_compile=${do_torch_compile} \
  skip_to_test=1 \
  ++debug_override=${debug_override} \
  ++solver_override=dopri5 \
  ++nfe_override=1 \
  ++do_stochastic_noising_override=0 \
  ++t_end_override=0.5 \
  ++do_plot_override=${do_plot_override} \


