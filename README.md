# HELM on the go
Plug and play HELM

## How to
1) Define specs file in `run_spec.conf`. 
	- Toy example: ` echo 'entries: [{description: "mmlu:subject=philosophy,model=geclm/gpt2_local", priority: 1}]' > run_specs.conf`
	- Better examples in `benchmark/presentation/*.conf`
2) From root: `python -m benchmark.run --conf-paths run_specs.conf --local --suite v1 --max-eval-instances 10 -n 1`