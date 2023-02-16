# HELM on the go
Plug and play HELM

## How to launch
```bash
echo 'entries: [{description: "mmlu:subject=philosophy,model=geclm/gpt2_local", priority: 1}]' > run_specs.conf
python -m src.helm.benchmark.run --conf-paths run_specs.conf --local --suite v1 --max-eval-instances 10 -n 1` 
```
- Better conf file examples in `benchmark/presentation/*.conf`
- Need a gpt2_local folder at the root containing checkpoint + tokenizer (in transformers format atm)

### Core questions
- Eval:
	- For each eval, data is loaded from source URL (don't see how this manages distribution easily)
	- Each eval (scenario) has its own class
- Model:
	- Adding a new model = adding new classes 
		- in `helm/proxy/client` to connect to the model
		- in `helm/benchmark/window_services` for the tokenizer
		- also need to edit `helm/benchmark/windows_service/window_service_factory.py`, `helm/proxy/models.py`, `helm/proxy/clients/auto_clent.py`
- Running an eval:
	- run.py > generate specs from kwargs, then launch eval using a `Runner.run_all`, which, for each eval, 1) creates a scenario, preprocesses eval data, converts evaluation data to requests (`DataPreprocessor.preprocess`), applied in parallel to the model (`Executor.execute` does a `parallel_map`, which eats all logs!), then displays results in a fancy way


- Good things we want to take from this
	- A lot of perturbation and robustness code seems to work well
- What we need to do better
	- Data loading for evals is done from source URL and we want it to be streamed from an s3 bucket
	- Adding a new model is a pain - how can we build an easy interface for the suite to be launched "on the go"? (as long as models respect an interface contract, like having a `generate`) 
	- Parallelization is done at the eval level, maybe we want to launch several evals in parallel

### Repo organisation and some more notes 
- proxy > connect to the different model/hosting services APIs, load the model, have it generate predictions > most of the code there is useless for us
- common > utils (logger, auth management, )
- benchmark > contains the nice robustness code and the scenarios 

- Adapter adapt > convert to requests
- Executor execute > fill up results
	- Server_service > make_request
		- authenticate if needed
		- self.client.make_request (where the client links the correct API for hosting model) (seems to link seamlessly with HF hub)
	- uses parallel_map to get examples from models
	- huggingface_client (proxy/clients/huggingface_client)


