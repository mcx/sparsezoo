`ModelDirectory` is a general class used to represent the directory with trained models. It exhaustively describes the model and enables further operations on the model including downloading, generating samples, validation, or inspection. Additionally, it serves as a "vessel" object that can be utilized to safely move the trained models between remote storage (servers) and local machines (client). 

`ModelDirectory` class object can be created using the rudimentary class object constructor, however, the intended way is to use two factory methods:

- `from_directory()` - that takes a local directory path (`str`) as an input
- `from_zoo_api` - that takes a list of dictionaries (structure returned by the API) as an input

During the initialization of the class object, the contents of the directory (or a list) will be parsed, so that we can match them to the attributes of the class. Those attributes represent the expected content, such as directory-type objects (e.g `directory`, `training`), as well as file-type objects(e.g. `model.onnx` or `model.md`). Note that at this stage no validation happens. This means that if no object is found, the respective attribute simply takes the value `None`.

Once created, one can call several public methods of the `ModelDirectory` class object:

1. `generate_outputs()` - this method can be used to automatically create output files, given the inference engine. There are two envisioned scenarios when one might use this function:


	- to validate the correctness of the model. Given the target `sample_outputs` data (that we are certain is correct), we can generate new outputs and compare them with the target to ensure that our model has not been changed at some point in the past.
	- to generate the target `sample_outputs` data. Once a new model is trained and we are planning to release it, the function may directly create `sample_outputs_{runtime_type}.tar.gz` folder, which will belong to the model directory stored on the server.

Note: to generate outputs, this class uses a helper class `InferenceRunner`.

2. `download()` - given that we created the class object using `from_zoo_api()` method, this method will automatically download the model directory from the server to our local machine

3. `validate()` - this method is used to validate the contents of the model directory. It shall be only called if there are files present on the local machine, otherwise, one should first call `download()` method to fetch them. The validation happens in two steps:

	- We validate the model by generating onnxruntime outputs from `sample_inputs` and comparing it with the target `sample_outputs`.
	- We call the helper class `IntegrationValidator` to check the structure of the model (presence and the names of the files).
	
I should call out that I have the hunch that this is part that I am going to spend some time polishing and making sure that the validation pipeline is exhaustive, covers all required cases, as well as all the files on the server properly adhere to the validation rules.

4. `analyze()` - this method is still not implemented. It is for now being developed by Kyle in isolation, but to my best knowledge, there will be a point in time when we integrate it into class. Long story short, it will be used to run analysis tools on the finished models, so that we have a nice overview of the model's performance, sparsification profile (e.g. which layers are pruned to which degree), etc.
